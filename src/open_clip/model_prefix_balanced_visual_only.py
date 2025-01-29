""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
import copy
import logging
import math
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import os 
import inspect
import numpy as np
import torch
import torch.nn.functional as F
import open_clip
from torch import nn
from PIL import Image
from torch.utils.checkpoint import checkpoint
from functools import partial
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from open_clip.transformer import VisionTransformer, LayerNormFp32,LayerNorm
from open_clip.model import  CLIPVisionCfg
from tqdm import tqdm 
from collections import defaultdict

def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)

def load_pkl_files_to_dict(directory):
    """
    Load all pickle (.pkl) files from a given directory into a single mega dictionary.
    Each .pkl file is assumed to contain a dictionary, and all dictionaries are merged.

    Args:
        directory (str): Path to the directory containing pickle files.

    Returns:
        dict: A merged dictionary containing all key-value pairs from all .pkl files.
              If there are duplicate keys, later files will overwrite earlier ones.
    """
    mega_dict = {}

    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                if not isinstance(data, dict):
                    raise ValueError(f"File {filename} does not contain a dictionary.")
                mega_dict.update(data)

    return mega_dict

def get_image_path(subsets, og_set):
    subset_with_image = {}
    for subset in subsets:
        for k in subset:
            image_path = og_set[k][1]
            label = og_set[k][0]
            subset_with_image[k] = [subset[k],image_path,(label-1)] #value is a list with the caption, image path,task label
    return subset_with_image

def get_data_balanced(train_all):
    reorg = defaultdict(lambda: defaultdict(list))

    for k in train_all:
        cap, img, label = train_all[k]
        reorg[img][label].append(k)

    train_all_reorg = {}
    i = 0
    while i < 1671835:
        for img in reorg:
            for label in reorg[img]:
                if len(reorg[img][label]) > 0:
                    idx = reorg[img][label][0]
                    train_all_reorg[i] = train_all[idx]
                    reorg[img][label].pop(0)
                    i+=1
    return train_all_reorg

class CLIPWrapper(nn.Module):
    def __init__(self, clip_model,num_tasks,clip_dim):
        super().__init__()  # Properly initialize the nn.Module superclass
        self.clip_model = clip_model
        self.task_embeddings = nn.Embedding(num_tasks, clip_dim)
    def extract_image_tokens(self,images):
        vision_trans = self.clip_model.visual
        images = vision_trans.conv1(images)  # shape = [*, width, grid, grid]\
        images = images.reshape(images.shape[0], images.shape[1], -1)  # shape = [*, width, grid ** 2]
        images = images.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        images = torch.cat([_expand_token(vision_trans.class_embedding, images.shape[0]).to(images.dtype), images], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        images = images + vision_trans.positional_embedding.to(images.dtype)
        
        return images
    
    def visual_transformer_forward_pass(self,x):
        x = self.clip_model.visual.patch_dropout(x)
        x = self.clip_model.visual.ln_pre(x)
        x = self.clip_model.visual.transformer(x)

        if self.clip_model.visual.attn_pool is not None:
            if self.clip_model.visual.attn_pool_contrastive is not None:
                # This is untested, WIP pooling that should match paper
                x = self.clip_model.visual.ln_post(x)  # TBD LN first or separate one after each pool?
                tokens = self.clip_model.visual.attn_pool(x)
                if self.clip_model.visual.attn_pool_type == 'parallel':
                    pooled = self.clip_model.visual.attn_pool_contrastive(x)
                else:
                    assert self.clip_model.visual.attn_pool_type == 'cascade'
                    pooled = self.clip_model.visual.attn_pool_contrastive(tokens)
            else:
                # this is the original OpenCLIP CoCa setup, does not match paper
                x = self.clip_model.visual.attn_pool(x)
                x = self.clip_model.visual.ln_post(x)
                pooled, tokens = self.clip_model.visual._global_pool(x)
        elif self.clip_model.visual.final_ln_after_pool:
            pooled, tokens = self.clip_model.visual._global_pool(x)
            pooled = self.clip_model.visual.ln_post(pooled)
        else:
            x = self.clip_model.visual.ln_post(x)
            pooled, tokens = self.clip_model.visual._global_pool(x)

        if self.clip_model.visual.proj is not None:
            pooled = pooled @ self.clip_model.visual.proj

        if self.clip_model.visual.output_tokens:
            return pooled, tokens
        
        return pooled
    
    def encode_image(self, image,tasks, normalize: bool = False):
        x = self.extract_image_tokens(image)
        tasks = self.task_embeddings(tasks)
        # print(x.shape)
        # print(tasks.shape)
        x = torch.cat((tasks.unsqueeze(1), x), dim=1)
        features = self.visual_transformer_forward_pass(x)
        return F.normalize(features, dim=-1) if normalize else features
    
    def forward(self,images,texts,tasks):
        # tasks = self.task_embeddings(tasks)
        image_features = self.encode_image(images,tasks, normalize=True) if images is not None else None
        text_features = self.clip_model.encode_text(texts, normalize=True) if texts is not None else None

        if self.clip_model.output_dict:
            out_dict = {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
            if self.clip_model.logit_bias is not None:
                out_dict['logit_bias'] = self.clip_model.logit_bias
            return out_dict

        if self.clip_model.logit_bias is not None:
            return image_features, text_features, self.clip_model.logit_scale.exp(), self.clip_model.logit_bias
        return image_features, text_features, self.clip_model.logit_scale.exp()

class CustomDataset(Dataset):
    def __init__(self, data_dict,preprocess,tokenizer, max_caption_length=77,transform=None):
        self.data = list(data_dict.values())
        self.max_caption_length = max_caption_length
        self.transform = transform or transforms.ToTensor()
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load and transform image
        caption, image_path, task_label = self.data[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.preprocess(image)
        task_label = torch.tensor(task_label, dtype=torch.long)

        return image,caption, task_label

def compute_loss(image_features, text_features, criterion, logit_scale):
    # Compute cosine similarity
    temperature = logit_scale
    logits_per_image = (image_features @ text_features.T) * temperature
    logits_per_text = (text_features @ image_features.T) * temperature
    labels = torch.arange(len(image_features), device=image_features.device)
    # logits = (image_features @ text_features.T)/temperature
    
    # Contrastive loss
    
    contrastive_loss = (criterion(logits_per_image, labels) + criterion(logits_per_text,labels))/2    
    
    return contrastive_loss
def recall_at_k(similarity_matrix, k):
    """
    Calculate Recall@k for a similarity matrix.

    Args:
    - similarity_matrix: 2D array of similarity scores (texts x images).
    - k: Rank threshold.

    Returns:
    - Recall@k as a float.
    """
    num_queries = similarity_matrix.shape[0]
    recalls = 0

    for query_idx in range(num_queries):
        # Get top-k indices for the query
        top_k_indices = torch.topk(similarity_matrix[query_idx], k, largest=True).indices.cpu().numpy()
        if query_idx in top_k_indices:  # Ground truth match is at index query_idx
            recalls += 1

    return recalls / num_queries

#load data 
one_train = load_pkl_files_to_dict("/home/as5957/vwp_metric/all_molmo_captions/train/one/")
two_train = load_pkl_files_to_dict("/home/as5957/vwp_metric/all_molmo_captions/train/two/")
three_train = load_pkl_files_to_dict("/home/as5957/vwp_metric/all_molmo_captions/train/three/")
four_train = load_pkl_files_to_dict("/home/as5957/vwp_metric/all_molmo_captions/train/four/")
five_train = load_pkl_files_to_dict("/home/as5957/vwp_metric/all_molmo_captions/train/five/")


one_val = load_pkl_files_to_dict("/home/as5957/vwp_metric/all_molmo_captions/val/one/")
two_val = load_pkl_files_to_dict("/home/as5957/vwp_metric/all_molmo_captions/val/two/")
three_val = load_pkl_files_to_dict("/home/as5957/vwp_metric/all_molmo_captions/val/three/")
four_val = load_pkl_files_to_dict("/home/as5957/vwp_metric/all_molmo_captions/val/four/")
five_val = load_pkl_files_to_dict("/home/as5957/vwp_metric/all_molmo_captions/val/five/")
with open("/home/as5957/vwp_metric/raw_prompts/train_prompts.pkl", 'rb') as file:
    train = pickle.load(file)

with open("/home/as5957/vwp_metric/raw_prompts/val_prompts.pkl", 'rb') as file:
    val = pickle.load(file)

train_all = get_image_path([one_train,two_train,three_train,four_train,five_train], train)
val_all = get_image_path([one_val,two_val,three_val,four_val,five_val], val)



# with open("./small_set.pkl", 'rb') as file:
#     train_items = pickle.load(file)

# Load pretrained CLIP model
batch_size = 512
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
train_dataset_1 = CustomDataset(train_all,preprocess,tokenizer) #change back to train_all
train_dataloader_1 = torch.utils.data.DataLoader(train_dataset_1, batch_size=batch_size, num_workers = 8, shuffle=True) 
# train_dataset = CustomDataset(train_items,preprocess,tokenizer)
# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True) 
train_all_reorg = get_data_balanced(train_all)
# train_all_reorg_items = {k:train_all_reorg[k] for k in list(train_all_reorg.keys())[:100]}
train_dataset_2 = CustomDataset(train_all_reorg,preprocess,tokenizer) #change back to train_all_reorg
train_dataloader_2 = torch.utils.data.DataLoader(train_dataset_2, batch_size=batch_size,num_workers = 8, shuffle=False) 

val_dataset = CustomDataset(val_all,preprocess,tokenizer)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,num_workers = 8, shuffle=False)
# val_dataset = CustomDataset(train_items,preprocess,tokenizer)
# val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=100,shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wrapped_model = CLIPWrapper(clip_model,5,768).to(device)

# Freeze text parameters
for param in wrapped_model.clip_model.transformer.parameters():
    param.requires_grad = False
for param in wrapped_model.task_embeddings.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam([
    {"params": wrapped_model.clip_model.parameters(), "lr": 1e-5},
    {"params": wrapped_model.task_embeddings.parameters(), "lr": 1e-4}
])

wrapped_model = wrapped_model.to(device)
criterion = nn.CrossEntropyLoss()

print("start training")
w_1,w_2 = 0.4,0.6 #weights for the losses 
best_val_loss = float('inf')  # Initialize best validation loss to a large value
patience = 3 # Number of epochs to wait for improvement
patience_counter = 0  # Counter for early stopping
best_model = None
for epoch in range(10):
    train_loss_1 = 0.0
    train_loss_2 = 0.0
    wrapped_model.train()  # Set model to training mode
    for images,texts, task_ids in tqdm(train_dataloader_1):
        images = images.to(device)
        texts = tokenizer(texts).to(device)
        task_ids = task_ids.to(device)
        optimizer.zero_grad()
        
        image_features, text_features, l_scale = wrapped_model(images,texts,task_ids)
        
        # Compute loss (e.g., contrastive loss)
        loss = compute_loss(image_features, text_features, criterion,l_scale)
        train_loss_1 += loss.item()
        loss.backward()
        optimizer.step()
    for images, texts, task_ids in tqdm(train_dataloader_2):
        images = images.to(device)
        texts = tokenizer(texts).to(device)
        task_ids = task_ids.to(device)
        optimizer.zero_grad()
        
        image_features, text_features, l_scale = wrapped_model(images,texts,task_ids)
        
        # Compute loss (e.g., contrastive loss)
        loss = compute_loss(image_features, text_features, criterion,l_scale)
        train_loss_2 += loss.item()
        loss.backward()
        optimizer.step()
    train_loss_total = w_1*train_loss_1 + w_2*train_loss_2
    avg_train_loss = train_loss_total / len(train_dataloader_1)
    print(f"Epoch {epoch+1}/100, Train Loss: {avg_train_loss:.4f}")

    # Validation loop
    wrapped_model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():
        for images,texts, task_ids in tqdm(val_dataloader):
            images = images.to(device)
            texts = tokenizer(texts).to(device)
            task_ids = task_ids.to(device)
            
            image_features, text_features, l_scale = wrapped_model(images,texts,task_ids)
        
            # Compute loss (e.g., contrastive loss)
            loss = compute_loss(image_features, text_features, criterion,l_scale)
            val_loss += loss.item()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity_mat = text_features @ image_features.T
    
    avg_val_loss = val_loss / len(val_dataloader)
    print(f"Epoch {epoch+1}/100, Val Loss: {avg_val_loss:.4f}")
    print(f"recall at k =1 scores: {recall_at_k(similarity_mat, 1)}")


    # Save model if validation loss improves
    if avg_val_loss < best_val_loss or epoch == 0:
        print(f"Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving model...")
        best_val_loss = avg_val_loss
        patience_counter = 0  # Reset patience counter
        # torch.save(wrapped_model.state_dict(), f"/home/as5957/vwp_metric/fine_tuned_clip/our_creative/clip_model_visual_only.pth")
        best_model = wrapped_model.state_dict()
    else:
        patience_counter += 1
        print(f"No improvement in validation loss. Patience counter: {patience_counter}/{patience}")
    
    # Stop saving after patience limit
    if patience_counter > patience:
        # torch.save(wrapped_model.state_dict(), f"/home/as5957/vwp_metric/fine_tuned_clip/our_creative/clip_model_visual_only.pth")
        print(f"Patience limit exceeded. No more saving at epoch {epoch}")
        break

