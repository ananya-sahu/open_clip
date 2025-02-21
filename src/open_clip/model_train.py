""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
import copy
import sys
import logging
import math
import pickle
import random
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
from torch.utils.data import Dataset,Sampler
from torchvision import transforms
from open_clip.transformer import VisionTransformer, LayerNormFp32,LayerNorm
from open_clip.model import  CLIPVisionCfg
from tqdm import tqdm 
from collections import defaultdict

def _expand_token(token, batch_size: int): #function from model.py 
    return token.view(1, 1, -1).expand(batch_size, -1, -1)

def load_pkl_files_to_dict(directory):
    # Loads all pickle (.pkl) files from a given directory into a single mega dictionary.
    # Args:
    #     directory (str): Path to the directory containing pickle files.
    # Returns:
    #     dict: A merged dictionary containing all key-value pairs from all .pkl files.

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
    while i < len(train_all):
        for img in reorg:
            for label in reorg[img]:
                if len(reorg[img][label]) > 0:
                    idx = reorg[img][label][0]
                    train_all_reorg[i] = train_all[idx]
                    reorg[img][label].pop(0)
                    i+=1
    return train_all_reorg

class CLIPWrapper(nn.Module): #adapted from model.py
    def __init__(self, clip_model,num_tasks,num_embeds,clip_dim,method = 'first'):
        super().__init__()  # Properly initialize the nn.Module superclass
        self.clip_model = clip_model
        # self.task_embeddings = nn.Embedding(num_tasks, clip_dim)
        self.task_embeddings = nn.ModuleList([nn.Embedding(num_tasks, clip_dim) for _ in range(num_embeds)])
        self.method = method

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
    
    def encode_image(self, image,tasks,mode = 'first', normalize: bool = False):
        x = self.extract_image_tokens(image)
        # tasks = self.task_embeddings(tasks)
        task_emebds = torch.stack([embed(tasks) for embed in self.task_embeddings], dim=1)
        if mode == 'first':
            x = torch.cat((task_emebds, x), dim=1)
        if mode == 'second':
            cls_token, image_tokens = x[:, :1, :], x[:, 1:, :]  # CLS token and remaining patches
            # Step 4: Insert task embeddings in between
            x = torch.cat([cls_token, task_emebds, image_tokens], dim=1)
        if mode == 'third':
             x = torch.cat((x,task_emebds), dim=1)
        features = self.visual_transformer_forward_pass(x)
        return F.normalize(features, dim=-1) if normalize else features
    
    def forward(self,images,texts,tasks):
        # tasks = self.task_embeddings(tasks)
        image_features = self.encode_image(images,tasks,mode = self.method, normalize=True) if images is not None else None
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

class CustomDataset(Dataset): #custom dataset loader for our dataset
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

class ShuffledBatchSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_batches = len(data_source) // batch_size
        self.batch_indices = [list(range(i * batch_size, (i + 1) * batch_size)) 
                              for i in range(self.num_batches)]

    def __iter__(self):
        # Shuffle the batches (order of batches)
        random.shuffle(self.batch_indices)

        for batch in self.batch_indices:
            # Shuffle samples inside each batch
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches

def compute_loss(image_features, text_features, criterion, logit_scale):
    # Compute cosine similarity
    temperature = logit_scale
    logits_per_image = (image_features @ text_features.T) * temperature
    logits_per_text = (text_features @ image_features.T) * temperature
    labels = torch.arange(len(image_features), device=image_features.device)
    contrastive_loss = (criterion(logits_per_image, labels) + criterion(logits_per_text,labels))/2    
    
    return contrastive_loss

def recall_at_k(similarity_matrix, k):
   
    # Calculate Recall@k for a similarity matrix.

    # Args:
    # - similarity_matrix: 2D array of similarity scores (texts x images).
    # k  = rank threshold.
    # Returns:
    # float
 
    num_queries = similarity_matrix.shape[0]
    recalls = 0

    for query_idx in range(num_queries):
        top_k_indices = torch.topk(similarity_matrix[query_idx], k, largest=True).indices.cpu().numpy()
        if query_idx in top_k_indices:  
            recalls += 1

    return recalls / num_queries

def main():
    #load data 
    train_paths = sys.argv[1:6]
    val_paths =  sys.argv[6:11]
    train_file = sys.argv[11]
    val_file = sys.argv[12]
    save_path = sys.argv[13]
    
    all_train_subsets = [load_pkl_files_to_dict(subset_train) for subset_train in train_paths]
    all_val_subsets = [load_pkl_files_to_dict(subset_val) for subset_val in val_paths]

    with open(train_file, 'rb') as file:
        train = pickle.load(file)

    with open(val_file, 'rb') as file:
        val = pickle.load(file)

    train_all = get_image_path(all_train_subsets, train)
    val_all = get_image_path(all_val_subsets, val)

    # Load pretrained CLIP model
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    train_dataset_1 = CustomDataset(train_all,preprocess,tokenizer)
    train_dataloader_1 = torch.utils.data.DataLoader(train_dataset_1, batch_size=512, num_workers = 8, shuffle=True) 
    train_all_reorg = get_data_balanced(train_all)
    train_dataset_2 = CustomDataset(train_all_reorg,preprocess,tokenizer)
    batch_sampler = ShuffledBatchSampler(train_dataset_2, batch_size = 512)
    train_dataloader_2 = torch.utils.data.DataLoader(train_dataset_2, batch_sampler=batch_sampler,num_workers = 8)
    val_dataset = CustomDataset(val_all,preprocess,tokenizer)
    batch_sampler2 = ShuffledBatchSampler(val_dataset, batch_size = 512)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,batch_sampler=batch_sampler2, num_workers = 8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapped_model = CLIPWrapper(clip_model,5,768, method='second').to(device)

    # Freeze all parameters except task embeddings
    for param in wrapped_model.clip_model.parameters():
        param.requires_grad = False
    for param in wrapped_model.task_embeddings.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam([
        {"params": wrapped_model.task_embeddings.parameters(), "lr": 1e-4}
    ])

    wrapped_model = wrapped_model.to(device)
    criterion = nn.CrossEntropyLoss()
    print("start training")

    w_1,w_2 = 0.6,0.4 #weights for the losses 
    best_val_loss = float('inf')  # Initialize best validation loss to a large value
    patience = 3 # Number of epochs to wait for improvement
    patience_counter = 0  # Counter for early stopping
    best_model = None
    best_recall = 0
    for epoch in range(10):
        train_loss = 0.0
        wrapped_model.train()  # Set model to training mode
        for (images1,texts1, task_ids1),(images2,texts2, task_ids2) in tqdm(zip(train_dataloader_1,train_dataloader_2)):
            images1 = images1.to(device)
            texts1 = tokenizer(texts1).to(device)
            task_ids1 = task_ids1.to(device)

            images2 = images2.to(device)
            texts2 = tokenizer(texts2).to(device)
            task_ids2 = task_ids2.to(device)
            
            optimizer.zero_grad()

            image_features1, text_features1, l_scale1 = wrapped_model(images1, texts1, task_ids1)
            loss1 = compute_loss(image_features1, text_features1, criterion, l_scale1)
            loss = w_1 * loss1
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            optimizer.zero_grad()
            image_features2, text_features2, l_scale2 = wrapped_model(images2, texts2, task_ids2)
            loss2 = compute_loss(image_features2, text_features2, criterion, l_scale2)
            loss = w_2 * loss2
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss  / len(train_dataloader_1)
        print(f"Epoch {epoch+1}/100, Train Loss: {avg_train_loss:.4f}")

        # Validation loop
        wrapped_model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        avg_recall = 0.0
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
                avg_recall += recall_at_k(similarity_mat, 1)
        
        avg_val_loss = val_loss / len(val_dataloader)
        avg_recall = avg_recall/len(val_dataloader)
        print(f"Epoch {epoch+1}/100, Val Loss: {avg_val_loss:.4f}")
        print(f"recall at k =1 scores: {avg_recall}")


        # Save model if validation loss improves
        if (avg_val_loss < best_val_loss and avg_recall > best_recall) or epoch == 0:
            print(f"Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving model...")
            best_val_loss = avg_val_loss
            avg_recall = best_recall
            patience_counter = 0  # Reset patience counter
            torch.save(wrapped_model.state_dict(), save_path)
            best_model = wrapped_model.state_dict()
        else:
            patience_counter += 1
            print(f"No improvement in validation loss. Patience counter: {patience_counter}/{patience}")
        
        # Stop saving after patience limit
        if patience_counter > patience:
            torch.save(best_model, save_path)
            print(f"Patience limit exceeded. No more saving at epoch {epoch}")
            break

if __name__ == "__main__":
    main()