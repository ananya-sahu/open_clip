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

# class CLIPWrapper:
#     def __init__(self, clip_model):
#         self.clip_model = clip_model

#     def encode_image(self, image, task_ids, normalize: bool = False):
#         features = self.clip_model.visual(image, task_ids)
#         return F.normalize(features, dim=-1) if normalize else features

#     def __getattr__(self, name):
#         if name == "encode_image":
#         # Do not delegate encode_image
#             return getattr(self, name)

#         return getattr(self.clip_model, name)


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
        if len(caption) > self.max_caption_length:  
            caption = caption[:self.max_caption_length]  # Truncate if too long
        elif len(caption) < self.max_caption_length:
            caption = caption.ljust(self.max_caption_length, ' ')  # Pad if too short
        # Convert task label to tensor
        task_label = torch.tensor(task_label, dtype=torch.long)

        return image,caption, task_label


class CustomVisionTransformer(VisionTransformer):
    def __init__(self, num_tasks: int = 5,task_embedding_dim=768, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_embeddings = nn.Embedding(num_tasks, task_embedding_dim)
        scale = 768 ** -0.5 #width = 768 here

        #taken from the way positonal token embedding initialized in original open_clip but adds in one more token for the task
        #embedding
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 2, 768))

    # def forward(self, x: torch.Tensor, task_ids: torch.Tensor):
    #     # x = self.conv1(x)
    #     # x = x.reshape(x.shape[0], x.shape[1], -1)
    #     # x = x.permute(0, 2, 1)

    #     # task_tokens = self.task_embeddings(task_ids).unsqueeze(1)
    #     # x = torch.cat([task_tokens, x], dim=1)

    #     # x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
    #     # x = x + self.positional_embedding.to(x.dtype)

    #     # x = self.patch_dropout(x)
    #     # x = self.ln_pre(x)
    #     # x = self.transformer(x)
    #     print(x.shape)
    #     task_tokens = self.task_embeddings(task_ids).unsqueeze(1)
    #     print(task_tokens.shape)
    #     x = torch.cat([task_tokens, x], dim=1)

    #     return super().forward(x)
    def forward(self, x: torch.Tensor, task_ids: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        task_tokens = self.task_embeddings(task_ids).unsqueeze(1)
        # class embeddings and positional embeddings
        class_token = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype)],dim=1)
      
        x = torch.cat([class_token, task_tokens, x], dim=1)
     
        # x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        x = self.patch_dropout(x)
        x = self.ln_pre(x)
        x = self.transformer(x)

        if self.attn_pool is not None:
            if self.attn_pool_contrastive is not None:
                # This is untested, WIP pooling that should match paper
                x = self.ln_post(x)  # TBD LN first or separate one after each pool?
                tokens = self.attn_pool(x)
                if self.attn_pool_type == 'parallel':
                    pooled = self.attn_pool_contrastive(x)
                else:
                    assert self.attn_pool_type == 'cascade'
                    pooled = self.attn_pool_contrastive(tokens)
            else:
                # this is the original OpenCLIP CoCa setup, does not match paper
                x = self.attn_pool(x)
                x = self.ln_post(x)
                pooled, tokens = self._global_pool(x)
        elif self.final_ln_after_pool:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)
        else:
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)

        if self.proj is not None:
            pooled = pooled @ self.proj

        if self.output_tokens:
            return pooled, tokens
        
        return pooled

def compute_loss(image_features, text_features, criterion, logit_scale):
    # Compute cosine similarity
    temperature = logit_scale.exp()
    logits_per_image = (image_features @ text_features.T) * temperature
    logits_per_text = (text_features @ image_features.T) * temperature
    labels = torch.arange(len(image_features), device=image_features.device)
    # logits = (image_features @ text_features.T)/temperature
    
    # Contrastive loss
    
    contrastive_loss = (criterion(logits_per_image, labels) + criterion(logits_per_text,labels))/2    
    
    return contrastive_loss

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


# Load pretrained CLIP model
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
train_dataset = CustomDataset(train_all,preprocess,tokenizer)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=512, num_workers = 8, shuffle=True) 
# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True) 

val_dataset = CustomDataset(val_all,preprocess,tokenizer)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=512,num_workers = 8, shuffle=True)

vision_heads = 768 // 64
vision_config = {
    'image_size': clip_model.visual.image_size,
    'patch_size': clip_model.visual.patch_size,
    "width": 768,
    "layers": 12,  # or (2, 2, 6, 2) for ResNet-like configurations
    "heads": vision_heads,
    "mlp_ratio": 4.0,
    "ls_init_value": None,  # Layer scale initial value
    "patch_dropout": 0.0,  # Fraction of patches to drop during training
    "attentional_pool": False,  # Use attentional pooling in the last embedding layer
    "attn_pooler_queries": 256,  # Number of queries for attentional pooler
    "attn_pooler_heads": 8,  # Number of heads for attentional pooling
    "no_ln_pre": False,  # Disable pre-transformer LayerNorm
    "final_ln_after_pool": False,  # Apply final LayerNorm after pooling
    "pool_type": "tok",  # Pooling type
    "pos_embed_type": "learnable",  # Type of positional embeddings
    "output_tokens": False,  # Whether to output all tokens or just pooled features,
    'output_dim': 512,
    'act_layer': nn.GELU,
    'norm_layer':LayerNorm
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# wrapped_model = CLIPWrapper(clip_model).to(device)
# wrapped_model.visual = CustomVisionTransformer(num_tasks=5, **vision_config)
# print(wrapped_model.visual)
clip_model.visual = CustomVisionTransformer(num_tasks=5, **vision_config)

# Freeze all parameters except task embeddings
# for param in clip_model.parameters():
#     param.requires_grad = False
for param in clip_model.visual.task_embeddings.parameters():
    param.requires_grad = True

# Set up optimizer (only for task embeddings)
optimizer = torch.optim.Adam(clip_model.parameters(), lr=1e-4)
# task_embeds_params = [param.name for param in clip_model.visual.task_embeddings.parameters()]
# other_params = [
#     param for param in clip_model.parameters()
#         if param.name not in task_embeds_params
# ]
# optimizer = torch.optim.Adam([
#     {"params": clip_model.visual.task_embeddings.parameters(), "lr": 1e-4},
#     {"params": other_params, "lr": 1e-4}
# ])

clip_model = clip_model.to(device)
criterion = nn.CrossEntropyLoss()
logit_scale = clip_model.logit_scale
print("here")

# Training loop
# for epoch in range(100):
#     train_loss = 0.0
#     clip_model.train() 
#     for images, texts, task_ids in train_dataloader:
#         images = images.to(device)
#         texts = tokenizer(texts).to(device)
#         task_ids = task_ids.to(device)
#         optimizer.zero_grad()
        
#         image_features = clip_model.visual(images, task_ids)
#         text_features = clip_model.encode_text(texts)
        
#         # Compute loss (e.g., contrastive loss)
#         loss = compute_loss(image_features, text_features,criterion,logit_scale)
#         train_loss += loss.item()
#         loss.backward()
#         optimizer.step()
#     avg_train_loss = train_loss / len(train_dataloader)
#     print(f"Epoch {epoch+1}/{10}, Train Loss: {loss.item()}")

#     # Validation loop
#     clip_model.eval()
#     val_loss = 0.0
#     with torch.no_grad():
#         for images, texts, task_ids in train_dataloader:
#             images = images.to(device)
#             texts = tokenizer(texts).to(device)
#             task_ids = task_ids.to(device)
            
#             image_features = clip_model.visual(images, task_ids)
#             text_features = clip_model.encode_text(texts)
            
#             loss = compute_loss(image_features, text_features,criterion, logit_scale)
#             val_loss += loss.item()
    
#     avg_val_loss = val_loss / len(train_dataloader)
    
#     print(f"Epoch {epoch+1}/{10}, Val Loss: {avg_val_loss:.4f}")
# torch.save(clip_model.state_dict(), f"/home/as5957/vwp_metric/fine_tuned_clip/our_creative_full/clip_model.pth")

best_val_loss = float('inf')  # Initialize best validation loss to a large value
patience = 3 # Number of epochs to wait for improvement
patience_counter = 0  # Counter for early stopping
best_model = None
for epoch in range(10):
    train_loss = 0.0
    clip_model.train()  # Set model to training mode
    for images, texts, task_ids in tqdm(train_dataloader):
        images = images.to(device)
        texts = tokenizer(texts).to(device)
        task_ids = task_ids.to(device)
        optimizer.zero_grad()
        
        image_features = clip_model.visual(images, task_ids)
        text_features = clip_model.encode_text(texts)
        
        # Compute loss (e.g., contrastive loss)
        loss = compute_loss(image_features, text_features, criterion,logit_scale)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    avg_train_loss = train_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/100, Train Loss: {avg_train_loss:.4f}")

    # Validation loop
    clip_model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():
        for images, texts, task_ids in tqdm(val_dataloader):
            images = images.to(device)
            texts = tokenizer(texts).to(device)
            task_ids = task_ids.to(device)
            
            image_features = clip_model.visual(images, task_ids)
            text_features = clip_model.encode_text(texts)
            
            loss = compute_loss(image_features, text_features, criterion,logit_scale)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_dataloader)
    print(f"Epoch {epoch+1}/100, Val Loss: {avg_val_loss:.4f}")

    # Save model if validation loss improves
    if avg_val_loss < best_val_loss or epoch == 0:
        print(f"Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving model...")
        best_val_loss = avg_val_loss
        patience_counter = 0  # Reset patience counter
        torch.save(clip_model.state_dict(), f"/home/as5957/vwp_metric/fine_tuned_clip/our_creative_full/clip_model.pth")
        best_model = clip_model.state_dict()
    else:
        patience_counter += 1
        print(f"No improvement in validation loss. Patience counter: {patience_counter}/{patience}")
    
    # Stop saving after patience limit
    if patience_counter > patience:
        torch.save(best_model, f"/home/as5957/vwp_metric/fine_tuned_clip/our_creative_full/clip_model.pth")
        print(f"Patience limit exceeded. No more saving at epoch {epoch}")
        break

