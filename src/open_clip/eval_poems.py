import pickle
import os 
import numpy as np
import torch
import json
import sys
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
from PIL import Image
from tqdm import tqdm


class CLIPWrapper(nn.Module):
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
def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)

def get_sim(all_images, dict_form,preprocess, tokenizer,device, wrapped_model,img_dir_path, task=None, baseline=True):
    image_paths = []
    text = []
    inds = []
    task_ids = []
    for i,img in tqdm(enumerate(all_images)):
        if ".jpg" in img:
            if task:
                task_ids.append(task)
            img_id = img.split(".")[0]
            path = f'{img_dir_path}/{img_id}.jpg'
            image_paths.append(preprocess(Image.open(path).convert("RGB")))
            inds.append(i)
            text.append(dict_form[int(img_id)]['poem'])
    num_texts = len(text)
    num_images = len(image_paths)
    similarity_matrix = torch.zeros((num_texts, num_images)).to(device)
    batch_size = 512
    with torch.no_grad():
        for batch_start in range(0, num_images, batch_size):
            batch_end = min(batch_start + batch_size, num_images)
            
            # Batch slicing
            task_batch = task_ids[batch_start:batch_end]
            text_batch = text[batch_start:batch_end]
            image_batch = image_paths[batch_start:batch_end]
            
            # Transfer to device
            images_b = torch.stack(image_batch).to(device)  # Ensure image_batch is a tensor
            texts_b = tokenizer(text_batch).to(device)  # Tokenize and move to device
            task_ids_b = torch.tensor(task_batch).to(device)  # Convert to tensor and move to device

            if baseline == False:
                image_features, text_features, l_scale = wrapped_model(images_b,texts_b,task_ids_b)
            else:
                image_features, text_features, l_scale = wrapped_model(images_b,texts_b)
        
            # Compute loss (e.g., contrastive loss)
            
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            # Update similarity matrix
            similarity_matrix[batch_start:batch_end, batch_start:batch_end] = text_features @ image_features.T
    return similarity_matrix

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

def average_rank(similarity_matrix):
    """
    Calculate the Average Rank of the correct match in a similarity matrix.

    Args:
    - similarity_matrix: 2D array of similarity scores (texts x images).

    Returns:
    - Average Rank as a float.
    """
    num_queries = similarity_matrix.shape[0]
    total_rank = 0
    for query_idx in range(num_queries):
        sorted_indices = torch.argsort(similarity_matrix[query_idx], descending=True).cpu().numpy()
        
        # Find the rank of the correct match (query_idx)
        rank = (sorted_indices == query_idx).nonzero()[0][0] + 1  
        total_rank += rank

    return total_rank / num_queries


def main():

    # Load pretrained CLIP model
    baseline = sys.argv[1] #true or false indicating if evaluting our model or baseline clip 
    checkpt  = sys.argv[2] #None or the checkpoint path of our model
    json_file= sys.argv[3] #path of MM poem dataset json file
    img_dir = sys.argv[4] #where MM poem images are stored
    task = sys.argv[5] #None if baseline otherwise int vals from 0 to 4 indicating distance levels 1 to 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if checkpoint:
        state_dict = torch.load(checkpt,weights_only=True)
        wrapped_model = CLIPWrapper(clip_model,5,5,768, method='third').to(device)
        wrapped_model.load_state_dict(state_dict)
        wrapped_model.to(device)
    else:
        wrapped_model = clip_model.to(device)

    all_images = os.listdir(img_dir)
    image_ids = []
    for img in all_images:
            image_ids.append(img.split(".")[0])

    with open(json_file, 'r') as file:
            data = json.load(file)

    dict_form = {}
    for d in data:
            dict_form[d['id']] = d

    similarity_matrix = get_sim(all_images, dict_form,preprocess, tokenizer,device, wrapped_model,img_dir, task=task, baseline=baseline)
    print("Recall@1:", recall_at_k(similarity_matrix, 1))
    print("Recall@5:", recall_at_k(similarity_matrix, 5))
    print("Recall@10:", recall_at_k(similarity_matrix, 10))
    print("Recall@20:", recall_at_k(similarity_matrix, 20))
    print("avg rank: " + str(average_rank(similarity_matrix)))

if __name__ == "__main__":
    main()




        
    
    

