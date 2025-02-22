import torch
import os
import open_clip
from torch import nn
from PIL import Image
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
import sys

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
        task_emebds = torch.stack([embed(tasks) for embed in self.task_embeddings], dim=1)
        if mode == 'first':
            x = torch.cat((task_emebds, x), dim=1)
        if mode == 'second':
            cls_token, image_tokens = x[:, :1, :], x[:, 1:, :]  # CLS token and remaining patches
            x = torch.cat([cls_token, task_emebds, image_tokens], dim=1)
        if mode == 'third':
             x = torch.cat((x,task_emebds), dim=1)
        features = self.visual_transformer_forward_pass(x)
        return F.normalize(features, dim=-1) if normalize else features
    
    def forward(self,images,texts,tasks):
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

def extract_texts_and_images(root_dirs,image_extensions={".jpg", ".jpeg", ".png", ".bmp", ".gif"}):
    """
    Args:
        root_dir (str): Path to the main directory of Metaphors.
    Returns:
        dict: Mapping of text labels to lists of image paths.
    """
    images_to_text = {}
    for root_dir in root_dirs:
        for text_label in os.listdir(root_dir):
            text_path = os.path.join(root_dir, text_label)
            if os.path.isdir(text_path):  
                for file in os.listdir(text_path):
                    file_path = os.path.join(text_path, file)
                    if os.path.isfile(file_path) and file.lower().endswith(tuple(image_extensions)):
                        images_to_text[file_path] = text_label
    return images_to_text



def compute_similarity(images_to_text, wrapped_model,task_num, tokenizer, device, preprocess,batch_size=512, base=True):
    """
    Computes the similarity matrix between all text labels and all images.
    """
    texts = []
    images = []

    for img_path in images_to_text:
        texts.append(images_to_text[img_path][1])
        images.append(img_path)
    num_texts = len(texts)    
    text_features_list = []
    image_features_list = []

    # Compute text embeddings
    with torch.no_grad():
        for batch_start in tqdm(range(0, num_texts, batch_size)):
            batch_end = min(batch_start + batch_size, num_texts)
            text_batch = texts[batch_start:batch_end]
            texts_b = tokenizer(text_batch).to(device)  # Tokenize and move to device
            if base==True:
                text_features = wrapped_model.encode_text(texts_b)  # Encode text
            else:
                text_features = wrapped_model.clip_model.encode_text(texts_b)  # Encode text
            text_features /= text_features.norm(dim=-1, keepdim=True)  # Normalize
            
            text_features_list.append(text_features)
        
            text_features = torch.cat(text_features_list, dim=0)  # Combine batches
            #preprcess the images:
            images_batch =  images[batch_start:batch_end]
            images_batch = [preprocess(Image.open(file_path).convert("RGB")) for file_path in images_batch]
            image_batch = torch.stack(images_batch).to(device)  # Ensure tensor
            tasks = torch.full((image_batch.size(0),), task_num).to(device) 
            if base==True:
                image_features = wrapped_model.encode_image(image_batch)
            else:
                image_features = wrapped_model.encode_image(image_batch, tasks)  # Encode images
            image_features /= image_features.norm(dim=-1, keepdim=True)  # Normalize
            
            image_features_list.append(image_features)

    image_features = torch.cat(image_features_list, dim=0)  # Combine batches
    text_features = torch.cat(text_features_list, dim=0)
    similarity_matrix = image_features @ text_features.T
    
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
    """
    num_queries = similarity_matrix.shape[0]
    total_rank = 0
    ranks = []
    for query_idx in range(num_queries):
        # Get sorted indices (descending order since higher similarity is better)
        sorted_indices = torch.argsort(similarity_matrix[query_idx], descending=True).cpu().numpy()
        
        # Find the rank of the correct match (query_idx)
        rank = (sorted_indices == query_idx).nonzero()[0][0] + 1 
        ranks.append(rank)
        total_rank += rank
    return total_rank / num_queries

def get_results(paths,wrapped_model,tokenizer, device,preprocess, dists = None, base=True):
    image_to_text_mapping = extract_texts_and_images(paths)
    if base==True:
        similarity_matrix= compute_similarity(image_to_text_mapping, wrapped_model,0,tokenizer, device,preprocess, batch_size=512,base=True)
        print("average rank: "+ str(average_rank(similarity_matrix)))
    else:
        for dist in dists:
            print(f"results at distance {dist}: ")
            print(f"distance {dist}")
            similarity_matrix= compute_similarity(image_to_text_mapping, wrapped_model,dist,tokenizer, device,preprocess, batch_size=512,base=False)
            print("recall @ k = 1: "+ str(recall_at_k(similarity_matrix, 1)))
            print("recall @ k = 5: "+ str(recall_at_k(similarity_matrix, 5)))
            print("recall @ k = 10: "+ str(recall_at_k(similarity_matrix,10)))
            print("recall @ k = 20: "+ str(recall_at_k(similarity_matrix, 20)))
            print("average rank: "+ str(average_rank(similarity_matrix)))


def main():
    baseline = sys.argv[1] #true or false for baseline clip or our clip model
    checkpt = sys.argv[2] #none or the checkpoint of our model
    root_dirs = sys.argv[3:9] # the root directories containing the metaphor image directories 

    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if baseline == True:
        wrapped_model = clip_model.to(device)
        get_results(root_dirs,wrapped_model,tokenizer, device,preprocess, dists = None, base=True)
    
    else:
        state_dict = torch.load(checkpt, weights_only=True) 
        wrapped_model = CLIPWrapper(clip_model,5,5,768, method='third').to(device)
        wrapped_model.load_state_dict(state_dict)
        wrapped_model.to(device)
        get_results(root_dirs,wrapped_model,tokenizer, device,preprocess,dists = [0,1,2,3,4], base=True)

if __name__ == "__main__":
    main()