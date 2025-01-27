import pickle
import os 
import numpy as np
import torch
import json
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

class CLIPWrapper:
    def __init__(self, clip_model):
        self.clip_model = clip_model

    def encode_image(self, image, task_ids, normalize: bool = False):
        features = self.clip_model.visual(image, task_ids)
        return F.normalize(features, dim=-1) if normalize else features

    def __getattr__(self, name):
        if name == "encode_image":
        # Do not delegate encode_image
            return getattr(self, name)

        return getattr(self.clip_model, name)


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

        
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 2, 768))

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

def compute_loss(image_features, text_features, criterion, temperature=0.07):
    # Compute cosine similarity
    labels = torch.arange(len(image_features), device=image_features.device)
    logits = (image_features @ text_features.T) / temperature
    
    # Contrastive loss
    
    contrastive_loss = (criterion(logits, labels) + criterion(logits.T,labels))/2    
    
    return contrastive_loss

#load data 
# one_train = load_pkl_files_to_dict("/home/as5957/vwp_metric/all_molmo_captions/train/one/")
# two_train = load_pkl_files_to_dict("/home/as5957/vwp_metric/all_molmo_captions/train/two/")
# three_train = load_pkl_files_to_dict("/home/as5957/vwp_metric/all_molmo_captions/train/three/")
# four_train = load_pkl_files_to_dict("/home/as5957/vwp_metric/all_molmo_captions/train/four/")
# five_train = load_pkl_files_to_dict("/home/as5957/vwp_metric/all_molmo_captions/train/five/")




# Load pretrained CLIP model
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
# train_dataset = CustomDataset(train_items,preprocess,tokenizer)
# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)



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


state_dict = torch.load("/home/as5957/vwp_metric/fine_tuned_clip/our_creative_full/clip_model.pth",weights_only=True)
clip_model.load_state_dict(state_dict)

clip_model = clip_model.to(device)
criterion = nn.CrossEntropyLoss()

# Training loop
# for epoch in range(10):
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
#         loss = compute_loss(image_features, text_features,criterion)
#         train_loss += loss.item()
#         loss.backward()
#         optimizer.step()
#     avg_train_loss = train_loss / len(train_dataloader)
#     print(f"Epoch {epoch+1}/{10}, Train Loss: {loss.item()}")

#     # Validation loop
#     clip_model.eval()
#     val_loss = 0.0
#     with torch.no_grad():
#         for images, texts, task_ids in val_dataloader:
#             images = images.to(device)
#             texts = tokenizer(texts).to(device)
#             task_ids = task_ids.to(device)
            
#             image_features = clip_model.visual(images, task_ids)
#             text_features = clip_model.encode_text(texts)
            
#             loss = compute_loss(image_features, text_features,criterion)
#             val_loss += loss.item()
    
#     avg_val_loss = val_loss / len(val_dataloader)
    
#     print(f"Epoch {epoch+1}/{10}, Val Loss: {avg_val_loss:.4f}")

json_file = '/home/as5957/vwp_metric/evaluation_corpora/multim_poem.json'
all_images = os.listdir('/mnt/swordfish-pool2/ananya/evaluation_corpora/multim_poem')
image_ids = []
for img in all_images:
        image_ids.append(img.split(".")[0])

with open(json_file, 'r') as file:
        data = json.load(file)

dict_form = {}
for d in data:
        dict_form[d['id']] = d

image_paths = []
text = []
inds = []
task_ids = []
task = 0
for i,img in tqdm(enumerate(all_images)):
    task_ids.append(task)
    img_id = img.split(".")[0]
    path = f'/mnt/swordfish-pool2/ananya/evaluation_corpora/multim_poem/{img_id}.jpg'
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

        print(task_ids_b.shape)  # Debug shape of task IDs

        # Extract features
        image_features = clip_model.visual(images_b, task_ids_b)
        text_features = clip_model.encode_text(texts_b)

        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Update similarity matrix
        similarity_matrix[batch_start:batch_end, batch_start:batch_end] = text_features @ image_features.T


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

print("Recall@1:", recall_at_k(similarity_matrix, 1))
print("Recall@5:", recall_at_k(similarity_matrix, 5))
print("Recall@10:", recall_at_k(similarity_matrix, 10))
print("Recall@20:", recall_at_k(similarity_matrix, 20))

        
    
    

