import torch
import os
import pickle
import open_clip
import numpy as np
from torch import nn
from PIL import Image
import torch.nn.functional as F
from collections import defaultdict
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

def compute_similarity(data, preprocess,wrapped_model,tokenizer, device, task = None, baseline = True):
    texts = []
    Dalle_imgs = []
    Complex_Dalle_imgs = []
    sim_maxes = []
   
    for text in list(data.keys()):
        texts.append(text)
        Dalle_imgs.append(preprocess(Image.open(data[text]['DALLE']).convert("RGB")))
        Complex_Dalle_imgs.append(preprocess(Image.open(data[text]['DALLE_CoT']).convert("RGB")))
    # Compute text embeddings
    with torch.no_grad():
        for i in range(0, len(texts)):        
            image_batch = torch.stack([Dalle_imgs[i],Complex_Dalle_imgs[i]]).to(device)  # Ensure tensor
            tasks = torch.full((image_batch.size(0),), task).to(device)
            texts_b = tokenizer(texts[i]).to(device)  # Tokenize and move to device

            if baseline == False:
                image_features = wrapped_model.encode_image(image_batch, tasks)  # Encode images
                text_features = wrapped_model.clip_model.encode_text(texts_b)  # Encode text
            else:
                image_features = wrapped_model.encode_image(image_batch)  # Encode images
                text_features = wrapped_model.encode_text(texts_b)  # Encode text
            
            text_features /= text_features.norm(dim=-1, keepdim=True)  # Normalize
            image_features /= image_features.norm(dim=-1, keepdim=True)  # Normalize
            similarity_matrix = text_features @ image_features.T
            max_similarity_idx = torch.argmax(similarity_matrix, dim=1)
            sim_maxes.append(max_similarity_idx.item())
        avg = np.mean(np.array(sim_maxes))
    
    return avg

def main():
    checkpt = sys.argv[1]
    baseline = sys.argv[2]
    metaphor_data_file = sys.argv[3]
    with open(metaphor_data_file, 'rb') as pickle_file:
        data = pickle.load(pickle_file)

    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    state_dict = torch.load(checkpt, weights_only=True) 
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if baseline == False:
        wrapped_model = CLIPWrapper(clip_model,5,5,768, method='third').to(device)
        wrapped_model.load_state_dict(state_dict)
        wrapped_model.to(device)

        print(compute_similarity(data, preprocess,wrapped_model,tokenizer, device, task = 0, baseline=False))
        print(compute_similarity(data, preprocess,wrapped_model,tokenizer, device,task = 1, baseline=False))
        print(compute_similarity(data, preprocess,wrapped_model,tokenizer, device,task = 2, baseline=False))
        print(compute_similarity(data, preprocess,wrapped_model,tokenizer, device,task = 3, baseline=False))
        print(compute_similarity(data, preprocess,wrapped_model,tokenizer, device,task = 4, baseline=False))
    else:
        wrapped_model = clip_model.to(device)
        print(compute_similarity(data, preprocess,wrapped_model,tokenizer, device)) #baseline

if __name__ == "__main__":
    main()
    