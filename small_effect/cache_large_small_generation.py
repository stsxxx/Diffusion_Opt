import torch
from diffusers import StableDiffusion3Pipeline
from diffusers import StableDiffusionXLPipeline

from huggingface_hub import login


import pandas as pd
import math
from typing import List, Tuple, Any
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, FlowMatchEulerDiscreteScheduler
# from pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
# from scheduling_euler_discrete import EulerDiscreteScheduler
from datasets import load_dataset
import argparse
from PIL import Image
from torchvision import models, transforms
from torch.nn.functional import adaptive_avg_pool2d
import numpy as np
from scipy import linalg
import skimage.metrics
import os
from transformers import CLIPProcessor, CLIPModel
import queue
from queue import Queue
import faiss
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image


parser = argparse.ArgumentParser(description="Stable diffusion test")
parser.add_argument("--bs", type=int, help="Batch size", default=0)
parser.add_argument("--np", type=bool, help="Enable negative prompt or not", default=False)
parser.add_argument("--eps", type=float, help="eps for DBSCAN", default=0.3)
parser.add_argument("--wd", type=float, help="time window size", default=1)
parser.add_argument("--gpus", type=int, help="number of gpus", default=16)

np.random.seed(42)



if torch.cuda.is_available():
    print(f"GPU is available. Number of GPUs: {torch.cuda.device_count()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    print("GPU is not available.")
    
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)

processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


def get_clip_score(images, text):

    inputs = processor(text=text, images=images, return_tensors="pt", truncation=True, padding=True,max_length=77)
    # print(inputs)

    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # print(outputs)
 
    logits_per_image = outputs.logits_per_image.max(dim=1)[0]
    # print(logits_per_image, logits_per_image.shape)  # 1,4
    # probs = logits_per_image.softmax(dim=1)
    # mean_score = torch.mean(logits_per_image,dim=0)
    # print(f"average CLIP:{mean_score}")
    return logits_per_image

def precompute_timesteps_for_labels(scheduler, labels, device):
    timesteps = []
    for label in labels:
        if label == 0 or label == 2:
            # For label 0, use full 50 iterations
            scheduler.set_timesteps(num_inference_steps=50, device=device)
            timesteps.append(scheduler.timesteps.tolist())
        elif label == 1:
            # For label 1, use last 40 iterations
            scheduler.set_timesteps(num_inference_steps=50, device=device)
            timesteps.append(scheduler.timesteps[-40:].tolist())         
        else:
            timesteps.append([])  

    return timesteps

def precompute_timesteps_for_labels_35(scheduler, labels, device, index):
    timesteps = []
    for label in labels:
        if label == 0:
            # For label 0, use full 50 iterations
            scheduler.set_timesteps(num_inference_steps=50, device=device)
            timesteps.append(scheduler.timesteps.tolist())
        elif label == 1:

            scheduler.set_timesteps(num_inference_steps=50, device=device)
            timesteps.append(scheduler.timesteps[index:].tolist())         
        else:
            timesteps.append([])  

    return timesteps

def create_full_path(example):
    """Create full path to image using `base_path` to COCO2017 folder."""
    example["image_path"] = os.path.join(PATH_TO_IMAGE_FOLDER, example["file_name"])
    return example

def add_poisson_seconds(group, lam=30):
    # Generate Poisson-distributed random offsets with mean `lam`, capped at 60 seconds
    random_offsets = np.minimum(np.random.poisson(lam=lam, size=len(group)), 60)
    return group + random_offsets

PATH_TO_IMAGE_FOLDER = "/data3/llama_model/yuchen/datasets/COCO2017"
args = parser.parse_args()



# ## model, scheduler and generator
# model_id = "stabilityai/stable-diffusion-xl-base-1.0"
# print(torch.version.cuda) 
# # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
# # pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe_xl = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16  # Use fp16 for efficiency if on GPU
)
pipe_xl = pipe_xl.to("cuda")  # Move to GPU if available

# # pipe.unet = UNet2DConditionModel.from_config(pipe.unet.config).to(torch.float16)
# pipe = pipe.to(device)
# torch.cuda.empty_cache()
# scheduler_xl = EulerDiscreteScheduler.from_config(pipe_xl.scheduler.config)
# full_timesteps_flow = precompute_timesteps_for_labels(scheduler,[0],"cpu")[0]
# print(full_timesteps_flow)
# full_timesteps = precompute_timesteps_for_labels(scheduler_xl,[0],"cpu")[0]
# cached_timestep = full_timesteps[k-1]
# print(len(full_timesteps))
# print("cached timestep:", cached_timestep)
# closest_index = min(range(len(full_timesteps_flow)), key=lambda i: abs(full_timesteps_flow[i] - cached_timestep))
# closest_index = k-1

# cached_timestep_3 = full_timesteps_flow[closest_index]
# # Output the result
# print(f"The closest value to {cached_timestep} is {full_timesteps_flow[closest_index]} at index {closest_index}.")


strengths = [ 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
ks = [ 5, 10, 15, 20, 25, 30]
clip_score_dict = {}
source_clip = {}

seed = 42 #any
generator = torch.Generator(device).manual_seed(seed)

embedding_dim = 768  # CLIP model output dimension
index = faiss.IndexFlatL2(embedding_dim)  # Using FAISS for ANN search

image_directory = "/data3/llama_model/yuchen/datasets/MJHQ/large1"

# Get a list of all image file paths in the directory
image_paths = [os.path.join(image_directory, img_file) for img_file in os.listdir(image_directory) if img_file.endswith(('.png', '.jpg', '.jpeg'))]

print(len(image_paths))

def extract_prompt(filename):
    """Extract prompt from filename by replacing underscores with spaces."""
    name = os.path.splitext(os.path.basename(filename))[0]  # Remove extension
    return re.sub(r'[_]+', ' ', name).strip()  # Convert underscores to spaces
# print(sorted_df['timestamp'].iloc[200000])
for i in tqdm(range(len(image_paths)), desc="Processing Requests"):
    path = image_paths[i]
    prompt = extract_prompt(path) 
    init_image = Image.open(path).convert("RGB") 
    clean_prompt = re.sub(r'[^\w\-_\.]', '_', prompt)[:300]
    for idx, strength in enumerate(strengths):
        model_ouputs_xl = pipe_xl(prompt=prompt,
                    image=init_image,
                    strength=strength,
                    guidance_scale=7.5,
                    negative_prompt=None,
                    height = 1024,
                    width = 1024,
                    generator=generator)
            # end_xl = time.time()
        # print("xl time:", end_xl - start_xl)
        
        filename_xl = f"/data3/llama_model/yuchen/datasets/MJHQ/small_{ks[idx]}/{clean_prompt}.png"
        try:
            model_ouputs_xl.images[0].save(filename_xl)
        except Exception as e:
            print(f"Failed to save image: {e}")
            print("image name:", prompt)