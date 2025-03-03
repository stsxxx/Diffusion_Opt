import torch
import time
import queue
import multiprocessing as mp
import pandas as pd
from diffusers import (
    StableDiffusion3Pipeline,
    FlowMatchEulerDiscreteScheduler,
)
from tqdm import tqdm
import os
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np
import re



image_directory = "/data6/stilex/large_small/nohit/"
image_paths = [os.path.join(image_directory, img_file) for img_file in os.listdir(image_directory) if img_file.endswith(('.png', '.jpg', '.jpeg'))]

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

strengths = [ 0.9, 0.8, 0.7, 0.6, 0.5]
def extract_prompt(filename):
    """Extract prompt from filename by replacing underscores with spaces."""
    name = os.path.splitext(os.path.basename(filename))[0]  # Remove extension
    return re.sub(r'[_]+', ' ', name).strip()  # Convert underscores to spaces

cached_latents = None

for i in tqdm(range(6666,10816)):
    path = image_paths[i]
    prompt = extract_prompt(path) 
    init_image = Image.open(path).convert("RGB") 
    for strength in strengths:
        latent = pipe.get_latents(prompt=prompt,
                            image=init_image,
                            strength= strength,
                            guidance_scale=7.5,
                            negative_prompt=None)
        if cached_latents is None:
            cached_latents = latent.cpu().clone()
        else:
            cached_latents = torch.cat((cached_latents,latent.cpu().clone()), dim=0)
    
# Print final size
print(f"Final cached_latents size: {cached_latents.shape}")  # Expected shape: (num_images * num_strengths, C, H, W)

# Save latents
save_path = "cached_latents_3.pt"
torch.save(cached_latents.cpu(), save_path)
print(f"Saved cached latents to {save_path}")
