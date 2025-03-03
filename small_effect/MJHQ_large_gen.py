import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusion3Pipeline
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, FlowMatchEulerDiscreteScheduler
import re

# **Paths**
meta_data_path = "/data3/llama_model/yuchen/datasets/MJHQ/meta_data.json"
dataset_base_path = "/data3/llama_model/yuchen/datasets/MJHQ/"
generated_images_path = os.path.join(dataset_base_path, "large1")  # Store generated images
original_images_path = os.path.join(dataset_base_path, "og")      # Store copies of original images

if torch.cuda.is_available():
    print(f"GPU is available. Number of GPUs: {torch.cuda.device_count()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    print("GPU is not available.")
    
# **Load SD 3.5 Model**
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16
).to("cuda")
scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)

seed = 42 #any
generator = torch.Generator(device).manual_seed(seed)

# **Ensure Save Directories Exist**
os.makedirs(generated_images_path, exist_ok=True)
os.makedirs(original_images_path, exist_ok=True)

# **Load Meta Data**
with open(meta_data_path, "r") as f:
    meta_data = json.load(f)
meta_data_subset = dict(list(meta_data.items())[28000:28200])

def precompute_timesteps_for_labels_35(scheduler, labels, device, index):
    timesteps = []
    for label in labels:
        if label == 0:
            # For label 0, use full 50 iterations
            scheduler.set_timesteps(num_inference_steps=50, device=device)
            timesteps.append(scheduler.timesteps.tolist())
        elif label == 1:
            # For label 1, use last 40 iterations
            scheduler.set_timesteps(num_inference_steps=50, device=device)
            timesteps.append(scheduler.timesteps[index:].tolist())         
        else:
            timesteps.append([])  

    return timesteps

# **Generate & Copy Images**
for image_id, data in tqdm(meta_data_subset.items(), desc="Processing Requests"):
    prompt = data["prompt"]
    category = data["category"]

    # **Construct Original Image Path**
    original_image_path = os.path.join(dataset_base_path, category, f"{image_id}.jpg")
    if not os.path.exists(original_image_path):
        print(f"Warning: {original_image_path} not found. Skipping.")
        continue

    # **Save Generated Image**

    # generated_image.save(os.path.join(generated_images_path, f"{image_id}.png"))

    timesteps_batch = precompute_timesteps_for_labels_35(scheduler, [0], "cpu",0)[0]

    og_image = Image.open(original_image_path).convert("RGB")
    prompt_embeds, pooled_prompt_embeds, latents = pipe.input_process(prompt = prompt,negative_prompt = None, generator=generator, callback_on_step_end=None,
        callback_on_step_end_tensor_inputs=["latents"], height=1024, width=1024)

    model_outputs = pipe(prompt = prompt, prompt_embeds = prompt_embeds, pooled_prompt_embeds = pooled_prompt_embeds, generator=generator, callback_on_step_end=None,
            callback_on_step_end_tensor_inputs=["current_latents"], timesteps_batch = timesteps_batch,
            cached_timestep=None ,labels_batch = 0, current_latents=latents, height=1024, width=1024)


    clean_prompt = re.sub(r'[^\w\-_\.]', '_', prompt)[:210]
    filename = os.path.join(generated_images_path, f"{clean_prompt}.png") 
    filename_og = os.path.join(original_images_path, f"{clean_prompt}.png") 
    try:
        model_outputs[1][0].save(filename)
        og_image.save(filename_og)
    except Exception as e:
        print(f"Failed to save image: {e}")
        print("image name:", prompt)


print("✅ All images processed and saved.")
