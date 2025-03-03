import torch
import time
import queue
import multiprocessing as mp
import pandas as pd
from diffusers import StableDiffusion3Pipeline, StableDiffusionXLImg2ImgPipeline
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, FlowMatchEulerDiscreteScheduler
import numpy as np
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import re

import os
from tqdm import tqdm

import numpy as np
import pandas as pd
import numpy as np


class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint, min_output, max_output):
        self.Kp = Kp  
        self.Ki = Ki  
        self.Kd = Kd  
        self.setpoint = setpoint  
        self.min_output = min_output  
        self.max_output = max_output 
        self.prev_error = 0  
        self.integral = 0  

    def compute(self, measured_value, dt):
        error = self.setpoint - measured_value  
        self.integral += error * dt  
        derivative = (error - self.prev_error) / dt if dt > 0 else 0 
        self.prev_error = error  

        # Compute PID output
        
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        
        output = np.clip(output, -self.max_output, self.max_output)
        return  output 

def generate_rapidly_increasing_seconds_from_start(num_requests, min_rate=2, max_rate=9, duration=100*60):
    """
    Generate request timestamps with a smoothly increasing request rate using a sigmoid function.

    Args:
        num_requests (int): Total number of requests.
        min_rate (float): Minimum request rate (req/min).
        max_rate (float): Maximum request rate (req/min).
        duration (int): Total duration in seconds.

    Returns:
        np.array: Array of `seconds_from_start` values.
    """
    # Convert rates to requests per second
    min_rate_per_sec = min_rate / 60
    max_rate_per_sec = max_rate / 60

    # Adjust sigmoid input range to ensure full scaling from min_rate to max_rate
    x = np.linspace(-2, 6, num_requests)  # Increase range to get a full sigmoid transition

    # Apply sigmoid transformation
    sigmoid_growth = 1 / (1 + np.exp(-1.5 * x))

    # Normalize sigmoid values to scale precisely between min_rate_per_sec and max_rate_per_sec
    request_rates = min_rate_per_sec + (max_rate_per_sec - min_rate_per_sec) * sigmoid_growth

    # Compute interarrival times inversely proportional to the rate
    interarrival_times = 1 / np.maximum(request_rates, 1e-3)  # Avoid division by zero

    # Cumulatively sum interarrival times to get timestamps
    seconds_from_start = np.cumsum(interarrival_times)

    return seconds_from_start
# def generate_rapidly_increasing_seconds_from_start(num_requests, min_rate=1, max_rate=10, duration=100*60):
#     """
#     Generate request timestamps where the request rate increases rapidly from a nonzero starting point.
    
#     - Starts at `min_rate` req/min and increases quadratically to `max_rate` req/min over `duration` seconds.
#     - Ensures a smaller initial gap and accelerates request frequency more aggressively.

#     Args:
#         num_requests (int): Total number of requests.
#         min_rate (float): Minimum starting request rate in req/min.
#         max_rate (float): Maximum request rate in req/min.
#         duration (int): Total duration in seconds.

#     Returns:
#         np.array: Array of `seconds_from_start` values.
#     """
#     # Convert rates to requests per second
#     min_rate_per_sec = min_rate / 60  
#     max_rate_per_sec = max_rate / 60  

#     # Generate time intervals
#     time_intervals = np.linspace(0, duration, num_requests)

#     # Quadratic increase from min_rate to max_rate
#     request_rates = min_rate_per_sec + ((max_rate_per_sec - min_rate_per_sec) / duration**2) * time_intervals**2

#     # Compute interarrival times inversely proportional to the rate
#     interarrival_times = 1 / np.maximum(request_rates, 1e-3)  # Avoid division by zero

#     # Cumulatively sum interarrival times to get timestamps
#     seconds_from_start = np.cumsum(interarrival_times)

#     return np.array(seconds_from_start)




def add_poisson_seconds(group, lam=30):
    # Generate Poisson-distributed random offsets with mean `lam`, capped at 60 seconds
    random_offsets = np.minimum(np.random.poisson(lam=lam, size=len(group)), 60)
    return group + random_offsets

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
def load_model(model_type, device):
    """Load the specified model type onto the given device."""
    if model_type == "sd3.5":
        return StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16
        ).to(device)
    elif model_type == "sdxl":
        return StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to(device)

def retrieve_best_image(text_embedding, final_image_embeddings, distribution, agg_k_distribution):
    """Retrieve the cached image with the highest cosine similarity and determine cache hit."""
    with torch.no_grad():
        text_norm = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        image_norm = final_image_embeddings / final_image_embeddings.norm(dim=-1, keepdim=True)
        similarity_scores = torch.matmul(text_norm, image_norm.T)
        similarity_scores = torch.clamp(similarity_scores, min=0)

    similarity_scores = similarity_scores.cpu().numpy().flatten()
    ranked_indices = np.argsort(-similarity_scores)
    highest_clip_score_index = ranked_indices[0]
    highest_clip_score = similarity_scores[highest_clip_score_index]

    k = None
    strength = None
    if highest_clip_score >= 0.3:
        k = 30
        strength = 0.4
        distribution[k] += 1
        agg_k_distribution[k] += 1 
    elif highest_clip_score >= 0.29:
        k = 25
        strength = 0.5
        distribution[k] += 1
        agg_k_distribution[k] += 1
    elif highest_clip_score >= 0.28:
        k = 15
        strength = 0.7
        distribution[k] += 1
        agg_k_distribution[k] += 1
    elif highest_clip_score >= 0.27:
        k = 10
        strength = 0.8
        distribution[k] += 1
        agg_k_distribution[k] += 1
    elif highest_clip_score >= 0.25:
        k = 5
        strength = 0.9
        distribution[k] += 1
        agg_k_distribution[k] += 1

    return highest_clip_score_index, highest_clip_score, k, strength, agg_k_distribution

def request_scheduler(
    hit_queue, nonhit_queue, sorted_df, start_time, request_rate_queue, final_image_embeddings, 
    cached_image_paths, shared_new_images, lock,  model, processor,worker_status,  cache_size=100000,  log_file="request_throughput_1000_PID.csv"
):
    """Fetch requests based on timestamps and push them to the appropriate queue based on cache retrieval."""
    request_count = 0
    hit_count = 0  # To record the number of hit requests
    k_distribution = {5: 0, 10: 0, 15: 0, 25: 0, 30: 0}  # Track requests by k values
    agg_k_distribution = {5: 0, 10: 0, 15: 0, 25: 0, 30: 0} 
    device = "cuda:0"
    minute = 0
    with open(log_file, "w") as f:
        f.write("timestamp,request_rate,throughput\n")

    last_check_time = time.time()
    last_check_time_queue = last_check_time
    request_count_per_min = 0
    size_of_queues = 0
    throughput = 0
    
    for _, row in sorted_df.iterrows():
        while time.time() - start_time < row['seconds_from_start']:
            time.sleep(0.5)
            
        request_arrival_time = time.time()  # Start time of request processing
        row['start_time'] = request_arrival_time  # Add start_time to request
        # Process new image paths from shared_new_images
        batch_new_image_paths = []
        while not shared_new_images.empty():
            batch_new_image_paths.append(shared_new_images.get())
        if batch_new_image_paths:
            new_images = [Image.open(image_path).convert("RGB") for image_path in batch_new_image_paths]
        else:
            new_images = []
        if new_images:
            new_image_tensors = processor(images=new_images, return_tensors="pt").to(device)

            with torch.no_grad():
                new_image_embeddings = model.get_image_features(**new_image_tensors)

            # Concatenate new embeddings and add new paths
            final_image_embeddings = torch.cat((final_image_embeddings, new_image_embeddings), dim=0)
            cached_image_paths.extend(batch_new_image_paths)

        # Evict old entries if cache exceeds the size
        if final_image_embeddings.size(0) > cache_size:
            # Calculate the number of items to evict
            num_to_evict = final_image_embeddings.size(0) - cache_size

            # Remove the oldest embeddings and cached paths
            final_image_embeddings = final_image_embeddings[num_to_evict:]
            cached_image_paths = cached_image_paths[num_to_evict:]

        prompt = [row['prompt']]
        texts = processor(text=prompt, return_tensors="pt", truncation=True, padding=True, max_length=77).to(device)
        with torch.no_grad():
            text_embedding = model.get_text_features(**texts)

        best_index, best_score, k, strength, agg_k_distribution= retrieve_best_image(text_embedding, final_image_embeddings, k_distribution, agg_k_distribution)
        print(agg_k_distribution)
        if k is not None:
            hit_count += 1  # Increment hit count
            row['retrieved_image_index'] = best_index
            row['retrieved_image_score'] = best_score
            row['retrieved_image_path'] = cached_image_paths[best_index]  # Retrieve path based on best index
            row['k'] = k
            row['strength'] = strength
            hit_queue.put(row.to_dict())
        else:
            row['retrieved_image_index'] = None
            row['retrieved_image_score'] = best_score
            row['retrieved_image_path'] = None
            row['k'] = None
            row['strength'] = None
            nonhit_queue.put(row.to_dict())
        
        request_count += 1
        request_count_per_min += 1
        current_time = time.time()
        
        if current_time - last_check_time_queue >= 60:
            elapsed_time = current_time - last_check_time_queue
            minute += 1
            new_size_of_queues =  nonhit_queue.qsize() + hit_queue.qsize()
            throughput = size_of_queues + request_count_per_min - new_size_of_queues
            size_of_queues = new_size_of_queues
            with open(log_file, "a") as f:
                f.write(f"{minute},{request_count_per_min / elapsed_time * 60},{throughput / elapsed_time * 60}\n")
            request_count_per_min = 0
            last_check_time_queue = current_time
            
        if current_time - last_check_time >= 180:
            elapsed_time = current_time - last_check_time
            request_rate = (request_count / elapsed_time * 60) if elapsed_time > 0 else 0
            hit_rate = hit_count / request_count if request_count > 0 else 0
            k_rates = {k: count / hit_count for k, count in k_distribution.items()}  # Normalize k counts

            # Combine all metrics into a single dictionary
            metrics = {
                "request_rate": request_rate,
                "hit_rate": hit_rate,
                "k_rates": k_rates
            }

            request_rate_queue.put(metrics)  # Send metrics to the global monitor
            request_count = 0
            hit_count = 0
            for k in k_distribution.keys():
                k_distribution[k] = 0  # Reset k distribution
            last_check_time = current_time
    
    while True:
        if hit_queue.empty() and nonhit_queue.empty():
            # Check if all workers have finished or dropped
            all_done = all(status in ["finished", "dropped"] for status in worker_status.values())
            if all_done:
                print("[Scheduler] All workers have finished. Terminating.")
                break  # Exit scheduler loop

        time.sleep(1)  # Prevent busy waiting
            

def global_monitor(request_rate_queue, control_queues, num_gpus, total_num_steps=50):
    """Monitor global request rate and dynamically decide model allocation."""
    large_model_throughput = 1 / (52.5/60)  # Requests/min
    small_model_throughput = 1 / (10.2/60)  # Requests/min
    Kp, Ki, Kd = 0.6, 0.01, 0.05  # PID tuning parameters (adjusts model allocation)
    pid_controller = PIDController(Kp, Ki, Kd, setpoint=0, min_output=-num_gpus, max_output=num_gpus)
    num_of_large = num_gpus
    prev_time = time.time()  # Track time between PID updates
    
    while True:
        try:
            metrics = request_rate_queue.get(timeout=10)
            request_rate = metrics["request_rate"]
            hit_rate = metrics["hit_rate"]
            k_rates = metrics["k_rates"]
            k_rate = 0
            print(f"[Global Monitor] Request rate: {request_rate:.2f} req/min, Hit rate: {hit_rate:.2f}")
            print(f"[Global Monitor] k distribution: {k_rates}")
            
            current_time = time.time()
            dt = (current_time - prev_time) / 60  # Time since last adjustment
            prev_time = current_time
            
            # Calculate workload for large and small models
            nonhit_workload = (1 - hit_rate) * request_rate
            for k, rate in k_rates.items():
                print(f"Rate for k={k}: {rate:.2%}")
                k_rate += rate * (1 - k / total_num_steps)
            hit_workload = hit_rate * request_rate * k_rate
            
            # Minimum number of large models to handle non-hit workload
            min_large_models = int(np.ceil(nonhit_workload / large_model_throughput))

            # Iterate to find the largest possible num_large
            num_large = min_large_models
            print("min large model:", num_large)
            while num_large <= num_gpus:
                # Check if this allocation satisfies the hit workload constraint
                available_throughput = (
                    num_large * large_model_throughput
                    - nonhit_workload
                    + (num_gpus - num_large) * small_model_throughput
                )
                if 0.9 * available_throughput >= hit_workload:
                    num_large += 1  # Try the next larger allocation
                    continue
                else:
                    num_large -= 1  # Step back to the largest valid configuration
                    break
            # Ensure the number of large models does not exceed the total GPUs
            num_large = max(1, min(num_large, num_gpus))
            print("exact number of large:", num_large)
            pid_controller.setpoint = num_large
            num_large_add = pid_controller.compute(num_of_large, dt)
            print("delta:",num_large_add )
            num_of_large += num_large_add
            print("real number of large:",num_of_large )
            
            N_large = int(round(num_of_large))
            print("allocated number of large:",N_large )
            
            N_large = max(1, min(N_large, num_gpus))
            N_small = num_gpus - N_large
            print(f"[Global Monitor] Allocating {N_large} large models and {N_small} small models.")

            # Allocate models
            model_allocation = ["sd3.5"] * N_large + ["sdxl"] * N_small
            for q in control_queues:
                q.put(model_allocation)
        except queue.Empty:
            continue


def worker(gpu_id, hit_queue, nonhit_queue, control_queue, shared_new_images,latency_queue ,worker_status):
    """Worker process to handle requests."""
    device = f"cuda:{gpu_id}"
    model_type = "sd3.5"
    seed = 42 #any
    generator = torch.Generator(device).manual_seed(seed)
    model = load_model(model_type, device)
    
    idle_counter = 0
    max_idle_iterations = 100 # Set a threshold for termination
    
    while True:
        try:
            if not control_queue.empty():
                new_allocation = control_queue.get()
                if model_type != new_allocation[gpu_id]:
                    print(f"[Worker {gpu_id}] Switching to {new_allocation[gpu_id]} model")
                    model_type = new_allocation[gpu_id]
                    model = load_model(model_type, device)

            if model_type == "sd3.5":
                try:
                    request = nonhit_queue.get_nowait()
                    idle_counter = 0
                except queue.Empty:
                    request = hit_queue.get(timeout=10)
                    idle_counter = 0
            else:
                request = hit_queue.get(timeout=10)
                idle_counter = 0
                
            prompt = request['prompt']
            clean_prompt = re.sub(r'[^\w\-_\.]', '_', prompt)[:300]
            
            if request['k'] is None:
                scheduler = FlowMatchEulerDiscreteScheduler.from_config(model.scheduler.config)
                timesteps_batch = precompute_timesteps_for_labels_35(scheduler, [0], "cpu",0)[0]
                prompt_embeds, pooled_prompt_embeds, latents = model.input_process(prompt = prompt,negative_prompt = None, generator=generator, callback_on_step_end=None,
            callback_on_step_end_tensor_inputs=["latents"])
                model_outputs = model(prompt = prompt, prompt_embeds = prompt_embeds, pooled_prompt_embeds = pooled_prompt_embeds, generator=generator, callback_on_step_end=None,
                callback_on_step_end_tensor_inputs=["current_latents"], timesteps_batch = timesteps_batch,
                labels_batch = 0, current_latents=latents, height=1024, width=1024)
                
                generated_image_path = f"/data6/stilex/serving/nohit/{clean_prompt}.png"
                
                try:
                    model_outputs[1][0].save(generated_image_path)
                    shared_new_images.put(generated_image_path)
                    
                except Exception as e:
                    print(f"Failed to save image: {e}")
                    print("image name:", prompt)

                print(f"[Worker {gpu_id}] Added new image path to shared list: {generated_image_path}")
                
            else:
                init_image = Image.open(request['retrieved_image_path']).convert("RGB")
                if model_type == "sd3.5":
                    model_ouputs = model.edit(prompt=prompt,
                        image=init_image,
                        strength=request['strength'],
                        guidance_scale=7.5,
                        negative_prompt=None)
                    generated_image_path = f"/data6/stilex/serving/hit/{clean_prompt}.png"
                    try:
                        model_ouputs.images[0].save(generated_image_path)
                        shared_new_images.put(generated_image_path)
                    except Exception as e:
                        print(f"Failed to save image: {e}")
                        print("image name:", prompt)
                        
                else:
                    model_ouputs = model(prompt=prompt,
                        image=init_image,
                        strength=request['strength'],
                        guidance_scale=7.5,
                        negative_prompt=None)
                    generated_image_path = f"/data6/stilex/serving/hit/{clean_prompt}.png"
                    try:
                        model_ouputs.images[0].save(generated_image_path)
                        shared_new_images.put(generated_image_path)
                    except Exception as e:
                        print(f"Failed to save image: {e}")
                        print("image name:", prompt)
                        
            finish_time = time.time() - request['start_time']
            print(f"[Worker {gpu_id}] Processed request latency: {finish_time}")
            latency_queue.put(finish_time)

        except queue.Empty:
            idle_counter += 1
            if idle_counter >= max_idle_iterations:
                print(f"[Worker {gpu_id}] No requests received for a while. Exiting...")
                worker_status[gpu_id] = "dropped"
                break
            continue
            
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    metadata_df = pd.read_parquet('/home/stilex/metadata.parquet')
    if 'timestamp' in metadata_df.columns:
        # print("First few entries in the 'timestamp' column:")
        # print(metadata_df['timestamp'].head())
        
        # Optionally sort by date
        # metadata_df['timestamp'] = pd.to_datetime(metadata_df['timestamp'])  # Ensure it's in datetime format
        sorted_df = metadata_df.sort_values(by='timestamp')
        start_time = sorted_df['timestamp'].iloc[0]
        
        # Add a new column for seconds relative to the start time
        sorted_df['seconds_from_start'] = (sorted_df['timestamp'] - start_time).dt.total_seconds()

        # Display the modified DataFrame
        print(sorted_df[['timestamp', 'seconds_from_start']].head())
        # print("Sorted DataFrame by timestamp:")
        # print(sorted_df.head())
    else:
        print("No timestamp column found in the DataFrame.")
    # print(sorted_df['prompt'][0])
    # sorted_df['seconds_from_start'] = sorted_df.groupby('seconds_from_start')['seconds_from_start'].transform(add_poisson_seconds)
    sorted_df = sorted_df.sort_values(by='seconds_from_start')
    # Generate timestamps for 1000 requests with min rate of 1 req/min
    
    selected_requests = sorted_df.iloc[50000:51000].copy()
    num_requests = len(selected_requests)
    selected_requests['seconds_from_start'] = generate_rapidly_increasing_seconds_from_start(
    num_requests, min_rate=1, max_rate=9
    )
        # Print each value with its index
    for i, seconds in enumerate(selected_requests['seconds_from_start']):
        print(f"Request {i+1}: {seconds:.2f}")
        
    device = "cuda:0"
    
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    image_directory = "/data6/stilex/large_small/nohit/"

    # Get a list of all image file paths in the directory
    cached_image_paths = [os.path.join(image_directory, img_file) for img_file in os.listdir(image_directory) if img_file.endswith(('.png', '.jpg', '.jpeg'))]

    # Batch size
    # batch_size = 128
    # num_batches = (len(cached_image_paths) + batch_size - 1) // batch_size

    # # Placeholder to store all image embeddings
    # image_embeddings = []

    # # Process each batch
    # for batch_idx in tqdm(range(num_batches)):
    #     # Select batch slice
    #     batch_start = batch_idx * batch_size
    #     batch_end = min((batch_idx + 1) * batch_size, len(cached_image_paths))
    #     batch_image_paths = cached_image_paths[batch_start:batch_end]

    #     # Load and preprocess images
    #     images = [Image.open(image_path).convert("RGB") for image_path in batch_image_paths]
    #     image_tensors = processor(images=images, return_tensors="pt").to(device)

    #     with torch.no_grad():
    #         # Generate image embeddings
    #         image_embeddings_batch = model.get_image_features(**image_tensors)
    #         image_embeddings.append(image_embeddings_batch)

    # # Concatenate all image embeddings into a single tensor
    # final_image_embeddings = torch.cat(image_embeddings, dim=0)
    # torch.save(final_image_embeddings, "final_image_embeddings.pt")
    # print(f"Generated embeddings for {len(cached_image_paths)} images.")
    
    final_image_embeddings = torch.load("final_image_embeddings.pt")
    start_time = time.time()

    num_gpus = torch.cuda.device_count()
    hit_queue = mp.Queue()
    nonhit_queue = mp.Queue()
    control_queues = [mp.Queue() for _ in range(num_gpus)]
    request_rate_queue = mp.Queue()
    latency_queue = mp.Queue()

    shared_new_images = mp.Queue()
    manager = mp.Manager()
    worker_status = manager.dict()

    lock = mp.Lock()

    # Start request scheduler process
    scheduler = mp.Process(target=request_scheduler, args=(
        hit_queue, nonhit_queue, selected_requests, start_time, request_rate_queue, final_image_embeddings, cached_image_paths, shared_new_images, lock, model, processor, worker_status
    ))
    scheduler.start()

    # Start global monitor process
    monitor = mp.Process(target=global_monitor, args=(request_rate_queue, control_queues, num_gpus))
    monitor.start()

    # Start worker processes
    workers = []
    for gpu_id in range(num_gpus):
        worker_status[gpu_id] = "starting"
        p = mp.Process(target=worker, args=(gpu_id, hit_queue, nonhit_queue, control_queues[gpu_id], shared_new_images, latency_queue, worker_status))
        p.start()
        workers.append(p)

    # Wait for scheduler to complete
    scheduler.join()

    # Terminate global monitor
    monitor.terminate()
    # Wait for all workers to finish
    for p in workers:
        p.join()
        
    all_latencies = []
    while not latency_queue.empty():
        all_latencies.append(latency_queue.get())

    print("\n🚀 **Final Latency Report**")
    for i, latency in enumerate(all_latencies):
        print(f"{latency:.4f}")

    print(f"\n📊 **Latency Summary**: Min = {min(all_latencies):.4f}s, Max = {max(all_latencies):.4f}s, Avg = {sum(all_latencies)/len(all_latencies):.4f}s")

    print("[Main] All requests processed.")
    
    
    
    