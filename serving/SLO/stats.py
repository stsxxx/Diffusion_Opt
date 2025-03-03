import os
import numpy as np

# Directory containing log files
log_dir = "/home/stilex/diffusion-opt/serving/SLO/logs"

# Process each log file in the directory
for log_file in os.listdir(log_dir):
    log_path = os.path.join(log_dir, log_file)

    if not os.path.isfile(log_path):
        continue  # Skip if it's not a file

    print(f"Processing {log_file}...")

    latencies = []
    with open(log_path, "r") as file:
        lines = file.readlines()

    # Find the index of "Final Latency Report"
    start_index = None
    for i, line in enumerate(lines):
        if "Final Latency Report" in line:
            start_index = i + 1
            break

    # Extract latencies directly from the next 1000 lines
    if start_index:
        latencies = [float(line.strip()) for line in lines[start_index : start_index + 1000] if line.strip()]

    # Compute statistics
    if latencies:
        total_count = len(latencies)
        p99_latency = np.percentile(latencies, 99)
        count_100 = sum(lat > 100 for lat in latencies)
        count_200 = sum(lat > 200 for lat in latencies)

        ratio_100 = count_100 / total_count if total_count > 0 else 0
        ratio_200 = count_200 / total_count if total_count > 0 else 0

        print(f"  99th Percentile Latency: {p99_latency:.2f} s")
        print(f"  Number of latencies > 100 s: {count_100} ({ratio_100})")
        print(f"  Number of latencies > 200 s: {count_200} ({ratio_200})\n")
    else:
        print("  No latency data found after 'Final Latency Report'.\n")
