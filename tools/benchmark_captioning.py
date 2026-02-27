#!/usr/bin/env python3
"""
BLIP Captioning Benchmark
===========================
Measure the throughput (FPS) and latency of the BLIP + BERT-tiny
captioning pipeline on your current hardware.

Usage:
    cd rema_net
    python -m tools.benchmark_captioning
"""

import time
import numpy as np
from PIL import Image

import torch

# Reuse the CaptionStream from the models package
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.streams import CaptionStream


def benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Initialising CaptionStream …")
    try:
        model = CaptionStream(device=device)
    except Exception as e:
        print(f"Error: {e}")
        return

    model.to(device)
    model.eval()

    dummy_image = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )

    # Warm-up
    print("\nWarm-up (10 iterations) …")
    with torch.no_grad():
        for i in range(10):
            _ = model(dummy_image)
            print(f"  {i + 1}/10", end="\r")

    # Benchmark
    print("\nBenchmark (50 iterations) …")
    times = []
    with torch.no_grad():
        for i in range(50):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            _ = model(dummy_image)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.time() - t0)
            print(f"  Iter {i + 1}: {times[-1] * 1000:.2f} ms", end="\r")

    avg = np.mean(times)
    fps = 1.0 / avg

    print(f"\n\nResults")
    print(f"  Average latency : {avg * 1000:.2f} ms")
    print(f"  Average FPS     : {fps:.2f}")

    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        print(f"  Peak GPU memory : {mem:.2f} MB")

    out = os.path.join(os.path.dirname(__file__), "..", "caption_benchmark_results.txt")
    with open(out, "w") as f:
        f.write(f"Device: {device}\n")
        f.write(f"Average Inference Time per Frame: {avg * 1000:.2f} ms\n")
        f.write(f"Average FPS: {fps:.2f}\n")
        if torch.cuda.is_available():
            f.write(f"Max Memory Allocated: {mem:.2f} MB\n")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    benchmark()
