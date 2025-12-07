"""Quick comparison between original and Taichi SPH solvers.

Run: python scripts/compare_performance.py
"""
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 70)
print("SPH Solver Performance Comparison")
print("=" * 70)

# Test configuration
config_path = Path("config/scene_config.yaml")
num_steps = 50

print(f"\nConfiguration: {config_path}")
print(f"Steps to simulate: {num_steps}")
print()

# Test 1: Original solver (Numba)
print("-" * 70)
print("TEST 1: Original WCSPH Solver (Numba)")
print("-" * 70)

from engine import WorldContainer

t_start = time.perf_counter()
container_original = WorldContainer.from_config_file(config_path, use_taichi=False)

# Warmup
for _ in range(3):
    container_original.step(export=False)

times_original = []
for i in range(num_steps):
    t0 = time.perf_counter()
    container_original.step(export=False)
    t1 = time.perf_counter()
    times_original.append((t1 - t0) * 1000)
    if i % 10 == 0:
        print(f"  Step {i:3d}: {times_original[-1]:.2f} ms")

import numpy as np
avg_original = np.mean(times_original)
std_original = np.std(times_original)
print(f"\n✓ Original solver: {avg_original:.2f} ± {std_original:.2f} ms/step")

# Test 2: Taichi solver (GPU)
print("\n" + "-" * 70)
print("TEST 2: Taichi WCSPH Solver (GPU-accelerated)")
print("-" * 70)

container_taichi = WorldContainer.from_config_file(config_path, use_taichi=True)

# Warmup (important for GPU - first run includes compilation)
print("  Warming up GPU...")
for _ in range(3):
    container_taichi.step(export=False)

times_taichi = []
for i in range(num_steps):
    t0 = time.perf_counter()
    container_taichi.step(export=False)
    t1 = time.perf_counter()
    times_taichi.append((t1 - t0) * 1000)
    if i % 10 == 0:
        print(f"  Step {i:3d}: {times_taichi[-1]:.2f} ms")

avg_taichi = np.mean(times_taichi)
std_taichi = np.std(times_taichi)
print(f"\n✓ Taichi solver: {avg_taichi:.2f} ± {std_taichi:.2f} ms/step")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Original (Numba):  {avg_original:.2f} ms/step")
print(f"Taichi (GPU):      {avg_taichi:.2f} ms/step")
print(f"\nSpeedup:           {avg_original / avg_taichi:.2f}x faster")
print(f"Time saved:        {avg_original - avg_taichi:.2f} ms/step")
print(f"\nFor 1000 steps:")
print(f"  Original: {avg_original * 1000 / 1000:.1f} seconds")
print(f"  Taichi:   {avg_taichi * 1000 / 1000:.1f} seconds")
print(f"  Saved:    {(avg_original - avg_taichi) * 1000 / 1000:.1f} seconds")
print("=" * 70)
