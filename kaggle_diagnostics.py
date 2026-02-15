#!/usr/bin/env python3
"""
Kaggle Hardware Diagnostics - Find optimal settings
"""
import torch
import os
import psutil

print("="*70)
print("🔍 KAGGLE HARDWARE DIAGNOSTICS")
print("="*70)

# CPU Info
print(f"\n💻 CPU:")
print(f"   Cores: {os.cpu_count()}")
print(f"   RAM: {psutil.virtual_memory().total / 1e9:.1f}GB")
print(f"   Available RAM: {psutil.virtual_memory().available / 1e9:.1f}GB")

# GPU Info
if torch.cuda.is_available():
    print(f"\n🎮 GPU:")
    gpu_count = torch.cuda.device_count()
    print(f"   Count: {gpu_count}")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f"\n   GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"      Total Memory: {props.total_memory / 1e9:.1f}GB")
        print(f"      Multi-Processors: {props.multi_processor_count}")
        print(f"      Max Threads/Block: {props.max_threads_per_block}")
        print(f"      Max Threads/MP: {props.max_threads_per_multi_processor}")
    
    total_vram = sum(torch.cuda.get_device_properties(i).total_memory for i in range(gpu_count))
    print(f"\n   Total VRAM: {total_vram / 1e9:.1f}GB")

# Recommendations
print("\n" + "="*70)
print("💡 RECOMMENDED SETTINGS FOR DUAL T4")
print("="*70)

if torch.cuda.is_available() and torch.cuda.device_count() == 2:
    total_vram_gb = total_vram / 1e9
    
    # Estimate batch size (assuming 32x32 grayscale images, ~4KB each)
    # Model + optimizer state ~2GB, leave 2GB buffer
    usable_vram = total_vram_gb - 4
    
    # Each sample: image (4KB) + gradients + activations ≈ 50KB
    max_batch = int((usable_vram * 1e9) / 50000)
    
    print(f"\n📊 Batch Size:")
    print(f"   Conservative: {max_batch // 2:,}")
    print(f"   Aggressive: {max_batch:,}")
    print(f"   BEAST MODE: {max_batch * 2:,} (may OOM, but try it!)")
    
    print(f"\n👷 Workers:")
    print(f"   num_workers: {os.cpu_count()}")
    print(f"   prefetch_factor: 8-16 (high for large batches)")
    print(f"   persistent_workers: True")
    
    print(f"\n⚡ Training:")
    print(f"   Mixed Precision: True (AMP)")
    print(f"   DataParallel: True (2 GPUs)")
    print(f"   Gradient Accumulation: 1 (not needed with large batch)")
    
    print(f"\n🎯 Expected Performance:")
    print(f"   Samples/sec: 30,000-50,000+")
    print(f"   GPU Memory: 12-14GB per GPU")
    print(f"   GPU Utilization: 90-100%")
    print(f"   Time per epoch: 3-5 minutes")
    print(f"   Total training (25 epochs): 1.5-2 hours")

print("\n" + "="*70)
