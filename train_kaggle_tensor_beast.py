#!/usr/bin/env python3
"""
🚀 KAGGLE TENSOR BEAST (VERSION 2.0 - USEFUL MODEL)
- Loads pre-packed SAFE SPLIT tensors (No data leakage).
- Data Augmentation on GPU (Random rotation, brightness, noise).
- Weighted Loss to handle empty square imbalance.
"""
import os
import sys
import time
import glob
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from datetime import timedelta
from torchvision import transforms

try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from chessimg2pos.chessclassifier import UltraEnhancedChessPieceClassifier

import argparse

# ============ ARGUMENT PARSING ============
parser = argparse.ArgumentParser(description='Kaggle Tensor Beast Training')
parser.add_argument('--data-dir', type=str, default='tensor_dataset_synthetic', help='Directory containing .pt chunks')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
parser.add_argument('--batch-size', type=int, default=4096, help='Batch size')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
args = parser.parse_args()

# ============ CONFIG ============
FEN_CHARS = "1RNBQKPrnbqkp"
USE_GRAYSCALE = True
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr

# ============ PATHS ============
base_dir = os.path.dirname(os.path.abspath(__file__))
tensor_dir = os.path.join(base_dir, args.data_dir)
output_model = os.path.join(base_dir, "models", "model_tensor_beast.pt")
checkpoint_path = os.path.join(base_dir, "models", "checkpoint_tensor_beast.pt")
os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)

# ============ DEVICE ============
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def get_gpu_mem():
    if torch.cuda.is_available():
        mem_info = []
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
            mem_info.append(f"GPU{i}:{allocated:.1f}/{reserved:.1f}GB")
        return " ".join(mem_info)
    return "N/A"

# ============ AUGMENTATION ============
# Apply these on the GPU batch for maximum speed
train_augmentations = nn.Sequential(
    transforms.RandomAffine(
        degrees=2,           # Very slight rotation
        translate=(0.04, 0.04), # Shift pieces slightly (up to roughly 1-1.5 pixels)
        scale=(0.96, 1.04)   # Slight scaling differences
    ),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)), # Simulate blurry rescales
)

def apply_noise(x, amount=0.15): # Heavy noise to simulate JPEG compression blockiness on empty squares
    noise = torch.randn_like(x) * amount
    return torch.clamp(x + noise, -1.0, 1.0) # Ensure we don't go out of normalized bounds

# ============ DATA LOADING ============
def load_tensors(pattern):
    print(f"📂 Loading {pattern} tensors from {tensor_dir}...")
    start = time.time()
    files = sorted(glob.glob(os.path.join(tensor_dir, pattern)))
    
    all_x, all_y = [], []
    for f in files:
        data = torch.load(f, map_location='cpu')
        all_x.append(data['x'])
        all_y.append(data['y'])
        print(f"   Loaded {os.path.basename(f)}...")
        
    x = torch.cat(all_x, dim=0)
    y = torch.cat(all_y, dim=0).long()
    print(f"✅ Loaded {x.size(0):,} images in {time.time()-start:.1f}s")
    return x, y

# ============ MAIN ============
def main():
    print("="*70)
    print("🔥 KAGGLE TENSOR BEAST V2 - ENHANCED FOR REAL-WORLD")
    print("="*70)

    # Load Train and Val separately (No Leakage!)
    train_x, train_y = load_tensors("train_chunk_*.pt")
    val_x, val_y = load_tensors("val_chunk_*.pt")
    
    train_ds = TensorDataset(train_x, train_y)
    val_ds = TensorDataset(val_x, val_y)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # Model
    model = UltraEnhancedChessPieceClassifier(
        num_classes=len(FEN_CHARS),
        use_grayscale=USE_GRAYSCALE
    ).to(device)

    if torch.cuda.device_count() > 1:
        print(f"🚀 DataParallel enabled ({torch.cuda.device_count()} GPUs)")
        model = nn.DataParallel(model)

    # Standard Loss: We no longer need to penalize 'Empty' since 
    # our generated positions are real legal chess games now!
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = GradScaler('cuda')

    # Resumption logic — load model/optimizer/scaler first, then build scheduler
    start_epoch, best_acc = 0, 0.0
    if os.path.exists(checkpoint_path):
        print(f"🔄 Loading checkpoint...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt['model_state_dict']
        is_dp = isinstance(model, nn.DataParallel)
        has_prefix = any(k.startswith('module.') for k in state_dict.keys())
        if is_dp and not has_prefix:
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
        elif not is_dp and has_prefix:
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt['best_acc']
        print(f"✅ Resumed from epoch {start_epoch} (Best: {best_acc:.4f})\n")

    # Build OneCycleLR AFTER knowing start_epoch so total_steps covers only remaining epochs.
    # OneCycleLR cannot safely be restored from checkpoint when the epoch count changes.
    remaining_epochs = max(EPOCHS - start_epoch, 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE*2,
        steps_per_epoch=len(train_loader), epochs=remaining_epochs
    )

    print(f"🚀 Training for {EPOCHS} epochs with Data Augmentation...")
    total_start = time.time()

    for epoch in range(start_epoch, EPOCHS):
        epoch_start = time.time()
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            # 1. To Device - Data is ALREADY normalized to [-1, 1] by generator via ToTensor+Normalize
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            # 2. GPU Augmentation
            with torch.no_grad():
                x = train_augmentations(x)
                x = apply_noise(x)
                
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                outputs = model(x)
                loss = criterion(outputs, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()
            
            if batch_idx % 50 == 0:
                progress = (batch_idx + 1) / len(train_loader) * 100
                elapsed = time.time() - epoch_start
                sps = (batch_idx * BATCH_SIZE) / elapsed if elapsed > 0 else 0
                eta_batch = (100 - progress) * (elapsed / progress) if progress > 0 else 0
                
                bar = '█' * int(30 * progress / 100) + ' ' * (30 - int(30 * progress / 100))
                print(f"\rEpoch {epoch+1}/{EPOCHS} |{bar}| {progress:5.1f}% | Loss: {loss.item():.4f} | {sps:,.0f} samples/s | {get_gpu_mem()} | ETA: {format_time(eta_batch)}", end="", flush=True)

        train_acc = correct / total
        
        # Validation (NO Augmentation)
        model.eval()
        val_correct, val_total = 0, 0
        print("\n🔍 Validating...", end="", flush=True)
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with autocast('cuda'):
                    outputs = model(x)
                _, pred = torch.max(outputs, 1)
                val_total += y.size(0)
                val_correct += (pred == y).sum().item()
        
        val_acc = val_correct / val_total
        epoch_time = time.time() - epoch_start
        total_elapsed = time.time() - total_start
        eta = (EPOCHS - epoch - 1) * (total_elapsed / (epoch - start_epoch + 1))

        print(f"\n\n{'='*70}")
        print(f"📊 Epoch {epoch+1}/{EPOCHS} Complete")
        print(f"   Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        print(f"   Epoch Time: {format_time(epoch_time)} | Total: {format_time(total_elapsed)} | ETA: {format_time(eta)}")
        print(f"   LR: {optimizer.param_groups[0]['lr']:.6f} | {get_gpu_mem()}")
        
        # SAVE
        model_to_save = model.module if isinstance(model, nn.DataParallel) else model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # Note: scheduler_state_dict not saved — OneCycleLR is reconstructed on resume
            'scaler_state_dict': scaler.state_dict(),
            'best_acc': max(best_acc, val_acc)
        }, checkpoint_path)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model_to_save.state_dict(), output_model)
            print(f"   ✨ NEW BEST MODEL SAVED! {best_acc:.4f}")
        print("="*70 + "\n")

if __name__ == "__main__":
    main()
