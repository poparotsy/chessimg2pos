#!/usr/bin/env python3
"""
🚀 KAGGLE TENSOR BEAST
- Loads pre-packed .pt tensors into System RAM.
- Zero-latency data loading.
- Safe VRAM management to avoid OOM.
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

try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from chessimg2pos.chessclassifier import UltraEnhancedChessPieceClassifier

# ============ CONFIG ============
FEN_CHARS = "1RNBQKPrnbqkp"
USE_GRAYSCALE = True
EPOCHS = 50
BATCH_SIZE = 4096 
LEARNING_RATE = 0.001

# ============ PATHS ============
base_dir = os.path.dirname(os.path.abspath(__file__))
tensor_dir = os.path.join(base_dir, "tensor_dataset")
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

# ============ DATA LOADING ============
def load_full_dataset():
    print(f"📂 Loading all tensors from {tensor_dir} into System RAM...")
    start = time.time()
    chunk_files = sorted(glob.glob(os.path.join(tensor_dir, "chunk_*.pt")))
    
    all_x = []
    all_y = []
    
    for f in chunk_files:
        data = torch.load(f, map_location='cpu')
        all_x.append(data['x'])
        all_y.append(data['y'])
        print(f"   Loaded {os.path.basename(f)}...")
        
    x = torch.cat(all_x, dim=0)
    y = torch.cat(all_y, dim=0).long()
    
    elapsed = time.time() - start
    print(f"✅ Loaded {x.size(0):,} images into RAM in {elapsed:.1f}s")
    return x, y

# ============ MAIN ============
def main():
    print("="*70)
    print("🔥 KAGGLE TENSOR BEAST - ZERO LATENCY TRAINING")
    print("="*70)

    # Load everything into RAM
    full_x, full_y = load_full_dataset()
    
    # Split
    indices = torch.randperm(full_x.size(0))
    split = int(0.9 * full_x.size(0))
    train_idx, val_idx = indices[:split], indices[split:]
    
    train_ds = TensorDataset(full_x[train_idx], full_y[train_idx])
    val_ds = TensorDataset(full_x[val_idx], full_y[val_idx])
    
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE*2,
        steps_per_epoch=len(train_loader), epochs=EPOCHS
    )
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler('cuda')

    # Resumption logic
    start_epoch, best_acc = 0, 0.0
    if os.path.exists(checkpoint_path):
        print(f"🔄 Loading checkpoint...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt['best_acc']
        print(f"✅ Resumed from epoch {start_epoch} (Best: {best_acc:.4f})\n")

    print(f"🚀 Training for {EPOCHS} epochs...")
    total_start = time.time()

    for epoch in range(start_epoch, EPOCHS):
        epoch_start = time.time()
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            # Move to GPU and convert to float32 ONLY for this batch
            x = x.to(device, non_blocking=True).float() / 255.0 # Simple normalization
            x = (x - 0.5) / 0.5 # [-1, 1]
            y = y.to(device, non_blocking=True)
            
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
                
                bar_len = 30
                filled = int(bar_len * progress / 100)
                bar = '█' * filled + ' ' * (bar_len - filled)
                print(f"\rEpoch {epoch+1}/{EPOCHS} |{bar}| {progress:5.1f}% | Loss: {loss.item():.4f} | {sps:,.0f} samples/s | {get_gpu_mem()}", end="", flush=True)

        train_acc = correct / total
        
        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        print("\n🔍 Validating...", end="", flush=True)
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True).float() / 255.0
                x = (x - 0.5) / 0.5
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
            'scheduler_state_dict': scheduler.state_dict(),
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
