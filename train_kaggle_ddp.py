#!/usr/bin/env python3
"""
DDP (DistributedDataParallel) version - Proper multi-GPU training
Launch with: torchrun --nproc_per_node=2 train_kaggle_ddp.py
"""
import os
import sys
import time
import glob
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import GradScaler, autocast
import numpy as np
from datetime import timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from chessimg2pos.chessclassifier import UltraEnhancedChessPieceClassifier
from chessimg2pos.chessdataset import ChessTileDataset, create_image_transforms

# ============ DDP SETUP ============
def setup_ddp():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

# ============ ARGUMENT PARSING ============
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--batch-size', type=int, default=8192)
parser.add_argument('--lr', type=float, default=0.003)
args = parser.parse_args()

# ============ CONFIG ============
local_rank = setup_ddp()
is_main = local_rank == 0

FEN_CHARS = "1RNBQKPrnbqkp"
USE_GRAYSCALE = True
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr

base_dir = os.path.dirname(os.path.abspath(__file__))
tiles_dir = os.path.join(base_dir, "images", "tiles_kaggle")
model_path = os.path.join(base_dir, "models", "model_ddp.pt")

device = torch.device(f'cuda:{local_rank}')
torch.backends.cudnn.benchmark = True

# ============ DATA LOADING ============
if is_main:
    print(f"📂 Loading data from: {tiles_dir}")

board_dirs = sorted([d for d in glob.glob(f"{tiles_dir}/*") if os.path.isdir(d)])
np.random.seed(42)
np.random.shuffle(board_dirs)

split_idx = int(0.9 * len(board_dirs))
train_dirs = board_dirs[:split_idx]
val_dirs = board_dirs[split_idx:]

if is_main:
    print(f"📊 Train: {len(train_dirs):,} boards | Val: {len(val_dirs):,} boards")

def collect_tiles(dirs):
    paths = []
    for d in dirs:
        paths.extend(glob.glob(os.path.join(d, "*.png")))
    return np.array(paths)

train_paths = collect_tiles(train_dirs)
val_paths = collect_tiles(val_dirs)

if is_main:
    print(f"🔄 Creating datasets... Train: {len(train_paths):,} | Val: {len(val_paths):,}")

train_ds = ChessTileDataset(train_paths, FEN_CHARS, USE_GRAYSCALE, 
                            create_image_transforms(USE_GRAYSCALE))
val_ds = ChessTileDataset(val_paths, FEN_CHARS, USE_GRAYSCALE, 
                          create_image_transforms(USE_GRAYSCALE))

# DDP Samplers
train_sampler = DistributedSampler(train_ds, shuffle=True)
val_sampler = DistributedSampler(val_ds, shuffle=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler,
                          num_workers=2, pin_memory=True, prefetch_factor=4)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, sampler=val_sampler,
                        num_workers=2, pin_memory=True)

# ============ MODEL ============
if is_main:
    print("🧠 Creating model...")

model = UltraEnhancedChessPieceClassifier(
    num_classes=len(FEN_CHARS),
    use_grayscale=USE_GRAYSCALE,
    dropout_rate=0.3
).to(device)

model = DDP(model, device_ids=[local_rank])

# ============ TRAINING SETUP ============
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LEARNING_RATE*2,
    steps_per_epoch=len(train_loader), epochs=EPOCHS
)
scaler = GradScaler('cuda')

# ============ TRAINING ============
if is_main:
    print(f"\n🚀 Training {EPOCHS} epochs with DDP on {dist.get_world_size()} GPUs\n")

for epoch in range(EPOCHS):
    train_sampler.set_epoch(epoch)
    model.train()
    
    epoch_start = time.time()
    train_loss, correct, total = 0.0, 0, 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        train_loss += loss.item()
        _, pred = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
        
        if is_main and batch_idx % 10 == 0:
            print(".", end="", flush=True)
    
    train_acc = correct / total
    epoch_time = time.time() - epoch_start
    
    # Validation
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with autocast('cuda'):
                outputs = model(inputs)
            
            _, pred = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (pred == labels).sum().item()
    
    val_acc = val_correct / val_total
    
    if is_main:
        print(f"\nEpoch {epoch+1}/{EPOCHS} | Train: {train_acc:.4f} | Val: {val_acc:.4f} | Time: {epoch_time:.1f}s")
        
        # Save model (only main process)
        if (epoch + 1) % 5 == 0:
            torch.save(model.module.state_dict(), model_path)
            print(f"💾 Saved: {model_path}")

if is_main:
    torch.save(model.module.state_dict(), model_path)
    print(f"\n✅ Training complete! Model: {model_path}")

cleanup_ddp()
