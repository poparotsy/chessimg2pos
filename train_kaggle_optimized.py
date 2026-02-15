#!/usr/bin/env python3
"""
KAGGLE OPTIMIZED TRAINING SCRIPT
- Dual T4 GPU support (30GB total VRAM)
- Mixed precision training (AMP)
- Board-level splitting for reliable validation
- Checkpoint resumption
- Uses UltraEnhancedChessPieceClassifier (best model)
"""
import os
import sys
import time
import glob
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from datetime import timedelta
from torch.amp import GradScaler, autocast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from chessimg2pos.chessclassifier import UltraEnhancedChessPieceClassifier
from chessimg2pos.chessdataset import ChessTileDataset, create_image_transforms

# ============ CONFIG ============
FEN_CHARS = "1RNBQKPrnbqkp"
USE_GRAYSCALE = True
EPOCHS = 25
BATCH_SIZE = 512  # Optimized for dual T4
LEARNING_RATE = 0.001
NUM_WORKERS = 4  # Kaggle has 4 vCPUs

# ============ PATHS ============
base_dir = os.path.dirname(os.path.abspath(__file__))
tiles_dir = os.path.join(base_dir, "images", "tiles_real")
output_model = os.path.join(base_dir, "models", "model_kaggle_ultra.pt")
checkpoint_path = os.path.join(base_dir, "models", "checkpoint_kaggle_ultra.pt")
os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)

# ============ DEVICE SETUP ============
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def get_memory_stats():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        return f"GPU Mem: {allocated:.2f}/{reserved:.2f}GB"
    return ""

print(f"🚀 Kaggle Optimized Training")
print(f"📍 Device: {device}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"🔢 GPU Count: {torch.cuda.device_count()}")
print(f"📂 Tiles: {tiles_dir}\n")

# ============ DATA LOADING ============
print("📂 Scanning board directories...")
board_dirs = sorted([d for d in glob.glob(f"{tiles_dir}/*") if os.path.isdir(d)])
print(f"📊 Found {len(board_dirs)} boards")

if not board_dirs:
    print(f"❌ No boards found in {tiles_dir}")
    exit(1)

# Board-level split (reproducible)
np.random.seed(42)
np.random.shuffle(board_dirs)
split_idx = int(len(board_dirs) * 0.8)
train_dirs = board_dirs[:split_idx]
val_dirs = board_dirs[split_idx:]

print(f"📈 Split: {len(train_dirs)} train, {len(val_dirs)} val boards")

def collect_tiles(dirs):
    paths = []
    for d in dirs:
        paths.extend(glob.glob(os.path.join(d, "*.png")))
    return np.array(paths)

print("🔍 Collecting tiles...")
train_paths = collect_tiles(train_dirs)
val_paths = collect_tiles(val_dirs)
print(f"🗂️  Train: {len(train_paths)} tiles | Val: {len(val_paths)} tiles\n")

# ============ DATASETS ============
train_dataset = ChessTileDataset(train_paths, FEN_CHARS, USE_GRAYSCALE, 
                                 create_image_transforms(USE_GRAYSCALE))
val_dataset = ChessTileDataset(val_paths, FEN_CHARS, USE_GRAYSCALE, 
                               create_image_transforms(USE_GRAYSCALE))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

print(f"✅ DataLoaders ready ({NUM_WORKERS} workers)\n")

# ============ MODEL ============
print("🧠 Creating UltraEnhancedChessPieceClassifier...")
model = UltraEnhancedChessPieceClassifier(
    num_classes=len(FEN_CHARS),
    use_grayscale=USE_GRAYSCALE
).to(device)

# Multi-GPU support
if torch.cuda.device_count() > 1:
    print(f"🚀 Using {torch.cuda.device_count()} GPUs (DataParallel)")
    model = nn.DataParallel(model)

# ============ TRAINING SETUP ============
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LEARNING_RATE*2, 
    steps_per_epoch=len(train_loader), epochs=EPOCHS
)
scaler = GradScaler('cuda') if device.type == 'cuda' else None

# ============ CHECKPOINT RESUME ============
start_epoch, best_acc = 0, 0.0
if os.path.exists(checkpoint_path):
    print(f"🔄 Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Handle DataParallel prefix
    state_dict = ckpt['model_state_dict']
    is_dp = isinstance(model, nn.DataParallel)
    has_prefix = any(k.startswith('module.') for k in state_dict.keys())
    
    if is_dp and not has_prefix:
        state_dict = {f'module.{k}': v for k, v in state_dict.items()}
    elif not is_dp and has_prefix:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if scaler and 'scaler_state_dict' in ckpt:
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    
    start_epoch = ckpt['epoch'] + 1
    best_acc = ckpt['best_acc']
    print(f"✅ Resumed from epoch {start_epoch} (Best: {best_acc:.4f})\n")
else:
    print(f"✅ Starting fresh training\n")

# ============ TRAINING LOOP ============
print(f"🚀 Training for {EPOCHS} epochs...\n")
total_start = time.time()

for epoch in range(start_epoch, EPOCHS):
    epoch_start = time.time()
    
    # TRAIN
    model.train()
    train_loss, correct, total = 0.0, 0, 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        if scaler:
            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        scheduler.step()
        
        train_loss += loss.item()
        _, pred = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
        
        if batch_idx % 20 == 0:
            progress = (batch_idx + 1) / len(train_loader) * 100
            mem = get_memory_stats()
            print(f"\rEpoch {epoch+1}/{EPOCHS} [{progress:5.1f}%] Loss: {loss.item():.4f} {mem}", 
                  end="", flush=True)
    
    train_acc = correct / total
    
    # VALIDATION
    model.eval()
    val_correct, val_total = 0, 0
    print("\n🔍 Validating...", end="", flush=True)
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            if scaler:
                with autocast('cuda'):
                    outputs = model(inputs)
            else:
                outputs = model(inputs)
            
            _, pred = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (pred == labels).sum().item()
    
    val_acc = val_correct / val_total
    
    # TIMING
    epoch_time = time.time() - epoch_start
    elapsed = time.time() - total_start
    completed = epoch - start_epoch + 1
    eta = (EPOCHS - epoch - 1) * (elapsed / completed)
    
    print(f"\n📊 Epoch {epoch+1}: Train={train_acc:.4f} Val={val_acc:.4f} "
          f"Time={format_time(epoch_time)} ETA={format_time(eta)}")
    
    # SAVE CHECKPOINT
    model_to_save = model.module if isinstance(model, nn.DataParallel) else model
    save_dict = {
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_acc': max(best_acc, val_acc)
    }
    if scaler:
        save_dict['scaler_state_dict'] = scaler.state_dict()
    
    torch.save(save_dict, checkpoint_path)
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model_to_save.state_dict(), output_model)
        print(f"✨ Best model saved! Acc: {best_acc:.4f}\n")
    else:
        print()

print(f"\n✅ Training Complete!")
print(f"🏆 Best Accuracy: {best_acc:.2%}")
print(f"🕒 Total Time: {format_time(time.time() - total_start)}")
print(f"💾 Model: {output_model}")
