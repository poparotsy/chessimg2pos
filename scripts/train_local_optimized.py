#!/usr/bin/env python3
"""
LOCAL TRAINING SCRIPT (ARM/x86 Compatible)
- Works on Apple Silicon (MPS), CUDA, and CPU
- Board-level splitting for reliable validation
- Checkpoint resumption
- Uses UltraEnhancedChessPieceClassifier (best model)
- Optimized batch size for local hardware
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from chessimg2pos.chessclassifier import UltraEnhancedChessPieceClassifier
from chessimg2pos.chessdataset import ChessTileDataset, create_image_transforms

# ============ CONFIG ============
FEN_CHARS = "1RNBQKPrnbqkp"
USE_GRAYSCALE = True
EPOCHS = 20
BATCH_SIZE = 128  # Conservative for local hardware
LEARNING_RATE = 0.001

# ============ PATHS ============
base_dir = os.path.dirname(os.path.abspath(__file__))
tiles_dir = os.path.join(base_dir, "images", "tiles_real")
output_model = os.path.join(base_dir, "models", "model_local_ultra.pt")
checkpoint_path = os.path.join(base_dir, "models", "checkpoint_local_ultra.pt")
os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)

# ============ DEVICE SETUP (ARM Compatible) ============
if torch.cuda.is_available():
    device = torch.device("cuda")
    use_amp = True
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    use_amp = False  # MPS doesn't support AMP yet
else:
    device = torch.device("cpu")
    use_amp = False

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

print(f"🚀 Local Training Script")
print(f"📍 Device: {device}")
if device.type == "cuda":
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
elif device.type == "mps":
    print(f"🍎 Apple Silicon GPU")
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

# Use fewer workers for local (avoid overhead)
num_workers = 2 if device.type != "cpu" else 0
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=num_workers, pin_memory=(device.type == "cuda"))
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=num_workers, pin_memory=(device.type == "cuda"))

print(f"✅ DataLoaders ready ({num_workers} workers)\n")

# ============ MODEL ============
print("🧠 Creating UltraEnhancedChessPieceClassifier...")
model = UltraEnhancedChessPieceClassifier(
    num_classes=len(FEN_CHARS),
    use_grayscale=USE_GRAYSCALE
).to(device)

# ============ TRAINING SETUP ============
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=2, verbose=True
)

# ============ CHECKPOINT RESUME ============
start_epoch, best_acc = 0, 0.0
if os.path.exists(checkpoint_path):
    print(f"🔄 Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    
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
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()
        _, pred = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
        
        if batch_idx % 50 == 0:
            progress = (batch_idx + 1) / len(train_loader) * 100
            print(f"\rEpoch {epoch+1}/{EPOCHS} [{progress:5.1f}%] Loss: {loss.item():.4f}", 
                  end="", flush=True)
    
    train_acc = correct / total
    
    # VALIDATION
    model.eval()
    val_correct, val_total = 0, 0
    print("\n🔍 Validating...", end="", flush=True)
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
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
    
    # Update scheduler
    scheduler.step(val_acc)
    
    # SAVE CHECKPOINT
    save_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_acc': max(best_acc, val_acc)
    }
    
    torch.save(save_dict, checkpoint_path)
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), output_model)
        print(f"✨ Best model saved! Acc: {best_acc:.4f}\n")
    else:
        print()

print(f"\n✅ Training Complete!")
print(f"🏆 Best Accuracy: {best_acc:.2%}")
print(f"🕒 Total Time: {format_time(time.time() - total_start)}")
print(f"💾 Model: {output_model}")
