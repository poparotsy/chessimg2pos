#!/usr/bin/env python3
"""
🚀 KAGGLE TURBO
- Based on Ultra-Beast logic
- SURGICAL FIX: Pre-loads all 80,000 tiles into VRAM (once).
- Uses stable DataParallel (No torchrun/DDP errors).
- 0ms Data Loading time during training.

Usage:
  python3 train_kaggle_turbo.py --epochs 25 --batch-size 4096
"""
import os
import sys
import time
import glob
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from datetime import timedelta
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from chessimg2pos.chessclassifier import UltraEnhancedChessPieceClassifier

# ============ ARGUMENT PARSING ============
parser = argparse.ArgumentParser(description='Kaggle Turbo Training')
parser.add_argument('--epochs', type=int, default=25, help='Number of epochs')
parser.add_argument('--batch-size', type=int, default=4096, help='Batch size (Optimized for DataParallel)')
parser.add_argument('--lr', type=float, default=0.003, help='Learning rate')
args = parser.parse_args()

# ============ CONFIG ============
FEN_CHARS = "1RNBQKPrnbqkp"
USE_GRAYSCALE = True
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr

# ============ PATHS ============
base_dir = os.path.dirname(os.path.abspath(__file__))
tiles_dir = os.path.join(base_dir, "images", "tiles_real")
if not os.path.exists(tiles_dir):
    tiles_dir = os.path.abspath(os.path.join(base_dir, "..", "images", "tiles_real"))

output_model = os.path.join(base_dir, "models", "model_turbo.pt")
checkpoint_path = os.path.join(base_dir, "models", "checkpoint_turbo.pt")
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

# ============ TURBO VRAM LOADING ============
def load_to_vram(paths, label_tag):
    print(f"📦 Pre-loading {label_tag} tiles ({len(paths):,}) into VRAM...")
    start_time = time.time()
    imgs = []
    lbls = []
    
    for i, path in enumerate(paths):
        # Piece type is the char before .png (e.g., .../f8_q.png -> q)
        piece_type = path[-5]
        if piece_type not in FEN_CHARS:
            continue
            
        try:
            # Load and decode ONCE
            img = Image.open(path).convert("L" if USE_GRAYSCALE else "RGB")
            img = img.resize((32, 32))
            imgs.append(np.array(img, dtype=np.uint8))
            lbls.append(FEN_CHARS.index(piece_type))
        except:
            continue
            
        if (i + 1) % 20000 == 0:
            print(f"   Processed {i+1}/{len(paths)}...")

    # Convert to Tensors
    x = torch.from_numpy(np.stack(imgs)).float()
    if USE_GRAYSCALE:
        x = x.unsqueeze(1)
    else:
        x = x.permute(0, 3, 1, 2)
        
    # Normalize
    x = (x - 127.5) / 127.5
    y = torch.tensor(lbls, dtype=torch.long)
    
    # MOVE TO GPU NOW
    x = x.to(device)
    y = y.to(device)
    
    elapsed = time.time() - start_time
    print(f"✅ {label_tag} Loaded: {elapsed:.1f}s | {x.element_size() * x.nelement() / 1e6:.1f}MB in VRAM")
    return TensorDataset(x, y)

print("="*70)
print("🔥 KAGGLE TURBO - VRAM ACCELERATED")
print("="*70)

# ============ DATA SCANNING ============
board_dirs = sorted([d for d in glob.glob(f"{tiles_dir}/*") if os.path.isdir(d)])
if not board_dirs:
    print(f"❌ No boards in {tiles_dir}")
    exit(1)

np.random.seed(42)
np.random.shuffle(board_dirs)
split = int(len(board_dirs) * 0.9)
train_dirs, val_dirs = board_dirs[:split], board_dirs[split:]

def get_paths(dirs):
    paths = []
    for d in dirs: paths.extend(glob.glob(os.path.join(d, "*.png")))
    return paths

train_paths = get_paths(train_dirs)
val_paths = get_paths(val_dirs)

# Create VRAM Datasets
train_ds = load_to_vram(train_paths, "TRAIN")
val_ds = load_to_vram(val_paths, "VAL")

# DataLoaders now just slice VRAM (extremely fast)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# ============ MODEL ============
model = UltraEnhancedChessPieceClassifier(
    num_classes=len(FEN_CHARS),
    use_grayscale=USE_GRAYSCALE
).to(device)

if torch.cuda.device_count() > 1:
    print(f"🚀 DataParallel enabled ({torch.cuda.device_count()} GPUs)")
    model = nn.DataParallel(model)

# ============ TRAINING SETUP ============
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LEARNING_RATE*2,
    steps_per_epoch=len(train_loader), epochs=EPOCHS
)
scaler = GradScaler('cuda')

# ============ TRAINING ============
print("\n" + "="*70)
print(f"🚀 TURBO TRAINING - {EPOCHS} Epochs | Batch {BATCH_SIZE}")
print("="*70 + "\n")

total_start = time.time()
best_acc = 0.0

for epoch in range(EPOCHS):
    epoch_start = time.time()
    model.train()
    
    train_loss, correct, total = 0.0, 0, 0
    
    for x, y in train_loader:
        optimizer.zero_grad(set_to_none=True)
        
        with autocast('cuda'):
            outputs = model(x)
            loss = criterion(outputs, y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        _, pred = torch.max(outputs, 1)
        total += y.size(0)
        correct += (pred == y).sum().item()
    
    train_acc = correct / total
    
    # Validation
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            with autocast('cuda'):
                outputs = model(x)
            _, pred = torch.max(outputs, 1)
            val_total += y.size(0)
            val_correct += (pred == y).sum().item()
    
    val_acc = val_correct / val_total
    epoch_time = time.time() - epoch_start
    
    print(f"Epoch {epoch+1:2d}/{EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
          f"Time: {epoch_time:.1f}s | {get_gpu_mem()}")
    
    if val_acc > best_acc:
        best_acc = val_acc
        model_to_save = model.module if isinstance(model, nn.DataParallel) else model
        torch.save(model_to_save.state_dict(), output_model)

print("\n" + "="*70)
print(f"🎉 TURBO COMPLETE! Best Acc: {best_acc:.2%}")
print(f"🕒 Total Time: {format_time(time.time() - total_start)}")
print(f"💾 Model: {output_model}")
print("="*70)
