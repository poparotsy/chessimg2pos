#!/usr/bin/env python3
"""
🚀 KAGGLE TENSOR BASIC
- Trains ChessPieceClassifier (basic model) on pre-packed tensors
- Faster training, simpler architecture
"""
import os
import sys
import time
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datetime import timedelta

try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from chessimg2pos.chessclassifier import ChessPieceClassifier

# ============ CONFIG ============
FEN_CHARS = "1RNBQKPrnbqkp"
USE_GRAYSCALE = True
EPOCHS = 30
BATCH_SIZE = 8192  # Larger batch for simpler model
LEARNING_RATE = 0.001

# ============ PATHS ============
base_dir = os.path.dirname(os.path.abspath(__file__))
tensor_dir = os.path.join(base_dir, "tensor_dataset")
output_model = os.path.join(base_dir, "models", "model_basic.pt")
checkpoint_path = os.path.join(base_dir, "models", "checkpoint_basic.pt")
os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)

# ============ DEVICE ============
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def get_gpu_mem():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return f"GPU:{allocated:.1f}/{reserved:.1f}GB"
    return "N/A"

# ============ DATA LOADING ============
print("="*70)
print("🔥 KAGGLE TENSOR BASIC - Simple & Fast")
print("="*70)

print(f"📂 Loading tensors from {tensor_dir}...")
start = time.time()
chunk_files = sorted(glob.glob(os.path.join(tensor_dir, "chunk_*.pt")))

all_x, all_y = [], []
for f in chunk_files:
    data = torch.load(f, map_location='cpu')
    all_x.append(data['x'])
    all_y.append(data['y'])
    print(f"   Loaded {os.path.basename(f)}")

x = torch.cat(all_x, dim=0)
y = torch.cat(all_y, dim=0).long()
print(f"✅ Loaded {x.size(0):,} images in {time.time()-start:.1f}s\n")

# Split
indices = torch.randperm(x.size(0))
split = int(0.9 * x.size(0))
train_idx, val_idx = indices[:split], indices[split:]

train_ds = TensorDataset(x[train_idx], y[train_idx])
val_ds = TensorDataset(x[val_idx], y[val_idx])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ============ MODEL ============
model = ChessPieceClassifier(
    num_classes=len(FEN_CHARS),
    use_grayscale=USE_GRAYSCALE
).to(device)

if torch.cuda.device_count() > 1:
    print(f"🚀 DataParallel enabled ({torch.cuda.device_count()} GPUs)")
    model = nn.DataParallel(model)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler('cuda')

# ============ CHECKPOINT ============
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

# ============ TRAINING ============
print(f"🚀 Training for {EPOCHS} epochs...\n")
total_start = time.time()

for epoch in range(start_epoch, EPOCHS):
    epoch_start = time.time()
    model.train()
    train_loss, correct, total = 0.0, 0, 0
    
    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch.to(device, non_blocking=True).float() / 255.0
        x_batch = (x_batch - 0.5) / 0.5
        y_batch = y_batch.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda'):
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()
        _, pred = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (pred == y_batch).sum().item()
        
        if batch_idx % 50 == 0:
            progress = (batch_idx + 1) / len(train_loader) * 100
            print(f"\rEpoch {epoch+1}/{EPOCHS} [{progress:5.1f}%] Loss: {loss.item():.4f} | {get_gpu_mem()}", end="", flush=True)
    
    train_acc = correct / total
    
    # Validation
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device, non_blocking=True).float() / 255.0
            x_batch = (x_batch - 0.5) / 0.5
            y_batch = y_batch.to(device, non_blocking=True)
            
            with autocast('cuda'):
                outputs = model(x_batch)
            _, pred = torch.max(outputs, 1)
            val_total += y_batch.size(0)
            val_correct += (pred == y_batch).sum().item()
    
    val_acc = val_correct / val_total
    scheduler.step(val_acc)
    
    epoch_time = time.time() - epoch_start
    total_elapsed = time.time() - total_start
    eta = (EPOCHS - epoch - 1) * (total_elapsed / (epoch - start_epoch + 1))
    
    print(f"\n📊 Epoch {epoch+1}/{EPOCHS} | Train: {train_acc:.4f} | Val: {val_acc:.4f} | Time: {format_time(epoch_time)} | ETA: {format_time(eta)}")
    
    # Save
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
        print(f"✨ NEW BEST: {best_acc:.4f}\n")

print(f"\n{'='*70}")
print(f"✅ Training Complete! Best Accuracy: {best_acc:.4f}")
print(f"📁 Model saved: {output_model}")
print(f"{'='*70}")
