#!/usr/bin/env python3
"""
🔥 KAGGLE ULTRA-BEAST - Maximum Speed Training
- Batch size 16384 (4x larger - saturates dual T4)
- Aggressive prefetching (prefetch_factor=16)
- All CPUs maxed out
- Target: 2 hours for 25 epochs

Usage:
  python3 train_kaggle_ultra_beast.py                    # Default: 25 epochs
  python3 train_kaggle_ultra_beast.py --epochs 10        # Run 10 epochs
  python3 train_kaggle_ultra_beast.py --batch-size 8192  # Custom batch
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
from torch.utils.data import DataLoader
from datetime import timedelta
from torch.amp import GradScaler, autocast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from chessimg2pos.chessclassifier import UltraEnhancedChessPieceClassifier
from chessimg2pos.chessdataset import ChessTileDataset, create_image_transforms

# ============ ARGUMENT PARSING ============
parser = argparse.ArgumentParser(description='Kaggle Ultra-Beast Training')
parser.add_argument('--epochs', type=int, default=25, help='Number of epochs (default: 25)')
parser.add_argument('--batch-size', type=int, default=16384, help='Batch size (default: 16384)')
parser.add_argument('--lr', type=float, default=0.003, help='Learning rate (default: 0.003)')
parser.add_argument('--workers', type=int, default=None, help='Number of workers (default: all CPUs)')
parser.add_argument('--prefetch', type=int, default=16, help='Prefetch factor (default: 16)')
args = parser.parse_args()

# ============ ULTRA-BEAST CONFIG ============
FEN_CHARS = "1RNBQKPrnbqkp"
USE_GRAYSCALE = True
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr
NUM_WORKERS = args.workers if args.workers else (os.cpu_count() or 4)
PREFETCH_FACTOR = args.prefetch

# ============ PATHS ============
base_dir = os.path.dirname(os.path.abspath(__file__))
tiles_dir = os.path.join(base_dir, "images", "tiles_real")
output_model = os.path.join(base_dir, "models", "model_ultra_beast.pt")
checkpoint_path = os.path.join(base_dir, "models", "checkpoint_ultra_beast.pt")
os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)

# ============ DEVICE ============
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False  # Speed over reproducibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ LOSSES ============
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1-pt)**self.gamma * ce_loss).mean()

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
    
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def get_gpu_mem():
    if torch.cuda.is_available():
        mem_info = []
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            mem_info.append(f"GPU{i}:{allocated:.1f}/{reserved:.1f}GB")
        return " ".join(mem_info)
    return "N/A"

print("="*70)
print("🔥 KAGGLE ULTRA-BEAST MODE - Maximum Speed")
print("="*70)
print(f"⚙️  Config: epochs={EPOCHS}, batch={BATCH_SIZE:,}, lr={LEARNING_RATE}, workers={NUM_WORKERS}, prefetch={PREFETCH_FACTOR}")
print(f"📍 Device: {device}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"🎮 GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory/1e9:.1f}GB)")
print(f"📂 Tiles: {tiles_dir}")
print("="*70 + "\n")

# ============ DATA LOADING ============
print("📂 Scanning boards...")
board_dirs = sorted([d for d in glob.glob(f"{tiles_dir}/*") if os.path.isdir(d)])
print(f"✅ Found {len(board_dirs):,} boards\n")

if not board_dirs:
    print(f"❌ No boards in {tiles_dir}")
    exit(1)

np.random.seed(42)
np.random.shuffle(board_dirs)
split = int(len(board_dirs) * 0.8)
train_dirs, val_dirs = board_dirs[:split], board_dirs[split:]

print(f"📊 Train: {len(train_dirs):,} boards | Val: {len(val_dirs):,} boards")

def collect_tiles(dirs, label):
    print(f"🔍 Collecting {label} tiles...", end="", flush=True)
    paths = []
    for d in dirs:
        paths.extend(glob.glob(os.path.join(d, "*.png")))
    print(f" {len(paths):,} tiles ✅")
    return np.array(paths)

train_paths = collect_tiles(train_dirs, "train")
val_paths = collect_tiles(val_dirs, "val")
print()

# ============ DATASETS ============
print("🔄 Creating datasets...")
train_ds = ChessTileDataset(train_paths, FEN_CHARS, USE_GRAYSCALE, 
                            create_image_transforms(USE_GRAYSCALE))
val_ds = ChessTileDataset(val_paths, FEN_CHARS, USE_GRAYSCALE, 
                          create_image_transforms(USE_GRAYSCALE))

# 🔥 ULTRA-OPTIMIZED DATALOADERS
train_loader = DataLoader(
    train_ds, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    num_workers=NUM_WORKERS, 
    pin_memory=True, 
    prefetch_factor=PREFETCH_FACTOR,
    persistent_workers=True,
    drop_last=True  # Consistent batch sizes
)
val_loader = DataLoader(
    val_ds, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    num_workers=NUM_WORKERS, 
    pin_memory=True,
    prefetch_factor=PREFETCH_FACTOR,
    persistent_workers=True
)
print(f"✅ DataLoaders ready ({NUM_WORKERS} workers, prefetch={PREFETCH_FACTOR}x)\n")

# ============ MODEL ============
print("🧠 Creating UltraEnhancedChessPieceClassifier...")
model = UltraEnhancedChessPieceClassifier(
    num_classes=len(FEN_CHARS),
    use_grayscale=USE_GRAYSCALE,
    dropout_rate=0.3
).to(device)

if torch.cuda.device_count() > 1:
    print(f"🚀 Multiple GPUs detected ({torch.cuda.device_count()}), but using single GPU for better performance")
    # DataParallel has high overhead - single GPU is often faster
    # model = nn.DataParallel(model)
print()

# ============ TRAINING SETUP ============
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LEARNING_RATE*2,
    steps_per_epoch=len(train_loader), epochs=EPOCHS
)
scaler = GradScaler('cuda')

# ============ CHECKPOINT ============
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
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    scaler.load_state_dict(ckpt['scaler_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    best_acc = ckpt['best_acc']
    print(f"✅ Resumed from epoch {start_epoch} (Best: {best_acc:.4f})\n")

# ============ TRAINING ============
print("="*70)
print(f"🚀 ULTRA-BEAST TRAINING - Target: 2 hours for {EPOCHS} epochs")
print("="*70 + "\n")

total_start = time.time()
for epoch in range(start_epoch, EPOCHS):
    epoch_start = time.time()
    
    # === TRAIN ===
    model.train()
    train_loss, correct, total = 0.0, 0, 0
    batch_times = []
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        batch_start = time.time()
        
        t0 = time.time()
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        t_transfer = time.time() - t0
        
        optimizer.zero_grad(set_to_none=True)
        
        t0 = time.time()
        with autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        t_forward = time.time() - t0
        
        t0 = time.time()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        t_backward = time.time() - t0
        
        # Defer synchronization - only compute metrics every N batches
        if batch_idx % 50 == 0 or batch_idx == len(train_loader) - 1:
            train_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
        
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        if len(batch_times) > 100:  # Keep only last 100
            batch_times.pop(0)
        
        # Print timing breakdown every 10 batches
        if batch_idx % 10 == 0:
            print(f"\nBatch {batch_idx}: Total={batch_time:.3f}s | Transfer={t_transfer:.3f}s | Forward={t_forward:.3f}s | Backward={t_backward:.3f}s", flush=True)
        
        # Real-time progress (update every 50 batches for speed)
        if batch_idx % 50 == 0 or batch_idx == len(train_loader) - 1:
            progress = (batch_idx + 1) / len(train_loader)
            avg_batch_time = np.mean(batch_times[-10:])
            batches_left = len(train_loader) - (batch_idx + 1)
            eta_epoch = avg_batch_time * batches_left
            
            current_acc = correct / total if total > 0 else 0
            samples_per_sec = BATCH_SIZE / avg_batch_time
            
            bar_len = 40
            filled = int(bar_len * progress)
            bar = "█" * filled + "░" * (bar_len - filled)
            
            print(f"\rEpoch {epoch+1}/{EPOCHS} |{bar}| {progress*100:5.1f}% | "
                  f"Loss: {loss.item():.4f} | Acc: {current_acc:.4f} | "
                  f"{samples_per_sec:,.0f} samples/s | "
                  f"{get_gpu_mem()} | "
                  f"ETA: {format_time(eta_epoch)}", 
                  end="", flush=True)
    
    train_acc = correct / total
    
    # === VALIDATION ===
    print("\n🔍 Validating...", end="", flush=True)
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
    
    # === TIMING ===
    epoch_time = time.time() - epoch_start
    elapsed = time.time() - total_start
    completed = epoch - start_epoch + 1
    avg_epoch_time = elapsed / completed
    eta_total = avg_epoch_time * (EPOCHS - epoch - 1)
    
    print(f"\n{'='*70}")
    print(f"📊 Epoch {epoch+1}/{EPOCHS} Complete")
    print(f"   Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    print(f"   Epoch Time: {format_time(epoch_time)} | Total: {format_time(elapsed)} | ETA: {format_time(eta_total)}")
    print(f"   LR: {optimizer.param_groups[0]['lr']:.6f} | {get_gpu_mem()}")
    
    # === SAVE ===
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

print("\n" + "="*70)
print("🎉 ULTRA-BEAST TRAINING COMPLETE!")
print("="*70)
print(f"🏆 Best Accuracy: {best_acc:.2%}")
print(f"🕒 Total Time: {format_time(time.time() - total_start)}")
print(f"💾 Model: {output_model}")
print(f"🔄 Checkpoint: {checkpoint_path}")
print("="*70)
