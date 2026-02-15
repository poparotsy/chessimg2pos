#!/usr/bin/env python3
"""
BEAST V4 (SAVAGE) - High-Throughput Training for Kaggle Dual T4 GPUs.
Pre-scans paths for maximum GPU utilization (0.000s Data lag).
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
import time
from datetime import datetime, timedelta
from torch.amp import GradScaler, autocast
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from chessimg2pos.chessclassifier import UltraEnhancedChessPieceClassifier
from chessimg2pos.chessdataset import ChessTileDataset, create_image_transforms

# --- Hardware Optimization ---
torch.backends.cudnn.benchmark = True

# --- Config ---
BATCH_SIZE = 4096
EPOCHS = 30
LEARNING_RATE = 0.001
FEN_CHARS = "1RNBQKPrnbqkp"
USE_GRAYSCALE = True

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

base_dir = os.path.dirname(os.path.abspath(__file__))
tiles_dir = os.path.join(base_dir, "images", "tiles_real")
if not os.path.exists(tiles_dir):
    tiles_dir = os.path.join(base_dir, "..", "images", "tiles_real")

output_model = os.path.join(base_dir, "models", "model_80k_beast_v4.pt")
checkpoint_path = os.path.join(base_dir, "models", "checkpoint_80k_beast_v4.pt")
os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1-pt)**self.gamma * ce_loss).mean()

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

def get_memory_stats():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        return f"Mem: {allocated:.2f}/{reserved:.2f} GB"
    return "Mem: N/A"

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def fast_tile_scan(board_dirs, label):
    """Ultra-fast path collection using os.scandir"""
    paths = []
    total = len(board_dirs)
    print(f"🔍 [{label}] Collecting tiles from {total} boards...", flush=True)
    start_t = time.time()
    for i, d in enumerate(board_dirs):
        if i % 5000 == 0 and i > 0:
            elapsed = time.time() - start_t
            speed = i / elapsed
            print(f"  ... {i}/{total} boards ({speed:.1f} boards/sec)", flush=True)
        with os.scandir(d) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(".png"):
                    paths.append(entry.path)
    print(f"✅ [{label}] Done! Found {len(paths)} tiles in {format_time(time.time() - start_t)}", flush=True)
    return np.array(paths)

def main():
    print(f"🐉 BEAST V4 (SAVAGE) - Target: {DEVICE}", flush=True)
    print(f"📂 Tiles Source: {tiles_dir}", flush=True)
    
    if not os.path.exists(tiles_dir):
        print(f"❌ Directory not found: {tiles_dir}", flush=True)
        return

    print("🔍 Listing board directories...", flush=True)
    board_dirs = sorted([f.path for f in os.scandir(tiles_dir) if f.is_dir()])
    
    if not board_dirs:
        print(f"❌ No subdirectories found at {tiles_dir}", flush=True)
        return

    np.random.seed(42)
    np.random.shuffle(board_dirs)
    split = int(len(board_dirs) * 0.8)
    train_dirs, val_dirs = board_dirs[:split], board_dirs[split:]

    # Pre-scan everything into memory for maximum speed
    train_paths = fast_tile_scan(train_dirs, "Train")
    
    # Speed Fix: Only validate on 1000 boards (64k tiles) for speed
    val_subset = val_dirs[:1000]
    val_paths = fast_tile_scan(val_subset, "Val")

    num_cpus = os.cpu_count() or 4
    train_ds = ChessTileDataset(train_paths, FEN_CHARS, USE_GRAYSCALE, create_image_transforms(USE_GRAYSCALE))
    val_ds = ChessTileDataset(val_paths, FEN_CHARS, USE_GRAYSCALE, create_image_transforms(USE_GRAYSCALE))

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=num_cpus, pin_memory=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=num_cpus, pin_memory=True
    )

    model = UltraEnhancedChessPieceClassifier(num_classes=len(FEN_CHARS), use_grayscale=USE_GRAYSCALE).to(DEVICE)
    if DEVICE.type == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion_focal = FocalLoss()
    criterion_smooth = LabelSmoothingLoss(classes=len(FEN_CHARS))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE*2, steps_per_epoch=len(train_loader), epochs=EPOCHS
    )
    scaler = GradScaler('cuda') if DEVICE.type == 'cuda' else None

    start_epoch, best_acc = 0, 0.0
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt['best_acc']

    total_start = time.time()
    for epoch in range(start_epoch, EPOCHS):
        epoch_start = time.time()
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        
        data_start = time.time()
        for i, (inputs, labels) in enumerate(train_loader):
            data_time = time.time() - data_start
            inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=DEVICE.type, enabled=(DEVICE.type != 'cpu')):
                outputs = model(inputs)
                loss = 0.7 * criterion_focal(outputs, labels) + 0.3 * criterion_smooth(outputs, labels)
            
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            scheduler.step()
            
            train_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

            if i % 10 == 0:
                elapsed = time.time() - total_start
                progress = (i + 1) / len(train_loader)
                batch_time = (time.time() - epoch_start) / (i + 1)
                eta_epoch = batch_time * (len(train_loader) - (i + 1))
                mem = get_memory_stats()
                print(f"\rEpoch {epoch+1} [{i}/{len(train_loader)}] | Loss: {loss.item():.4f} | Data: {data_time:.3f}s | {format_time(elapsed)} < {format_time(eta_epoch)} | {mem}", end="", flush=True)
            
            data_start = time.time()

        train_acc = correct / total
        
        model.eval()
        val_correct, val_total = 0, 0
        print("\n🔍 Validating...", flush=True)
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                with torch.autocast(device_type=DEVICE.type, enabled=(DEVICE.type != 'cpu')):
                    outputs = model(inputs)
                _, pred = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (pred == labels).sum().item()
        
        val_acc = val_correct / val_total
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - total_start
        completed = epoch - start_epoch + 1
        eta = (EPOCHS - epoch - 1) * (elapsed / completed)
        
        print(f"📊 Epoch {epoch+1} Summary: Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Time: {format_time(epoch_time)} | ETA: {format_time(eta)}", flush=True)

        save_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': max(best_acc, val_acc)
        }
        torch.save(save_data, checkpoint_path)
        
        if val_acc > best_acc:
            best_acc = val_acc
            m = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(m.state_dict(), output_model)
            print(f"✨ Best model updated! Acc: {best_acc:.4f}", flush=True)

    print(f"\n✅ BEAST Training Complete! Total Time: {format_time(time.time() - total_start)}", flush=True)

if __name__ == "__main__":
    main()
