#!/usr/bin/env python3
"""
BEAST V4 - High-Throughput Training for Kaggle Dual T4 GPUs.
Optimized with Lazy Board Loading to start instantly.
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
from chessimg2pos.chessdataset import create_image_transforms

# --- Hardware Optimization ---
torch.backends.cudnn.benchmark = True

# --- Config ---
BATCH_SIZE = 2048
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

class LazyChessBoardDataset(Dataset):
    def __init__(self, board_dirs, fen_chars, use_grayscale=True, transform=None):
        self.board_dirs = board_dirs
        self.fen_chars = fen_chars
        self.use_grayscale = use_grayscale
        self.transform = transform
        self.tiles_per_board = 64
        # Cache to avoid repeated disk scans
        self.last_board_idx = -1
        self.cached_tiles = []
        
    def __len__(self):
        return len(self.board_dirs) * self.tiles_per_board

    def __getitem__(self, idx):
        board_idx = idx // self.tiles_per_board
        tile_idx = idx % self.tiles_per_board
        
        # Only scan the directory if we've moved to a new board
        if board_idx != self.last_board_idx:
            board_path = self.board_dirs[board_idx]
            self.cached_tiles = sorted([os.path.join(board_path, f) for f in os.listdir(board_path) if f.endswith('.png')])
            self.last_board_idx = board_idx
        
        if not self.cached_tiles:
            # Fallback if directory is empty
            return torch.zeros((1 if self.use_grayscale else 3, 32, 32)), 0

        image_path = self.cached_tiles[tile_idx % len(self.cached_tiles)]
        piece_type = image_path[-5]
        label = self.fen_chars.index(piece_type)
        
        img = Image.open(image_path)
        img = img.convert('L' if self.use_grayscale else 'RGB')
        if self.transform:
            img = self.transform(img)
            
        return img, label

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

def main():
    print(f"🐉 BEAST V4 (LAZY) - Target: {DEVICE}", flush=True)
    print(f"📂 Tiles Source: {tiles_dir}", flush=True)
    
    if not os.path.exists(tiles_dir):
        print(f"❌ Directory not found: {tiles_dir}", flush=True)
        return

    print("🔍 Listing board directories...", flush=True)
    board_dirs = sorted([f.path for f in os.scandir(tiles_dir) if f.is_dir()])
    
    if not board_dirs:
        print(f"❌ No subdirectories found at {tiles_dir}", flush=True)
        return

    print(f"✅ Found {len(board_dirs)} boards. Total estimated tiles: {len(board_dirs)*64}", flush=True)
    np.random.seed(42)
    np.random.shuffle(board_dirs)
    split = int(len(board_dirs) * 0.8)
    train_dirs, val_dirs = board_dirs[:split], board_dirs[split:]

    num_cpus = os.cpu_count() or 4
    train_ds = LazyChessBoardDataset(train_dirs, FEN_CHARS, USE_GRAYSCALE, create_image_transforms(USE_GRAYSCALE))
    val_ds = LazyChessBoardDataset(val_dirs, FEN_CHARS, USE_GRAYSCALE, create_image_transforms(USE_GRAYSCALE))

    # Boards are already shuffled above, so we set shuffle=False here
    # This allows the Lazy cache to work and kills the 14s lag
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=num_cpus, pin_memory=True, prefetch_factor=4,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=num_cpus, pin_memory=True, prefetch_factor=2,
        persistent_workers=True
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
                # Simple ETA based on current epoch progress
                progress = (i + 1) / len(train_loader)
                batch_time = (time.time() - epoch_start) / (i + 1)
                eta_epoch = batch_time * (len(train_loader) - (i + 1))
                
                mem = get_memory_stats()
                print(f"\rEpoch {epoch+1} [{i}/{len(train_loader)}] | Loss: {loss.item():.4f} | Data: {data_time:.3f}s | {format_time(elapsed)} < {format_time(eta_epoch)} | {mem}", end="", flush=True)
            
            data_start = time.time()

        train_acc = correct / total
        
        model.eval()
        val_correct, val_total = 0, 0
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
        
        print(f"\n📊 Epoch {epoch+1} Summary: Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Time: {format_time(epoch_time)} | ETA: {format_time(eta)}", flush=True)

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
