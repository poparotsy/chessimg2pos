#!/usr/bin/env python3
"""
Optimized version for Kaggle GPU (T4/P100).
Includes AMP (Mixed Precision), Parallel Data Loading, and CUDNN Tuning.
"""
import os
import glob
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import sys
from torch.amp import GradScaler, autocast

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from chessimg2pos.chessclassifier import UltraEnhancedChessPieceClassifier
from chessimg2pos.chessdataset import ChessTileDataset, create_image_transforms

# GPU Optimization: Enable CUDNN auto-tuner
torch.backends.cudnn.benchmark = True

base_dir = os.path.dirname(os.path.abspath(__file__))
# Creating directory for models if it doesn't exist
os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)

output_model = os.path.join(base_dir, "models", "model_real_80k_v3.pt")
checkpoint_path = os.path.join(base_dir, "models", "checkpoint_real_80k_v3.pt")
tiles_dir = os.path.join(base_dir, "images", "tiles_real")

# Config - Optimized for Kaggle
fen_chars = "1RNBQKPrnbqkp"
use_grayscale = True
epochs = 30 # Increased slightly for better convergence with larger batches
batch_size = 1024 # Increased significantly for GPU saturation
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load real data by BOARD (directory)
print(f"📂 Scanning boards in: {tiles_dir}")
board_dirs = sorted([d for d in glob.glob(f"{tiles_dir}/*") if os.path.isdir(d)])
print(f"📊 Found {len(board_dirs)} unique boards")

if len(board_dirs) == 0:
    print(f"❌ No board directories found in {tiles_dir}.")
    exit(1)

np.random.seed(42)
np.random.shuffle(board_dirs)

split_idx = int(len(board_dirs) * 0.8)
train_board_dirs = board_dirs[:split_idx]
test_board_dirs = board_dirs[split_idx:]

print(f"📈 Splitting: {len(train_board_dirs)} boards for training, {len(test_board_dirs)} for validation")

def get_tiles_from_boards(dirs):
    paths = []
    for d in dirs:
        paths.extend(glob.glob(os.path.join(d, "*.png")))
    return np.array(paths)

print("🔍 Collecting tile paths...")
train_paths = get_tiles_from_boards(train_board_dirs)
test_paths = get_tiles_from_boards(test_board_dirs)
print(f"🗂️  Total tiles -> Train: {len(train_paths)}, Val: {len(test_paths)}")

print("🔄 Creating datasets...")
train_dataset = ChessTileDataset(train_paths, fen_chars, use_grayscale, create_image_transforms(use_grayscale))
test_dataset = ChessTileDataset(test_paths, fen_chars, use_grayscale, create_image_transforms(use_grayscale))

# GPU Optimization: num_workers > 0 and pin_memory=True
# Kaggle kernels usually have 2 or 4 vCPUs.
num_cpus = os.cpu_count() or 2
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=num_cpus, 
    pin_memory=True,
    prefetch_factor=2
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=num_cpus, 
    pin_memory=True
)
print(f"✅ Datasets ready (using {num_cpus} workers)")

# Create model
print(f"🧠 Creating UltraEnhancedChessPieceClassifier...")
model = UltraEnhancedChessPieceClassifier(
    num_classes=len(fen_chars), 
    use_grayscale=use_grayscale
).to(device)

# Multi-GPU support
if torch.cuda.device_count() > 1:
    print(f"🚀 Detected {torch.cuda.device_count()} GPUs. Enabling DataParallel.")
    model = nn.DataParallel(model)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

# GPU Optimization: Gradient Scaler for Mixed Precision
scaler = GradScaler('cuda') if device.type == 'cuda' else None

# Resume from checkpoint if exists
start_epoch = 0
best_acc = 0.0
if os.path.exists(checkpoint_path):
    print(f"🔄 Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle DataParallel prefix differences
    state_dict = checkpoint['model_state_dict']
    is_dp_model = isinstance(model, nn.DataParallel)
    # Check if the keys have 'module.' prefix
    has_dp_prefix = any(k.startswith('module.') for k in state_dict.keys())
    
    if is_dp_model and not has_dp_prefix:
        state_dict = {f'module.{k}': v for k, v in state_dict.items()}
    elif not is_dp_model and has_dp_prefix:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if scaler and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_acc = checkpoint['best_acc']
    print(f"✅ Resumed from epoch {start_epoch}, best acc: {best_acc:.4f}")
else:
    print(f"✅ Starting fresh training on {device}")

print(f"\n🚀 Training with Board-Level Split for {epochs} epochs...\n")

for epoch in range(start_epoch, epochs):
    model.train()
    train_loss, correct, total = 0.0, 0, 0
    print(f"Epoch {epoch+1}/{epochs} - Training...", end="", flush=True)
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # GPU Optimization: non_blocking=True
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        # GPU Optimization: set_to_none=True is faster than zero_grad()
        optimizer.zero_grad(set_to_none=True)
        
        # GPU Optimization: Automatic Mixed Precision
        if scaler:
            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if batch_idx % 20 == 0:
            print(".", end="", flush=True)
    
    train_acc = correct / total
    print(f" Train: {train_acc:.4f}", end="", flush=True)
    
    model.eval()
    val_correct, val_total = 0, 0
    print(" - Validating...", end="", flush=True)
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # Using autocast for validation as well
            if scaler:
                with autocast('cuda'):
                    outputs = model(inputs)
            else:
                outputs = model(inputs)
                
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_acc = val_correct / val_total
    print(f" Val: {val_acc:.4f}", end="")
    
    scheduler.step(val_acc)
    
    # Save checkpoint - always save underlying model to keep checkpoints clean
    model_to_save = model.module if isinstance(model, nn.DataParallel) else model
    
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_acc': max(best_acc, val_acc),
    }
    if scaler:
        checkpoint_data['scaler_state_dict'] = scaler.state_dict()
        
    torch.save(checkpoint_data, checkpoint_path)
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model_to_save.state_dict(), output_model)
        print(f" ✅ Saved (best: {best_acc:.4f})")
    else:
        print()
            if scaler:
                with autocast('cuda'):
                    outputs = model(inputs)
            else:
                outputs = model(inputs)
                
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_acc = val_correct / val_total
    print(f" Val: {val_acc:.4f}", end="")
    
    scheduler.step(val_acc)
    
    # Save checkpoint
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_acc': max(best_acc, val_acc),
    }
    if scaler:
        checkpoint_data['scaler_state_dict'] = scaler.state_dict()
        
    torch.save(checkpoint_data, checkpoint_path)
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), output_model)
        print(f" ✅ Saved (best: {best_acc:.4f})")
    else:
        print()

print(f"\n✅ Done! Best reliable accuracy: {best_acc:.2%}")
print(f"💾 Model: {output_model}")
print(f"🔄 Checkpoint: {checkpoint_path}")
