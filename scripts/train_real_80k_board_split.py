#!/usr/bin/env python3
"""
Train on real chess board tiles with BOARD-LEVEL splitting to avoid data leakage.
"""
import os
import glob
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from chessimg2pos.chessclassifier import UltraEnhancedChessPieceClassifier
from chessimg2pos.chessdataset import ChessTileDataset, create_image_transforms

base_dir = os.path.dirname(os.path.abspath(__file__))
output_model = os.path.join(base_dir, "models", "model_real_80k_v2.pt")
tiles_dir = os.path.join(base_dir, "images", "tiles_real")

# Config
fen_chars = "1RNBQKPrnbqkp"
use_grayscale = True
epochs = 20
batch_size = 128
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load real data by BOARD (directory)
print(f"📂 Scanning boards in: {tiles_dir}")
board_dirs = [d for d in glob.glob(f"{tiles_dir}/*") if os.path.isdir(d)]
print(f"📊 Found {len(board_dirs)} unique boards")

if len(board_dirs) == 0:
    print("❌ No board directories found. Run: python3 prepare_all_datasets.py")
    exit(1)

# Shuffle boards, not tiles
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
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
print("✅ Datasets ready")

# Create model
print(f"🧠 Creating UltraEnhancedChessPieceClassifier...")
model = UltraEnhancedChessPieceClassifier(
    num_classes=len(fen_chars), 
    use_grayscale=use_grayscale
).to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

print(f"\n🚀 Training with Board-Level Split for {epochs} epochs...\n")

best_acc = 0.0
for epoch in range(epochs):
    model.train()
    train_loss, correct, total = 0.0, 0, 0
    print(f"Epoch {epoch+1}/{epochs} - Training...", end="", flush=True)
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        if batch_idx % 500 == 0:
            print(".", end="", flush=True)
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_acc = correct / total
    print(f" Train: {train_acc:.4f}", end="", flush=True)
    
    model.eval()
    val_correct, val_total = 0, 0
    print(" - Validating...", end="", flush=True)
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_acc = val_correct / val_total
    print(f" Val: {val_acc:.4f}", end="")
    
    scheduler.step(val_acc)
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), output_model)
        print(f" ✅ Saved (best: {best_acc:.4f})")
    else:
        print()

print(f"\n✅ Done! Best reliable accuracy: {best_acc:.2%}")
