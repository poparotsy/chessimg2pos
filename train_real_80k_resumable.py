#!/usr/bin/env python3
"""
Train from scratch on real chess board tiles (80k+ images) with checkpoint resumption
"""
import os
import glob
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from chessimg2pos.chessclassifier import UltraEnhancedChessPieceClassifier
from chessimg2pos.chessdataset import ChessTileDataset, create_image_transforms

base_dir = os.path.dirname(os.path.abspath(__file__))
output_model = os.path.join(base_dir, "models", "model_real_80k.pt")
checkpoint_path = os.path.join(base_dir, "models", "checkpoint_real_80k.pt")
tiles_dir = os.path.join(base_dir, "images", "tiles_real")

# Config
fen_chars = "1RNBQKPrnbqkp"
use_grayscale = True
epochs = 20
batch_size = 128
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load real data
print(f"📂 Loading tiles from: {tiles_dir}")
all_paths = np.array(glob.glob(f"{tiles_dir}/*/*.png"))
print(f"📊 Found {len(all_paths)} tiles from 80k+ real boards")

if len(all_paths) == 0:
    print("❌ No tiles found. Run: python3 prepare_all_datasets.py")
    exit(1)

np.random.seed(42)  # Fixed seed for reproducible splits
np.random.shuffle(all_paths)
split = int(len(all_paths) * 0.8)
train_paths, test_paths = all_paths[:split], all_paths[split:]
print(f"📈 Train: {len(train_paths)}, Test: {len(test_paths)}")

print("🔄 Creating datasets...")
train_dataset = ChessTileDataset(train_paths, fen_chars, use_grayscale, create_image_transforms(use_grayscale))
test_dataset = ChessTileDataset(test_paths, fen_chars, use_grayscale, create_image_transforms(use_grayscale))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
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

# Resume from checkpoint if exists
start_epoch = 0
best_acc = 0.0
if os.path.exists(checkpoint_path):
    print(f"🔄 Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_acc = checkpoint['best_acc']
    print(f"✅ Resumed from epoch {start_epoch}, best acc: {best_acc:.4f}")
else:
    print(f"✅ Starting fresh training on {device}")

print(f"\n🚀 Training on 80k+ real boards for {epochs} epochs...\n")

for epoch in range(start_epoch, epochs):
    # Train
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
    
    # Validate
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
    
    # Save checkpoint every epoch
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_acc': max(best_acc, val_acc),
    }, checkpoint_path)
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), output_model)
        print(f" ✅ Saved (best: {best_acc:.4f})")
    else:
        print()

print(f"\n✅ Done! Best accuracy: {best_acc:.2%}")
print(f"💾 Model: {output_model}")
print(f"🔄 Checkpoint: {checkpoint_path}")
print()
print("Test it:")
print("  python3 test_2d_model.py puzzle.png")
