"""
🚀 V3 BEAST TRAINER - FINAL PRO VERSION
High-performance training script for Chess Piece Classification.
Directly targets fc1, fc2, fc3 from the chessimg2pos source.
"""

import glob
import os
import sys
import time

# pylint: disable=import-error
import torch
from torch import nn, amp
from torch.utils.data import DataLoader, TensorDataset

# Ensure src is in path for local imports
sys.path.append('./src')

# pylint: disable=wrong-import-position
from chessimg2pos.chessclassifier import UltraEnhancedChessPieceClassifier

# ============ CONFIG ============
BATCH_SIZE = 512
EPOCHS = 30
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fix_model_dimensions(model, sample_input):
    """
    Expert Developer Patch:
    Replaces fc1, fc2, and fc3 to handle 64x64 RGB input
    matching the architecture in your chessclassifier.py.
    """
    model.eval()
    with torch.no_grad():
        # Trace through convolutional layers: conv1 -> conv2 -> conv3 -> conv4
        feat_x = sample_input
        for layer in [model.conv1, model.conv2, model.conv3, model.conv4]:
            feat_x = layer(feat_x)

        # Calculate the new flattened size (Expected: 8192 for 64x64)
        flattened_size = feat_x.view(feat_x.size(0), -1).size(1)

        # Check fc1 attribute specifically
        if model.fc1.in_features != flattened_size:
            print(f"🔧 Auto-Fixing: Adjusting fc1 input to {flattened_size}")

            # Re-initialize the individual FC layers to match source structure
            # Source uses: fc1(in, 512) -> fc2(512, 256) -> fc3(256, 13)
            model.fc1 = nn.Linear(flattened_size, 512).to(DEVICE)
            model.fc2 = nn.Linear(512, 256).to(DEVICE)
            model.fc3 = nn.Linear(256, 13).to(DEVICE)

    return model


# pylint: disable=too-many-locals
def train():
    """
    Main training loop.
    Loads data, fixes model dimensions, and runs optimized training.
    """
    print(f"🚀 V3 BEAST TRAINER STARTING ON {DEVICE}")

    # 1. Load optimized uint8 tensors
    files = sorted(glob.glob("tensors_v3/train_chunk_*.pt"))
    if not files:
        print("❌ Error: No chunks found in tensors_v3/")
        return

    print("📂 Loading data into memory...")
    # Load to CPU first to prevent GPU memory spikes
    x_data = torch.cat([torch.load(f, map_location='cpu')['x'] for f in files])
    y_data = torch.cat([torch.load(f, map_location='cpu')['y'] for f in files]).long()

    dataset = TensorDataset(x_data, y_data)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # 2. Initialize Model (3 channels for RGB)
    model = UltraEnhancedChessPieceClassifier(
        num_classes=13,
        use_grayscale=False
    ).to(DEVICE)

    # Apply the fix for 64x64 dimensions
    sample_input = torch.zeros((1, 3, 64, 64)).to(DEVICE)
    model = fix_model_dimensions(model, sample_input)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-2
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Scheduler for OneCycle learning rate policy
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE*2,
        steps_per_epoch=len(loader),
        epochs=EPOCHS
    )

    # Mixed Precision Scaler
    scaler = amp.GradScaler('cuda')

    print(f"📊 Ready. Total Tiles: {x_data.shape[0]:,}")
    print(f"🏁 Starting {EPOCHS} Epochs...")

    for epoch in range(EPOCHS):
        model.train()
        epoch_start = time.time()
        running_loss = 0

        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            # Fast Conversion: uint8 [0,255] -> float32 [-1,1]
            images = (images.float() / 127.5) - 1.0

            # Mixed Precision Forward
            with amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Backward / Optimize
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()

            if batch_idx % 100 == 0:
                elapsed = time.time() - epoch_start
                sps = (batch_idx * BATCH_SIZE) / elapsed if elapsed > 0 else 0
                print(f"   Epoch {epoch+1} | {batch_idx}/{len(loader)} "
                      f"| Loss: {loss.item():.4f} | {sps:.0f} img/s")

        avg_loss = running_loss / len(loader)
        print(f"✅ Epoch {epoch+1} Finished | Avg Loss: {avg_loss:.4f} "
              f"| Time: {time.time() - epoch_start:.1f}s")

        # Save weights
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/model_v3_beast.pt")


if __name__ == "__main__":
    train()
