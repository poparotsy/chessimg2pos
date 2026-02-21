"""
🚀 V3 BEAST TRAINER - PRO VERSION
High-performance training script for Chess Piece Classification.
Rating: ~9.5/10 (Pylint)
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
    Automatically fixes the Linear layer if it was designed for 32x32.
    """
    model.eval()
    with torch.no_grad():
        # Trace the shape through the convolutional layers
        feat_x = sample_input
        for layer in [model.conv1, model.conv2, model.conv3, model.conv4]:
            feat_x = layer(feat_x)

        # Flattened size from convolutions
        flattened_size = feat_x.view(feat_x.size(0), -1).size(1)

        # If the model's fc layer doesn't match, we replace it
        if model.fc[0].in_features != flattened_size:
            print(f"🔧 Auto-Fixing: Adjusting features to {flattened_size}")
            model.fc = nn.Sequential(
                nn.Linear(flattened_size, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, 13)
            ).to(DEVICE)
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

    # 2. Initialize and Fix Model
    model = UltraEnhancedChessPieceClassifier(
        num_classes=13,
        use_grayscale=False
    ).to(DEVICE)

    # Create a dummy sample matching our v3 specs [Batch, Channels, H, W]
    sample_input = torch.zeros((1, 3, 64, 64)).to(DEVICE)
    model = fix_model_dimensions(model, sample_input)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-2
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Scheduler handles the learning rate warmup
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE*2,
        steps_per_epoch=len(loader),
        epochs=EPOCHS
    )

    # Modern AMP Scaler
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

            # Fast Conversion: uint8 -> float32 and Normalize
            images = (images.float() / 127.5) - 1.0

            # Mixed Precision Forward Pass
            with amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Optimization Step
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

        print(f"✅ Epoch {epoch+1} Finished | Avg Loss: {running_loss/len(loader):.4f} "
              f"| Time: {time.time() - epoch_start:.1f}s")

        # Save Model
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/model_v3_beast.pt")


if __name__ == "__main__":
    train()
