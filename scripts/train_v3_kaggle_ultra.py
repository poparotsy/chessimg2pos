"""
🚀 V3 BEAST TRAINER - ULTRA ROBUST PRO (CLEAN)
Features: Dual-GPU, 64x64 RGB, Checkpoint Resumption, and Safe Saving.
Modularized to achieve 10/10 Pylint score.
"""

import glob
import os
import time

# pylint: disable=import-error
import torch
from torch import nn, amp
from torch.utils.data import DataLoader, TensorDataset

# ============ CONFIG ============
BATCH_SIZE = 1024
EPOCHS = 30
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "models/checkpoint_v3.pt"
BEST_MODEL_PATH = "models/model_v3_beast.pt"


# pylint: disable=too-few-public-methods
class StandaloneBeastClassifier(nn.Module):
    """Standalone 64x64 RGB Architecture for Chess Tile Classification."""

    def __init__(self, num_classes=13):
        super().__init__()
        self.conv1 = self._make_block(3, 64)
        self.conv2 = self._make_block(64, 128)
        self.conv3 = self._make_block(128, 256)
        self.conv4 = self._make_block(256, 512)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    @staticmethod
    def _make_block(in_c, out_c):
        """Standard Convolution-Normalization-ReLU-Pooling block."""
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    # pylint: disable=arguments-differ
    def forward(self, x):
        """Passes the input through convolutional and linear layers."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.classifier(x)


def load_v3_data():
    """Loads and prepares the training data from byte-tensors."""
    print("📂 Loading byte-tensors...")
    files = sorted(glob.glob("tensors_v3/train_chunk_*.pt"))
    if not files:
        raise FileNotFoundError("No chunks found in tensors_v3/")

    x_data = torch.cat([torch.load(f, map_location='cpu')['x'] for f in files])
    y_data = torch.cat([torch.load(f, map_location='cpu')['y'] for f in files]).long()

    # Shape Detection
    if x_data.shape[-1] == 3:
        print("🔄 Permuting data to [N, C, H, W]")
        x_data = x_data.permute(0, 3, 1, 2)

    return TensorDataset(x_data, y_data)


def save_checkpoint(model, optimizer, scaler, stats):
    """Saves the current state and best model weights."""
    os.makedirs("models", exist_ok=True)
    raw_model = model.module if isinstance(model, nn.DataParallel) else model

    # Save Checkpoint
    torch.save({
        'epoch': stats['epoch'],
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_loss': min(stats['best_loss'], stats['avg_loss'])
    }, CHECKPOINT_PATH)

    # Save Best Model
    if stats['avg_loss'] < stats['best_loss']:
        torch.save(raw_model.state_dict(), BEST_MODEL_PATH)
        print(f"✨ New Best Model Saved (Loss: {stats['avg_loss']:.4f})")
        return stats['avg_loss']
    return stats['best_loss']


# pylint: disable=too-many-locals
def train():
    """Main training loop with multi-GPU and checkpoint support."""
    print(f"🚀 V3 ULTRA STARTING ON {DEVICE}")

    dataset = load_v3_data()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=4, pin_memory=True)

    model = StandaloneBeastClassifier(num_classes=13).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = amp.GradScaler('cuda')

    start_epoch = 0
    best_loss = float('inf')

    if os.path.exists(CHECKPOINT_PATH):
        print(f"🔄 Resuming from: {CHECKPOINT_PATH}")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('best_loss', float('inf'))

    if torch.cuda.device_count() > 1:
        print(f"🚀 Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    remaining = max(EPOCHS - start_epoch, 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE*2,
        steps_per_epoch=len(loader), epochs=remaining
    )

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_start = time.time()
        running_loss = 0

        for batch_idx, (images, labels) in enumerate(loader):
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE)
            images = (images.float() / 127.5) - 1.0

            with amp.autocast('cuda'):
                loss = criterion(model(images), labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item()

            if batch_idx % 100 == 0:
                sps = (batch_idx * BATCH_SIZE) / (time.time() - epoch_start + 1e-6)
                print(f"   Ep {epoch+1} | {batch_idx}/{len(loader)} "
                      f"| {sps:.0f} img/s | Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(loader)
        print(f"✅ Epoch {epoch+1} Finished | Avg Loss: {avg_loss:.4f}")

        best_loss = save_checkpoint(
            model, optimizer, scaler,
            {'epoch': epoch, 'avg_loss': avg_loss, 'best_loss': best_loss}
        )


if __name__ == "__main__":
    train()
