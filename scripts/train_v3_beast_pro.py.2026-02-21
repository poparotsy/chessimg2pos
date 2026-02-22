"""
🚀 V3 BEAST TRAINER - PRO ULTRA
Modularized for 10/10 Pylint score.
Features: Weighted Loss, Dual-GPU, 64x64 RGB, and Checkpoints.
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
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
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


def get_training_setup(model, train_loader, start_epoch):
    """Initializes optimizer, criterion, and scheduler."""
    weights = torch.ones(13).to(DEVICE)
    weights[0], weights[1:] = 0.1, 2.0  # Weighted Loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.05)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    remaining = max(EPOCHS - start_epoch, 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE * 2,
        steps_per_epoch=len(train_loader), epochs=remaining
    )
    return optimizer, criterion, scheduler


def save_v3_checkpoint(model, optimizer, scaler, stats):
    """Saves checkpoint and best model."""
    os.makedirs("models", exist_ok=True)
    raw = model.module if hasattr(model, 'module') else model
    torch.save({
        'epoch': stats['epoch'],
        'model_state_dict': raw.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_loss': min(stats['best_loss'], stats['avg_loss'])
    }, CHECKPOINT_PATH)
    if stats['avg_loss'] < stats['best_loss']:
        torch.save(raw.state_dict(), BEST_MODEL_PATH)
        return stats['avg_loss']
    return stats['best_loss']


def train():
    """Main training loop."""
    print(f"🚀 V3 ULTRA STARTING ON {DEVICE}")
    files = sorted(glob.glob("tensors_v3/train_chunk_*.pt"))
    x_data = torch.cat([torch.load(f, map_location='cpu')['x'] for f in files])
    y_data = torch.cat([torch.load(f, map_location='cpu')['y']
                       for f in files]).long()
    if x_data.shape[-1] == 3:
        x_data = x_data.permute(0, 3, 1, 2)

    loader = DataLoader(TensorDataset(x_data, y_data), batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=4, pin_memory=True)
    model = StandaloneBeastClassifier(num_classes=13).to(DEVICE)
    scaler, start_epoch, best_loss = amp.GradScaler('cuda'), 0, float('inf')

    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch, best_loss = ckpt['epoch'] + \
            1, ckpt.get('best_loss', float('inf'))

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer, criterion, scheduler = get_training_setup(
        model, loader, start_epoch)

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_start, running_loss = time.time(), 0
        for _, (imgs, lbls) in enumerate(loader):
            imgs, lbls = imgs.to(DEVICE, non_blocking=True), lbls.to(DEVICE)
            with amp.autocast('cuda'):
                loss = criterion(model((imgs.float() / 127.5) - 1.0), lbls)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        print(
            f"✅ Epoch {
                epoch +
                1} | Loss: {
                avg_loss:.4f} | Time: {
                time.time() -
                epoch_start:.1f}s")
        best_loss = save_v3_checkpoint(model, optimizer, scaler,
                                       {'epoch': epoch, 'avg_loss': avg_loss, 'best_loss': best_loss})


if __name__ == "__main__":
    train()
