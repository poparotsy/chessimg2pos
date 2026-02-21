"""
🚀 V3 BEAST TRAINER - ULTRA VERSION
Integrates your original robust DataParallel logic with the V3 64x64 Standalone architecture.
"""

import glob
import os
import time

# pylint: disable=import-error
import torch
from torch import nn, amp
from torch.utils.data import DataLoader, TensorDataset


# pylint: disable=too-few-public-methods
class StandaloneBeastClassifier(nn.Module):
    """Standalone 64x64 RGB Architecture."""
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
        """Standard Conv block."""
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
    # pylint: disable=arguments-differ
    def forward(self, x):
        """Forward pass."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.classifier(x)
        return x


# ============ CONFIG ============
BATCH_SIZE = 1024  # Increased for Dual-GPU
EPOCHS = 30
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# pylint: disable=too-many-locals
def train():
    """Main training loop using your original DataParallel logic."""
    print(f"🚀 V3 BEAST STARTING ON {DEVICE}")
    files = sorted(glob.glob("tensors_v3/train_chunk_*.pt"))
    if not files:
        print("❌ Error: No chunks found.")
        return

    x_data = torch.cat([torch.load(f, map_location='cpu')['x'] for f in files])
    y_data = torch.cat([torch.load(f, map_location='cpu')['y'] for f in files]).long()

    if x_data.shape[-1] == 3:
        x_data = x_data.permute(0, 3, 1, 2)
    dataset = TensorDataset(x_data, y_data)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=4, pin_memory=True)
    # INITIALIZE MODEL
    model = StandaloneBeastClassifier(num_classes=13).to(DEVICE)

    # YOUR ORIGINAL DATAPARALLEL LOGIC
    if torch.cuda.device_count() > 1:
        print(f"🚀 Multi-GPU Enabled: Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE*2,
        steps_per_epoch=len(loader), epochs=EPOCHS
    )
    scaler = amp.GradScaler('cuda')

    print(f"📊 Total Tiles: {x_data.shape[0]:,} | Batch Size: {BATCH_SIZE}")

    for epoch in range(EPOCHS):
        model.train()
        epoch_start = time.time()
        running_loss = 0

        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            images = (images.float() / 127.5) - 1.0

            with amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()

            if batch_idx % 50 == 0:
                elapsed = time.time() - epoch_start
                sps = (batch_idx * BATCH_SIZE) / (elapsed + 1e-6)
                print(f"   Epoch {epoch+1} | {batch_idx}/{len(loader)} | {sps:.0f} img/s")

        print(f"✅ Epoch {epoch+1} Finished | Avg Loss: {running_loss/len(loader):.4f}")

        # YOUR ORIGINAL SAFE SAVE LOGIC
        os.makedirs("models", exist_ok=True)
        model_to_save = model.module if isinstance(model, nn.DataParallel) else model
        torch.save(model_to_save.state_dict(), "models/model_v3_beast.pt")


if __name__ == "__main__":
    train()
