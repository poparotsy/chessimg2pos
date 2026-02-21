"""
🚀 V3 BEAST TRAINER - STANDALONE VERSION
Optimized for 64x64 RGB Chess Tiles.
Includes shape-detection and Pylint-compliant formatting.
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
    """Optimized architecture for 64x64 RGB Chess Tiles."""

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
        """Standard Conv-BN-ReLU-Pool block."""
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    # pylint: disable=arguments-differ
    def forward(self, x):
        """Forward pass through the network."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.classifier(x)
        return x


# ============ CONFIG ============
BATCH_SIZE = 512
EPOCHS = 30
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# pylint: disable=too-many-locals
def train():
    """Main training loop with robust shape handling."""
    print(f"🚀 V3 BEAST STARTING ON {DEVICE}")

    files = sorted(glob.glob("tensors_v3/train_chunk_*.pt"))
    if not files:
        print("❌ Error: No chunks found in tensors_v3/ directory.")
        return

    print("📂 Loading byte-tensors into RAM...")
    x_data = torch.cat([torch.load(f, map_location='cpu')['x'] for f in files])
    y_data = torch.cat([torch.load(f, map_location='cpu')['y'] for f in files]).long()

    # DETECT AND FIX SHAPE: Ensure [N, C, H, W]
    if x_data.shape[-1] == 3:
        print("🔄 Detected [N, H, W, C]. Permuting to [N, C, H, W]...")
        x_data = x_data.permute(0, 3, 1, 2)

    dataset = TensorDataset(x_data, y_data)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True
    )

    model = StandaloneBeastClassifier(num_classes=13).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE*2,
        steps_per_epoch=len(loader), epochs=EPOCHS
    )
    scaler = amp.GradScaler('cuda')

    print(f"📊 Ready. Tiles: {x_data.shape[0]:,} | Input: {x_data.shape[1:]}")

    for epoch in range(EPOCHS):
        model.train()
        epoch_start = time.time()
        running_loss = 0

        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            # Normalization (uint8 [0, 255] -> float32 [-1.0, 1.0])
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

            if batch_idx % 100 == 0:
                elapsed = time.time() - epoch_start
                sps = (batch_idx * BATCH_SIZE) / (elapsed + 1e-6)
                print(f"   Epoch {epoch+1} | {batch_idx}/{len(loader)} "
                      f"| Loss: {loss.item():.4f} | {sps:.0f} img/s")

        avg_l = running_loss / len(loader)
        print(f"✅ Epoch {epoch+1} Finished | Avg Loss: {avg_l:.4f}")

        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/model_v3_beast.pt")


if __name__ == "__main__":
    train()
