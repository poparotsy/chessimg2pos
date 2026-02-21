import torch
import torch.nn as nn
import glob, os, time
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

# IMPORT your existing class
# Note: Ensure you have your src folder in the path
import sys
sys.path.append('./src')
from chessimg2pos.chessclassifier import UltraEnhancedChessPieceClassifier

# ============ CONFIG ============
BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    print(f"🔥 Training on {DEVICE}...")
    
    # 1. Load optimized uint8 tensors
    files = sorted(glob.glob("tensors_v3/train_chunk_*.pt"))
    if not files:
        print("❌ No tensor files found in tensors_v3/")
        return

    x_list, y_list = [], []
    for f in files:
        data = torch.load(f)
        x_list.append(data['x'])
        y_list.append(data['y'])
    
    x = torch.cat(x_list)
    y = torch.cat(y_list).long()
    
    print(f"📊 Dataset Loaded: {x.shape[0]:,} tiles")
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # 2. Initialize Model for RGB + 64px
    # NOTE: If your UltraEnhanced class is hardcoded for 32px, 
    # you may need to adjust the final Linear layer size.
    model = UltraEnhancedChessPieceClassifier(num_classes=13, use_grayscale=False).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE*2, 
                                                   steps_per_epoch=len(loader), epochs=EPOCHS)

    # 3. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        start = time.time()
        
        for batch_idx, (images, labels) in enumerate(loader):
            # A. Move to Device
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # B. ON-THE-FLY NORMALIZATION (uint8 -> float32)
            # [B, 64, 64, 3] -> [B, 3, 64, 64]
            images = images.permute(0, 3, 1, 2).float() / 255.0
            images = (images - 0.5) / 0.5 # Range [-1, 1]

            # C. Forward / Backward
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            if batch_idx % 200 == 0:
                print(f"   Epoch {epoch+1} [{batch_idx}/{len(loader)}] | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"🏁 Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.4f} | Time: {time.time()-start:.1f}s")
        
        # Save progress
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/model_v3_beast.pt")

    print("✨ Training Finished! Model saved as models/model_v3_beast.pt")

if __name__ == "__main__":
    train()

