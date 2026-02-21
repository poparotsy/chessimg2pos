import torch
import torch.nn as nn
import glob, os, time
from torch.utils.data import DataLoader, TensorDataset
import sys

# Ensure src is in path
sys.path.append('./src')
from chessimg2pos.chessclassifier import UltraEnhancedChessPieceClassifier

# ============ HIGH PERFORMANCE CONFIG ============
BATCH_SIZE = 1024  # Increased 4x to saturate GPU
EPOCHS = 30
LEARNING_RATE = 2e-3 # Slightly higher LR for larger batch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    print(f"🚀 HIGH-THROUGHPUT TRAINING ON {DEVICE}")
    
    # 1. Load optimized uint8 tensors
    files = sorted(glob.glob("tensors_v3/train_chunk_*.pt"))
    x = torch.cat([torch.load(f)['x'] for f in files])
    y = torch.cat([torch.load(f)['y'] for f in files]).long()
    
    print(f"📊 Total Tiles: {x.shape[0]:,}")
    
    # pin_memory=True speeds up the CPU -> GPU transfer significantly
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                        num_workers=4, pin_memory=True, drop_last=True)

    model = UltraEnhancedChessPieceClassifier(num_classes=13, use_grayscale=False).to(DEVICE)
    
    # Compile model for massive speed boost if using PyTorch 2.0+
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            print("⚡ PyTorch 2.0 Compilation Enabled!")
        except:
            pass

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE*2, 
                                                   steps_per_epoch=len(loader), epochs=EPOCHS)

    # Use GradScaler for faster half-precision training
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(EPOCHS):
        model.train()
        start = time.time()
        
        for batch_idx, (images, labels) in enumerate(loader):
            # Move to GPU with non_blocking=True
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            # Fast Conversion: (Byte to Float)
            # Notice: No .permute() needed because we fixed the generator!
            with torch.cuda.amp.autocast(): # Mixed Precision
                images = (images.float() / 127.5) - 1.0 # Faster than /255 then -0.5/0.5
                outputs = model(images)
                loss = criterion(outputs, labels)

            optimizer.zero_grad(set_to_none=True) # Performance optimization
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            if batch_idx % 50 == 0:
                # Calculate Samples Per Second
                elapsed = time.time() - start
                sps = (batch_idx * BATCH_SIZE) / elapsed if elapsed > 0 else 0
                print(f"   Epoch {epoch+1} | Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f} | {sps:.0f} img/s")

        print(f"✅ Epoch {epoch+1} Finished in {time.time()-start:.1f}s")
        
        # Save Model
        os.makedirs("models", exist_ok=True)
        # Handle compiled model state dict saving
        state = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
        torch.save(state, "models/model_v3_beast.pt")

if __name__ == "__main__":
    train()

