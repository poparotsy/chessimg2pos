import torch
import torch.nn as nn
import glob
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

# IMPORT your existing class, but we must override the input layer
from src.chessimg2pos.chessclassifier import UltraEnhancedChessPieceClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    # Load all chunks
    files = glob.glob("tensors_v3/train_*.pt")
    x = torch.cat([torch.load(f)['x'] for f in files])
    y = torch.cat([torch.load(f)['y'] for f in files]).long()
    
    loader = DataLoader(TensorDataset(x, y), batch_size=256, shuffle=True)

    # INITIALIZE MODEL
    # Ensure you modify the UltraEnhanced class or use this wrapper:
    model = UltraEnhancedChessPieceClassifier(num_classes=13, use_grayscale=False).to(device)
    
    # If the original code was 32x32, the Linear layers will error. 
    # You may need to adjust the Final Linear layer in chessclassifier.py to 
    # handle the larger flattened spatial dimension (64x64 -> 32x32 pool etc).
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    model.train()
    for epoch in range(20):
        total_loss = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            # Additional GPU-side augmentation
            # images = transforms.ColorJitter(0.2, 0.2)(images) 

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f}")
        torch.save(model.state_dict(), "models/model_v3_beast.pt")

if __name__ == "__main__":
    train()

