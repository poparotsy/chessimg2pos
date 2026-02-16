#!/usr/bin/env python3
"""
🚀 KAGGLE STREAM TURBO
- Streams large datasets into VRAM in chunks, avoiding OOM for large total dataset sizes.
- Parallelized CPU loading for each chunk.
- Uses stable DataParallel (No torchrun/DDP errors).
- Zero data loading time during training once chunk is loaded.

Usage:
  python3 train_kaggle_stream_turbo.py --epochs 25 --batch-size 4096
"""
import os
import sys
import time
import glob
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, TensorDataset
from datetime import timedelta
import concurrent.futures

try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from chessimg2pos.chessclassifier import UltraEnhancedChessPieceClassifier

# ============ ARGUMENT PARSING ============
parser = argparse.ArgumentParser(description='Kaggle Stream Turbo Training')
parser.add_argument('--epochs', type=int, default=25, help='Number of epochs')
parser.add_argument('--batch-size', type=int, default=4096, help='Batch size (Optimized for DataParallel)')
parser.add_argument('--lr', type=float, default=0.003, help='Learning rate')
parser.add_argument('--vram-chunk-size', type=int, default=500000, help='Number of images to load into VRAM per chunk (e.g., 500000)')
parser.add_argument('--num-cpu-workers', type=int, default=4, help='Number of CPU workers for parallel pre-loading (more than 4 might cause errors on Kaggle).')
args = parser.parse_args()

# ============ CONFIG ============
FEN_CHARS = "1RNBQKPrnbqkp"
USE_GRAYSCALE = True # Assume grayscale based on model and memory constraints
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr
VRAM_CHUNK_SIZE = args.vram_chunk_size
NUM_CPU_WORKERS = args.num_cpu_workers if args.num_cpu_workers else (os.cpu_count() if os.cpu_count() else 4)

# ============ PATHS ============
base_dir = os.path.dirname(os.path.abspath(__file__))
tiles_dir = os.path.join(base_dir, "images", "tiles_real")
if not os.path.exists(tiles_dir):
    tiles_dir = os.path.abspath(os.path.join(base_dir, "..", "images", "tiles_real"))

output_model = os.path.join(base_dir, "models", "model_stream_turbo.pt")
checkpoint_path = os.path.join(base_dir, "models", "checkpoint_stream_turbo.pt")
os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)

# ============ DEVICE ============
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def get_gpu_mem():
    if torch.cuda.is_available():
        mem_info = []
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
            mem_info.append(f"GPU{i}:{allocated:.1f}/{reserved:.1f}GB")
        return " ".join(mem_info)
    return "N/A"

# ============ PARALLEL IMAGE PROCESSING FOR CHUNKS ============
def process_single_image(path_idx_tuple, fen_chars, use_grayscale):
    idx, path = path_idx_tuple
    piece_type = path[-5]
    if piece_type not in fen_chars:
        return None # Skip invalid piece types

    try:
        img = Image.open(path).convert("L" if use_grayscale else "RGB")
        img = img.resize((32, 32))
        return idx, np.array(img, dtype=np.uint8), fen_chars.index(piece_type)
    except Exception:
        return None # Skip corrupt images

# ============ STREAMING DATASET ============
class VramStreamingDataset(Dataset):
    def __init__(self, all_image_paths, fen_chars, use_grayscale, vram_chunk_size, num_cpu_workers, device):
        self.all_image_paths = all_image_paths
        self.fen_chars = fen_chars
        self.use_grayscale = use_grayscale
        self.vram_chunk_size = vram_chunk_size
        self.num_cpu_workers = num_cpu_workers
        self.device = device

        self.num_chunks = (len(self.all_image_paths) + self.vram_chunk_size - 1) // self.vram_chunk_size
        self.current_chunk_idx = -1
        self.current_chunk_data = None # (images_tensor, labels_tensor)
        self.next_chunk_future = None
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.num_cpu_workers)
        
        print(f"Dataset Initialized: {len(self.all_image_paths):,} total images, {self.num_chunks} VRAM chunks of {self.vram_chunk_size} each.")

    def _load_chunk_async(self, chunk_idx):
        if chunk_idx >= self.num_chunks:
            return None

        print(f"📦 Pre-loading VRAM chunk {chunk_idx + 1}/{self.num_chunks}...")
        chunk_start_idx = chunk_idx * self.vram_chunk_size
        chunk_end_idx = min((chunk_idx + 1) * self.vram_chunk_size, len(self.all_image_paths))
        chunk_paths_with_idx = [(i, self.all_image_paths[i]) for i in range(chunk_start_idx, chunk_end_idx)]

        start_time = time.time()
        
        # Parallel processing of images within the chunk
        futures = [self.executor.submit(process_single_image, p_idx_tuple, self.fen_chars, self.use_grayscale) 
                   for p_idx_tuple in chunk_paths_with_idx]
        
        processed_results = []
        errors = 0
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                processed_results.append(result)
            else:
                errors += 1
        
        if errors > 0:
            print(f"   ⚠️ Warning: Skipped {errors} invalid or corrupt images in chunk {chunk_idx + 1}.")

        if not processed_results:
            print(f"❌ Error: No valid images loaded for chunk {chunk_idx + 1}!")
            return None # Indicate failed chunk load

        # Sort by original index to maintain order
        processed_results.sort(key=lambda x: x[0])
        imgs_np = [res[1] for res in processed_results]
        lbls_np = [res[2] for res in processed_results]

        # Convert to Tensors
        x = torch.from_numpy(np.stack(imgs_np)).float()
        if self.use_grayscale:
            x = x.unsqueeze(1)
        else:
            x = x.permute(0, 3, 1, 2)
            
        # Normalize
        x = (x - 127.5) / 127.5
        y = torch.tensor(lbls_np, dtype=torch.long)
        
        # Move to GPU NOW
        x = x.to(self.device)
        y = y.to(self.device)
        
        elapsed = time.time() - start_time
        print(f"✅ Chunk {chunk_idx + 1} Loaded: {elapsed:.1f}s | {x.element_size() * x.nelement() / 1e6:.1f}MB in VRAM")
        
        return (x, y)

    def _get_chunk(self, chunk_idx):
        if chunk_idx != self.current_chunk_idx:
            # Clear previous chunk from VRAM
            if self.current_chunk_data:
                del self.current_chunk_data
                torch.cuda.empty_cache()
            
            # Load the current chunk, potentially from pre-fetched future
            if self.next_chunk_future and chunk_idx == self.current_chunk_idx + 1:
                self.current_chunk_data = self.next_chunk_future.result()
                self.next_chunk_future = None # Future consumed
            else:
                self.current_chunk_data = self._load_chunk_async(chunk_idx)
            
            if self.current_chunk_data is None:
                raise ValueError(f"Failed to load VRAM chunk {chunk_idx + 1}")

            self.current_chunk_idx = chunk_idx
            
            # Start pre-fetching the next chunk asynchronously
            if chunk_idx + 1 < self.num_chunks:
                self.next_chunk_future = self.executor.submit(self._load_chunk_async, chunk_idx + 1)
        
        return self.current_chunk_data

    def __len__(self):
        # This dataset doesn't have a fixed length in the traditional sense for __getitem__
        # Instead, it represents the total number of images
        return len(self.all_image_paths)
    
    def __del__(self):
        if self.executor:
            self.executor.shutdown(wait=True)
        if self.current_chunk_data:
            del self.current_chunk_data
            torch.cuda.empty_cache()


print("="*70)
print("🔥 KAGGLE STREAM TURBO - VRAM STREAMING ACCELERATED")
print("="*70)

# ============ DATA SCANNING ============
board_dirs = sorted([d for d in glob.glob(f"{tiles_dir}/*") if os.path.isdir(d)])
if not board_dirs:
    print(f"❌ No boards in {tiles_dir}")
    sys.exit(1)

np.random.seed(42)
np.random.shuffle(board_dirs)
split = int(len(board_dirs) * 0.9)
train_dirs, val_dirs = board_dirs[:split], board_dirs[split:]

def get_paths(dirs):
    paths = []
    for d in dirs: paths.extend(glob.glob(os.path.join(d, "*.png")))
    return paths

print(f"🔍 Found {len(train_dirs):,} training boards and {len(val_dirs):,} validation boards.")
all_train_paths = get_paths(train_dirs)
all_val_paths = get_paths(val_dirs)
print(f"📊 Total training images: {len(all_train_paths):,} | Total validation images: {len(all_val_paths):,}\n")


# Create Streaming Datasets
train_streaming_ds = VramStreamingDataset(
    all_train_paths, FEN_CHARS, USE_GRAYSCALE, VRAM_CHUNK_SIZE, NUM_CPU_WORKERS, device
)
val_streaming_ds = VramStreamingDataset(
    all_val_paths, FEN_CHARS, USE_GRAYSCALE, VRAM_CHUNK_SIZE, NUM_CPU_WORKERS, device
)

print("\n🎉 Initial VRAM chunks loaded. Commencing training loop...\n")


# ============ MODEL ============
model = UltraEnhancedChessPieceClassifier(
    num_classes=len(FEN_CHARS),
    use_grayscale=USE_GRAYSCALE
).to(device)

if torch.cuda.device_count() > 1:
    print(f"🚀 DataParallel enabled ({torch.cuda.device_count()} GPUs)")
    model = nn.DataParallel(model)

# ============ TRAINING SETUP ============
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LEARNING_RATE*2,
    steps_per_epoch= (len(all_train_paths) // BATCH_SIZE), # Approx steps per epoch
    epochs=EPOCHS
)
scaler = GradScaler('cuda')

# ============ CHECKPOINT ============
start_epoch, best_acc = 0, 0.0
if os.path.exists(checkpoint_path):
    print(f"🔄 Loading checkpoint...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    state_dict = ckpt['model_state_dict']
    is_dp = isinstance(model, nn.DataParallel)
    has_prefix = any(k.startswith('module.') for k in state_dict.keys())
    
    if is_dp and not has_prefix:
        state_dict = {f'module.{k}': v for k, v in state_dict.items()}
    elif not is_dp and has_prefix:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if 'scheduler_state_dict' in ckpt: # Scheduler might not be saved in older ckpts
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if 'scaler_state_dict' in ckpt: # Scaler might not be saved in older ckpts
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    
    start_epoch = ckpt['epoch'] + 1
    best_acc = ckpt['best_acc']
    print(f"✅ Resumed from epoch {start_epoch} (Best: {best_acc:.4f})\n")

# ============ TRAINING LOOP ============
print(f"🚀 STREAM TURBO TRAINING - {EPOCHS} Epochs | Batch {BATCH_SIZE} | Chunk Size {VRAM_CHUNK_SIZE}")
print("="*70 + "\n")

total_start = time.time()

for epoch in range(start_epoch, EPOCHS):
    epoch_start = time.time()
    model.train()
    
    train_loss, correct, total = 0.0, 0, 0
    num_batches_processed = 0

    # Iterate through chunks for training
    for chunk_idx in range(train_streaming_ds.num_chunks):
        current_chunk_x, current_chunk_y = train_streaming_ds._get_chunk(chunk_idx)
        chunk_dataset = TensorDataset(current_chunk_x, current_chunk_y)
        chunk_loader = DataLoader(chunk_dataset, batch_size=BATCH_SIZE, shuffle=True)

        for x, y in chunk_loader:
            optimizer.zero_grad(set_to_none=True)
            
            with autocast('cuda'):
                outputs = model(x)
                loss = criterion(outputs, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step() # Step per batch
            
            _, pred = torch.max(outputs, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()
            train_loss += loss.item()
            num_batches_processed += 1
            
            # Print per-batch progress
            if num_batches_processed % 50 == 0: # Update every 50 batches
                progress = (num_batches_processed * BATCH_SIZE) / len(train_streaming_ds) * 100
                print(f"\rEpoch {epoch+1:2d}/{EPOCHS} | Chunk {chunk_idx+1}/{train_streaming_ds.num_chunks} | "
                      f"Batch {num_batches_processed} | Loss: {loss.item():.4f} | Acc: {correct/total:.4f} | "
                      f"Prog: {progress:.1f}% | {get_gpu_mem()}", end="", flush=True)

    train_acc = correct / total
    
    # Validation
    model.eval()
    val_correct, val_total = 0, 0
    val_loss = 0.0
    with torch.no_grad():
        for chunk_idx in range(val_streaming_ds.num_chunks):
            current_chunk_x, current_chunk_y = val_streaming_ds._get_chunk(chunk_idx)
            chunk_dataset = TensorDataset(current_chunk_x, current_chunk_y)
            chunk_loader = DataLoader(chunk_dataset, batch_size=BATCH_SIZE, shuffle=False)

            for x, y in chunk_loader:
                with autocast('cuda'):
                    outputs = model(x)
                
                loss = criterion(outputs, y) # For val_loss reporting
                val_loss += loss.item()

                _, pred = torch.max(outputs, 1)
                val_total += y.size(0)
                val_correct += (pred == y).sum().item()
    
    val_acc = val_correct / val_total
    epoch_time = time.time() - epoch_start
    
    print(f"Epoch {epoch+1:2d}/{EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
          f"Avg Train Loss: {train_loss / num_batches_processed:.4f} | Avg Val Loss: {val_loss / len(val_streaming_ds) * BATCH_SIZE:.4f} | " # Approximate val loss
          f"Time: {epoch_time:.1f}s | {get_gpu_mem()}")
    
    # === SAVE CHECKPOINT ===
    model_to_save = model.module if isinstance(model, nn.DataParallel) else model
    save_dict = {
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_acc': max(best_acc, val_acc)
    }
    torch.save(save_dict, checkpoint_path)

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model_to_save.state_dict(), output_model)
        print(f"   ✨ NEW BEST MODEL SAVED! {best_acc:.4f}")
    
    print() # Newline for next epoch or completion message

print("\n" + "="*70)
print(f"🎉 STREAM TURBO COMPLETE! Best Acc: {best_acc:.2%}")
print(f"🕒 Total Time: {format_time(time.time() - total_start)}")
print(f"💾 Model: {output_model}")
print("="*70)
