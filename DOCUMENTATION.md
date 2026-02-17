# 🐉 Chessimg2pos: The Definitive Guide (Industrial Edition)

This document is the "Source of Truth" for the project. It covers the core library, the active training scripts, and the modern tensor-based pipeline.

---

## 🏗️ 1. Core Module (`src/chessimg2pos/`)
The reusable library used by all scripts. **Status: ACTIVE & CRITICAL**

### 🧠 `chessclassifier.py` (Architectures)
*   **`UltraEnhancedChessPieceClassifier` [RECOMMENDED]**: 
    *   **Features**: Simple Attention Mechanism, Dual Pooling (Global Max + Global Average), and a deep MLP head.
    *   **Use Case**: High-accuracy recognition across diverse chess sets.
*   **`EnhancedChessPieceClassifier`**: Mid-tier architecture with Batch Normalization and spatial dropout.
*   **`ChessPieceClassifier`**: Simple 3-layer CNN for fast/low-power inference.

### 🦖 `trainer.py` (Legacy Engine)
*   **Status**: Replaced by direct training scripts for better Kaggle performance, but remains available for custom standard training.

---

## 🚀 2. Active Training Pipeline (The Tensor System)
These are the primary tools for current development.

### 🥇 `train_kaggle_tensor_beast.py` [PRIMARY]
*   **Purpose**: Maximum speed and accuracy using pre-packed tensors.
*   **Optimizations**: 
    *   **RAM-Resident Loading**: Loads entire 5GB dataset into System RAM.
    *   **GPU Augmentation**: Real-time random rotation/noise on the GPU.
    *   **OneCycleLR**: Faster convergence using a cycling learning rate.

### 🎨 `generate_lichess_tensors.py`
*   **Purpose**: Generates millions of synthetic Lichess boards directly into binary tensors.
*   **Variety**: Randomly swaps between Wood, Blue, Green, Marble, and Classic themes.

### 📦 `tensor_packer_kaggle_dataset.py`
*   **Purpose**: Packs millions of PNGs into optimized tensors.
*   **Feature**: Implements **Board-Level Splitting** to ensure the model learns features, not background noise (prevents data leakage).

---

## 🔍 3. Active Inference Scripts
*   **`src/chessimg2pos/predictor.py`**: The production-ready class for your bot. 
*   **`scripts/test_model.py`**: Benchmarking tool for trained `.pt` files.

---

## 📂 4. Legacy Scripts (Keep for reference only)
*   **`scripts/train_kaggle_stream_turbo.py`**: Replaced by Tensor Beast.
*   **`scripts/train_kaggle_optimized.py`**: Replaced by Tensor Beast.
*   **`archive/`**: Contains early experiments and 3D rendering scripts.

---

## 🛠️ 5. The "Industrial" Playbook

1.  **Generate Data**: Use `generate_lichess_tensors.py` for a puzzle-expert model.
2.  **Pack Data**: Use `tensor_packer_kaggle_dataset.py` if you have real-world photos.
3.  **Train**: Run `train_kaggle_tensor_beast.py`.
4.  **Inference**: Deploy the resulting `.pt` model using `ChessPositionPredictor`.

---

## 📈 6. Comparison Table

| Feature | Legacy (v3) | **Modern (Industrial)** |
| :--- | :---: | :---: |
| **Data Format** | PNG Files | **Binary Tensors (.pt)** |
| **Loading Speed** | 1 Hour / Epoch | **3 Minutes / Epoch** |
| **Data Leakage** | Possible (Tile split) | **Fixed (Board split)** |
| **Augmentation** | Static | **GPU-Dynamic** |
| **Stability** | Prone to OOM | **Safe (RAM Managed)** |
