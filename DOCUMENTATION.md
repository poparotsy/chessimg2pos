# 🐉 Chessimg2pos: The Definitive Guide (v4 Beast Edition)

This document is the "Source of Truth" for the project. It covers the core library, the active training scripts, and the legacy artifacts.

---

## 🏗️ 1. Core Module (`src/chessimg2pos/`)
The reusable library used by all scripts. **Status: ACTIVE & CRITICAL**

### 🧠 `chessclassifier.py` (Architectures)
*   **`UltraEnhancedChessPieceClassifier` [RECOMMENDED]**: 
    *   **Features**: Simple Attention Mechanism (focuses on piece outlines), Dual Pooling (Global Max + Global Average), and a deep MLP head.
    *   **Use Case**: High-accuracy recognition across diverse chess sets (Kaggle/80k).
*   **`EnhancedChessPieceClassifier`**: Mid-tier architecture with Batch Normalization and spatial dropout.
*   **`ChessPieceClassifier`**: Simple 3-layer CNN for fast/low-power inference.

### 🦖 `trainer.py` (The Super Trainer)
*   **Class: `ChessRecognitionTrainer`**: The "High-Level" engine.
*   **Advanced Features**:
    *   **Focal Loss**: Mathematically forces the model to ignore "easy" squares and focus on hard piece distinctions (like Queen vs Rook).
    *   **Label Smoothing**: Prevents the model from becoming "cocky," ensuring it remains sensitive to noise and blur.
    *   **Cosine Annealing**: A cycling learning rate that helps the model escape local minima.

### 🔮 `predictor.py` (The Brain)
*   **Class: `ChessPositionPredictor`**: Converts pixels to FEN.
*   **Capabilities**: Heatmap generation, confidence scoring per square, and Lichess-link generation.

---

## 🚀 2. Active Training Scripts (Root & Scripts/)
These are the "Usable" scripts for current development.

### 🥇 `train_80k_beast_v4.py` [STATUS: PRIMARY BEAST]
*   **Purpose**: The most aggressive training script in the repo, built for Kaggle Dual T4/P100.
*   **Optimizations**: 
    *   **Batch Size 2048**: Satures GPUs to stop idle time.
    *   **Persistent Workers**: Solves the CPU-loading bottleneck.
    *   **Mixed Precision (AMP)**: Speeds up training on Tensor Cores.
*   **Usage**: Run this on Kaggle for 1-2 days to get a 99.9%+ accuracy model.

### 📦 `scripts/prepare_all_datasets.py` [STATUS: UTILITY]
*   **Purpose**: Processes millions of raw tiles from GitHub/Kaggle sources.
*   **Feature**: Implements "Board-Level Splitting" to ensure the model learns *chess* and doesn't just memorize specific background pixels from the training set.

### 🎯 `scripts/generate_queen_rook_data.py` [STATUS: UTILITY]
*   **Purpose**: Programmatically generates thousands of boards with only Queens and Rooks. 
*   **Strategy**: Use this if the 80k dataset still shows confusion between these two pieces.

---

## 🔍 3. Active Inference Scripts
*   **`scripts/chess_recognizer_ultra.py`**: The production-ready script for your bot. Returns JSON with detailed confidence stats.
*   **`scripts/chess_recognizer.py`**: The standard version using the `Enhanced` model.

---

## 📂 4. Legacy Scripts (The "Old" Files)
These files are kept for historical reference or specific edge cases. **Status: LEGACY**

*   **`train_real_80k_resumable_v3_kaggle.py`**: Replaced by v4 Beast. Uses standard Adam and no focal loss.
*   **`train_3d_rendered.py`**: Used for the initial 3D dataset (prior to the 80k real image collection).
*   **`train_ultra_full.py`**: A wrapper for the `Trainer` class; functional but less optimized for Kaggle than the Beast script.
*   **`train_ultra_from_tiles.py` / `train_ultra_model.py`**: Early experiments with the Ultra architecture.

---

## 🛠️ 5. The "Beast" Playbook (How to Rerun Training)

1.  **Prepare Data**: Ensure tiles are at `../images/tiles_real/` (or symlinked on Kaggle).
2.  **Select Architecture**: The Beast script defaults to `UltraEnhanced`.
3.  **Tune Aggression**: If your GPU has 16GB+ VRAM, check `train_80k_beast_v4.py` and ensure `BATCH_SIZE` is at least 1024.
4.  **Monitor**: Watch the `Data:` and `Mem:` telemetry.
    *   If `Data` > 0.5s: Increase `num_workers`.
    *   If `Mem` < 5GB: Double your `BATCH_SIZE`.

---

## 📈 6. Feature Comparison Table

| Feature | Standard (v2) | Kaggle (v3) | **BEAST (v4)** |
| :--- | :---: | :---: | :---: |
| **Loss** | CrossEntropy | CrossEntropy | **Focal + Smoothing** |
| **Batch Size** | 32 | 1024 | **2048+** |
| **Optimizer** | Adam | Adam | **AdamW + Clipping** |
| **Speed** | Slow | Fast | **Maximum** |
| **Q vs R Focus** | No | No | **YES** |
