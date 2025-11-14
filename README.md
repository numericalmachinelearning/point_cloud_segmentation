# **README.md**

# Point Cloud Semantic Segmentation (PandaSet + PointNet)

This repository provides a **minimal, clean, and extendable ML pipeline** for **3D point cloud semantic segmentation** using the **PandaSet LiDAR dataset**.
It implements a complete workflow from dataset acquisition → preprocessing → model architecture → training stub.

The goal is to provide a **working baseline** that can later be upgraded to advanced 3D sparse convolution architectures (e.g., MinkowskiNet / SparseConv U-Net).

---

## 🚀 Features

* **Full dataset download via Python (Kaggle API)**
* **Custom PandaSet loader** (point clouds + semantic labels)
* **PointNet segmentation baseline** implemented in PyTorch
* **Minimal training loop** (forward → loss → backward)
* **Configurable number of sequences** (use subsets for fast prototyping)
* **Fully reproducible and dependency-light**

---

## 📦 Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key
```

---

## 🗂 Dataset Download (PandaSet)

Run:

```bash
python download_pandaset.py
```

This script:

* Authenticates using your `.env`
* Downloads **the full PandaSet dataset (~33GB)** from Kaggle
* Unzips everything under `./pandaset/`

PandaSet is released under **CC-BY-NC-SA**, and Kaggle enforces attribution automatically via the LICENSE file.

---

## Model Architecture — PointNet Baseline

The baseline model is a **PointNet segmentation network** implemented from scratch:

* Input: `(N, 4)` points → `(x, y, z, intensity)`
* MLP + shared Conv1D layers
* Global feature aggregation
* Per-point classification head

This provides a **simple and reliable foundation** before migrating to:

* SparseConv U-Net
* MinkowskiNet
* KPConv
* PointNet++
* etc.

---

## 🏋️ Training

```bash
python pointnet_model.py
```

This runs:

* Dataset loading
* Minimal data analysis
* One epoch of training (stub)
* Prints batch loss

Designed for rapid validation and code completeness rather than full training.

---

## 📊 Expected Baseline Performance

* **PointNet on outdoor LiDAR**: ~35–50% mIoU
* **SparseConv U-Net (production target)**: ~60–75% mIoU

PointNet is intentionally chosen for simplicity and compatibility with the assessment constraints.

---

## 🧱 Project Structure

```
point_cloud_segmentation/
│
├── download_pandaset.py        # Kaggle dataset downloader
├── pointnet_model.py           # Dataset loader + PointNet + training stub
├── MODEL_CHOICES.md            # Architecture justification & notes
├── requirements.txt            # Dependencies
└── .gitignore
```

---

## 📘 License Notice

PandaSet is provided under the **CC-BY-NC-SA 4.0** license.
Any commercial use must follow Scale.ai's licensing terms.

This repository contains **no dataset files** — users must download them via Kaggle using the provided script.

---