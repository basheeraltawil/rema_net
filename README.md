# REMA-Net — Robust Embedded Multimodal Action Network

> A clean, modular reorganisation of the REMA-Net codebase for **Human Action Recognition (HAR)**.  
> Everything you need to **train**, **evaluate**, and **run real-time inference** is here.

---

## 📁 Project Structure

```
rema_net/
│
├── config/                         # ⚙️  All settings in one place
│   ├── default.py                  #     Hyperparameters, paths, model config
│   └── datasets.py                 #     Dataset registry (HMDB51, UCF101, HAA500)
│
├── models/                         # 🧠  Neural network definitions
│   ├── temporal_transformer.py     #     PositionalEncoding + TemporalTransformer
│   ├── streams.py                  #     SpatialStream, TemporalStream, CaptionStream, CaptionGating, MHA
│   └── rema_net.py                 #     REMANet (main multi-stream model)
│
├── data/                           # 📦  Data loading & preprocessing
│   ├── dataset.py                  #     ActionRecognitionDataset + collate function
│   └── transforms.py               #     RGB / Flow / Augmented transform pipelines
│
├── tools/                          # 🔧  Standalone utilities
│   ├── benchmark_captioning.py     #     Measure BLIP+BERT latency/FPS
│   └── verify_model_shapes.py      #     Sanity-check tensor dimensions
│
├── train.py                        # 🏋️  Training entry point
├── test.py                         # 📊  Evaluation & confusion matrices
├── inference.py                    # 🎥  Real-time video inference demo
└── README.md                       # 📖  This file
```

---

## 🧠 Model Overview

REMA-Net fuses **four streams** for robust action recognition:

| # | Stream | Input | Backbone | Purpose |
|---|--------|-------|----------|---------|
| 1 | **Spatial (RGB)** | Colour frames | ResNet-18 + TemporalTransformer | Appearance features |
| 2 | **Temporal (Flow)** | Optical flow | ResNet-18 + TemporalTransformer | Motion features |
| 3 | **Caption (BLIP)** | PIL images → text | BLIP + BERT-tiny (frozen) | Semantic anchor |
| 4 | **Gated Fusion** | RGB feat ⊕ Caption feat | Learned gate (0–1) | Blends vision & language |

During **training**, the caption stream acts as a *teacher* via **cross-modal knowledge distillation** — the spatial and temporal streams learn to mimic its outputs.

During **inference**, predictions from all four streams are **averaged** for the final result.

---

## 🛠️ Prerequisites

```bash
# 1. Create a conda environment (recommended)
conda create -n cuda_venv python=3.10 -y
conda activate cuda_venv

# 2. Install PyTorch with CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. Install all other dependencies
pip install -r requirements.txt
```

> **CPU-only?** Skip step 2 — `pip install -r requirements.txt` will install a CPU-only PyTorch.

---

## 🚀 How to Use — Step by Step

> **All commands below assume you are inside the `rema_net/` directory.**

```bash
cd /media/basheer/OVGU/Projects/HAR_Project/HAR_NETWORK_Journalv2/REMA_NET_AFTER_REVISION/rema_net
```

---

### Step 1 — Supported Datasets

REMA-Net supports three datasets out of the box. Pass `--dataset` to switch:

| Dataset | Classes | Format | `--dataset` flag |
|---------|---------|--------|-----------------|
| **HMDB51** | 51 | CSV (`clip_path`, `label`, `label_original`) | `hmdb51` (default) |
| **UCF101** | 101 | TXT (space-separated, 1-indexed labels) | `ucf101` |
| **HAA500** | 500 | CSV (`clip_path`, `label`, `label_original`) | `haa500` |

Dataset paths are configured in `config/datasets.py`. Open it and adjust `data_dir` / CSV / TXT paths to match your machine.

Checkpoints are saved to `rema_checkpoints_after_revision/<dataset>/` (configured in `config/datasets.py` → `CKPT_BASE`).

---

### Step 2 — Verify the Model Architecture

Before training, run a quick shape-check to make sure everything is wired correctly:

```bash
python -m tools.verify_model_shapes
```

Expected output:
```
✓ Forward pass successful!
  Spatial  output : torch.Size([2, 51])
  Temporal output : torch.Size([2, 51])
  Caption  output : torch.Size([2, 51])
  Fused    output : torch.Size([2, 51])
  Gate     weight : torch.Size([2, 1])
✓ All shapes match expected (2, 51)
```

---

### Step 3 — Train the Model

```bash
# HMDB51 (default)
python train.py --dataset hmdb51

# UCF101
python train.py --dataset ucf101

# HAA500
python train.py --dataset haa500
```

**With custom arguments:**

```bash
python train.py --dataset ucf101 --epochs 50 --batch_size 16 --k_folds 3
```

**What happens during training:**
1. Train and val splits are combined, then split into K folds.
2. For each fold a fresh REMANet is created with ImageNet-pretrained ResNet-18.
3. Training uses cross-modal knowledge distillation (caption → spatial/temporal/fused).
4. Best model (by val accuracy) and per-epoch checkpoints are saved.
5. Metrics are logged to Weights & Biases.

**Outputs** (saved to `rema_checkpoints_after_revision/<dataset>/`):
- `rema_net_<dataset>_best.pth` — best validation accuracy weights
- `checkpoint_epoch_N.pth` — resumable checkpoints (model + optimizer + scheduler)
- `rema_net_<dataset>_final.pth` — final weights after the last fold

**Resume training** — if a checkpoint exists it will be loaded automatically.

---

### Step 4 — Evaluate on Test Set

```bash
python test.py --dataset hmdb51 --checkpoint /path/to/best_model.pth
python test.py --dataset ucf101 --checkpoint /path/to/best_model.pth
python test.py --dataset haa500 --checkpoint /path/to/best_model.pth
```

**Outputs:**
- Per-stream accuracy printed to console (Spatial, Temporal, Caption, Gated, Ensemble)
- `cm_spatial.png`, `cm_temporal.png`, `cm_caption.png`, `cm_gated.png`, `cm_ensemble.png` — confusion matrix plots
- `stream_comparison.png` — bar chart comparing all streams

---

### Step 5 — Real-Time Inference

#### A) Multimodal Hybrid Mode (all streams, highest accuracy)

```bash
# Webcam
python inference.py --dataset hmdb51 --checkpoint /path/to/best_model.pth --video 0

# Video file
python inference.py --dataset ucf101 --checkpoint /path/to/best_model.pth --video /path/to/video.mp4
```

#### B) Visual-Only Distilled Mode (no BLIP, 100+ FPS on GPU)

```bash
python inference.py --dataset haa500 --checkpoint /path/to/best_model.pth --video 0 --no_caption
```

**Hotkeys while running:**
| Key | Action |
|-----|--------|
| `s` | Save a screenshot to `./real_time_vis/` |
| `q` | Quit and print the Performance Summary Report |

**Outputs on exit:**
- Performance report printed to console (latency, FPS, P50/P95, GPU memory)
- `Multimodal-Hybrid.csv` or `Visual-Only-Distilled.csv` — summary stats
- `*_LOGS.csv` — per-frame detailed log

---

### Step 6 — Benchmark the Captioning Pipeline

Measure how fast BLIP + BERT-tiny runs on your hardware:

```bash
python -m tools.benchmark_captioning
```

**Output:**
- Average latency (ms) and FPS printed to console
- `caption_benchmark_results.txt` saved to the project root

---

## ⚙️ Key Configuration Reference (`config/default.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_CLASSES` | 51 | Number of action classes (HMD51) |
| `NUM_SEGMENTS` | 20 | Frames sampled per video |
| `NUM_HEADS` | 8 | Attention heads |
| `LEARNING_RATE` | 0.001 | Initial LR (decays ×0.1 every 10 epochs) |
| `BATCH_SIZE` | 16 | Training batch size |
| `K_FOLDS` | 3 | Cross-validation folds |
| `DISTILL_ALPHA` | 0.3 | Weight of KD loss |
| `DISTILL_TEMPERATURE` | 3.0 | Softmax temperature for KD |
| `NUM_WORKERS` | 0 | DataLoader workers (0 = safest for CUDA) |

---

## 🗂️ Dataset Format

The dataset root should contain **one sub-folder per action class**, each holding video files.  
Annotations are CSV files with these columns:

| Column | Type | Example |
|--------|------|---------|
| `clip_path` | str | `brush_hair/clip_001.avi` |
| `label` | int | `0` |
| `label_original` | str | `brush_hair` *(optional, for display)* |

---

## 📝 Notes

- The original scripts in the parent directory are **untouched**. This `rema_net/` folder is a clean, modular copy.
- First-run captioning will download BLIP (~1 GB) and BERT-tiny (~17 MB) from Hugging Face.
- Extracted frames and embeddings are **cached** next to the video files (`*_cache.pt`) for fast subsequent loading.
- All confusion-matrix and CSV outputs are saved in the **current working directory** (i.e. `rema_net/`).

---

## 📚 Further Reading

| Document | Location (parent directory) |
|----------|---------------------------|
| Technical details (math formulations) | `../technical_details.md` |
| Review response (point-by-point) | `../review_response.md` |
| System enhancements report | `../system_enhancements_report.md` |
