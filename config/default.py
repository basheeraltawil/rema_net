"""
REMA-Net Configuration
======================
All hyperparameters, paths, and settings in one place.
Edit this file to change training behaviour, model architecture, or data paths.
"""

import os
import torch

# ==============================================================================
# Hardware
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# Training Hyperparameters
# ==============================================================================
LEARNING_RATE = 0.001
NUM_EPOCHS    = 50
BATCH_SIZE    = 16
K_FOLDS       = 3

# ==============================================================================
# Model Architecture
# ==============================================================================
NUM_CLASSES   = 51    # Number of action classes in HMD51
NUM_SEGMENTS  = 20    # Number of frames sampled from each video
NUM_HEADS     = 8     # Number of attention heads in Transformer / MHA
NUM_WORKERS   = 0     # DataLoader workers (0 avoids CUDA multiprocessing errors)

# Feature dimensions
SPATIAL_DIM = 512     # ResNet-18 output feature size
CAPTION_DIM = 128     # BERT-tiny output feature size ([CLS] token)

# ==============================================================================
# Knowledge Distillation
# ==============================================================================
DISTILL_ALPHA       = 0.3   # Weight for the distillation loss term
DISTILL_TEMPERATURE = 3.0   # Softmax temperature (higher = softer distributions)

# ==============================================================================
# Data & Checkpoint Paths
# ==============================================================================
# Dataset-specific paths (data_dir, CSVs, checkpoint dirs) are now managed
# centrally in  config/datasets.py .  Use:
#     from config.datasets import get_dataset_config
#     cfg = get_dataset_config("hmdb51")   # or "ucf101", "haa500"
#
# The defaults below are kept ONLY as legacy fallbacks for scripts that
# have not yet been updated to use --dataset.
# ==============================================================================
DATA_DIR   = "/media/basheer/OVGU/Datasets/hmd51"
TRAIN_CSV  = "/media/basheer/OVGU/Datasets/hmd51/train1.csv"
VAL_CSV    = "/media/basheer/OVGU/Datasets/hmd51/val1.csv"
TEST_CSV   = "/media/basheer/OVGU/Datasets/hmd51/test1.csv"

CKPT_DIR   = "/media/basheer/OVGU/Projects/HAR_Project/rema_checkpoints_after_revision/hmdb51"
CKPT_PATH  = os.path.join(CKPT_DIR, "checkpoint.pth")
FINAL_PATH = os.path.join(CKPT_DIR, "rema_net_hmdb51_final.pth")
BEST_PATH  = os.path.join(CKPT_DIR, "rema_net_hmdb51_best.pth")

# ==============================================================================
# Weights & Biases  (experiment tracking)
# ==============================================================================
WANDB_PROJECT = "MAT Network HAR"
WANDB_API_KEY = "c8d9d016a8275bfc528124784b962f5782613b27"

# ==============================================================================
# Pre-trained Model Names  (downloaded automatically on first run)
# ==============================================================================
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
TEXT_MODEL_NAME = "prajjwal1/bert-tiny"
