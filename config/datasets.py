"""
Dataset Protocols
=================
Central registry of all supported datasets.

Each entry defines:
    - data_dir      : root directory containing video files
    - train_csv     : training annotations (CSV with clip_path + label columns)
    - val_csv       : validation annotations
    - test_csv      : test annotations
    - num_classes   : number of action categories
    - name          : human-readable name (used in wandb, prints, checkpoint dirs)

To add a new dataset, just add a new key to DATASETS below and, if needed,
a custom parser function.

Usage from other scripts:
    from config.datasets import get_dataset_config
    cfg = get_dataset_config("ucf101")   # returns a dict
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ==============================================================================
# Base checkpoint directory  (one sub-folder per dataset)
# ==============================================================================
CKPT_BASE = "/media/basheer/OVGU/Projects/HAR_Project/rema_checkpoints_after_revision"


# ==============================================================================
# Dataset registry
# ==============================================================================
DATASETS = {
    # ------------------------------------------------------------------
    "hmdb51": {
        "name":        "HMDB51",
        "data_dir":    "/media/basheer/OVGU/Datasets/hmd51",
        "train_csv":   "/media/basheer/OVGU/Datasets/hmd51/train1.csv",
        "val_csv":     "/media/basheer/OVGU/Datasets/hmd51/val1.csv",
        "test_csv":    "/media/basheer/OVGU/Datasets/hmd51/test1.csv",
        "num_classes": 51,
    },
    # ------------------------------------------------------------------
    "ucf101": {
        "name":        "UCF101",
        "data_dir":    "/media/basheer/OVGU/Datasets/ucf101/UCF-101",
        "test_data_dir": "/media/basheer/OVGU/Datasets/ucf101/UCF-101",
        "train_txt":   "/media/basheer/OVGU/Datasets/ucf101/trainlist01.txt",
        "test_txt":    "/media/basheer/OVGU/Datasets/ucf101/testlist01.txt",
        "num_classes": 101,
    },
    # ------------------------------------------------------------------
    "haa500": {
        "name":        "HAA500",
        "data_dir":    "/media/basheer/OVGU/Datasets/haa500_v1_1",
        "train_csv":   "/media/basheer/OVGU/Datasets/haa500_v1_1/train.csv",
        "val_csv":     "/media/basheer/OVGU/Datasets/haa500_v1_1/val.csv",
        "test_csv":    "/media/basheer/OVGU/Datasets/haa500_v1_1/test.csv",
        "num_classes": 500,
    },
}


# ==============================================================================
# UCF-101 parser  (space-separated TXT → DataFrame with clip_path, label)
# ==============================================================================

def _parse_ucf101_txt(txt_path, has_labels=True):
    """
    Parse UCF-101 annotation files.

    Train format : ``ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi 1``
    Test  format : ``ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi``

    Labels in the train file are **1-indexed** — we convert to 0-indexed.
    For the test file (no labels), the class index is inferred from the
    folder name sorted alphabetically.
    """
    rows = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            clip_path = parts[0]
            if has_labels and len(parts) >= 2:
                label = int(parts[1]) - 1          # 1-indexed → 0-indexed
            else:
                # infer from folder name later
                label = -1
            # derive a human-readable class name from the folder
            class_name = clip_path.split("/")[0]
            rows.append({
                "clip_path":      clip_path,
                "label":          label,
                "label_original": class_name,
            })

    df = pd.DataFrame(rows)

    # If test file had no labels, assign them based on sorted class names
    if (df["label"] == -1).any():
        sorted_classes = sorted(df["label_original"].unique())
        class_to_idx = {c: i for i, c in enumerate(sorted_classes)}
        df["label"] = df["label_original"].map(class_to_idx)

    return df


# ==============================================================================
# Public API
# ==============================================================================

def get_dataset_config(dataset_name):
    """
    Return a normalised config dict for any supported dataset.

    The returned dict always contains:
        name, data_dir, train_df, val_df, test_df, test_data_dir,
        num_classes, ckpt_dir, ckpt_path, final_path, best_path

    ``train_df``, ``val_df``, ``test_df`` are pandas DataFrames
    with at least ``clip_path`` and ``label`` columns.
    """
    key = dataset_name.lower().replace("-", "").replace("_", "")
    # allow common aliases
    aliases = {"hmd51": "hmdb51", "hmdb": "hmdb51", "ucf": "ucf101", "haa": "haa500"}
    key = aliases.get(key, key)

    if key not in DATASETS:
        supported = ", ".join(DATASETS.keys())
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Supported: {supported}"
        )

    cfg = dict(DATASETS[key])  # shallow copy

    # ---- checkpoint paths ----
    cfg["ckpt_dir"]   = os.path.join(CKPT_BASE, key)
    cfg["ckpt_path"]  = os.path.join(cfg["ckpt_dir"], "checkpoint.pth")
    cfg["final_path"] = os.path.join(cfg["ckpt_dir"], f"rema_net_{key}_final.pth")
    cfg["best_path"]  = os.path.join(cfg["ckpt_dir"], f"rema_net_{key}_best.pth")

    # ---- load / build DataFrames ----
    if key == "ucf101":
        # Parse TXT files
        full_train_df = _parse_ucf101_txt(cfg["train_txt"], has_labels=True)
        test_df       = _parse_ucf101_txt(cfg["test_txt"],  has_labels=False)

        # Split 10 % of training for validation (stratified by label)
        train_df, val_df = train_test_split(
            full_train_df, test_size=0.1, random_state=42,
            stratify=full_train_df["label"],
        )
        train_df = train_df.reset_index(drop=True)
        val_df   = val_df.reset_index(drop=True)

        cfg["train_df"] = train_df
        cfg["val_df"]   = val_df
        cfg["test_df"]  = test_df
        # test videos live in a different root for UCF-101
        cfg.setdefault("test_data_dir", cfg["data_dir"])

    else:
        # Standard CSV datasets (HMDB51, HAA500)
        cfg["train_df"] = pd.read_csv(cfg["train_csv"])
        cfg["val_df"]   = pd.read_csv(cfg["val_csv"])
        cfg["test_df"]  = pd.read_csv(cfg["test_csv"])
        cfg["test_data_dir"] = cfg["data_dir"]

    print(f"[{cfg['name']}]  classes={cfg['num_classes']}  "
          f"train={len(cfg['train_df'])}  val={len(cfg['val_df'])}  "
          f"test={len(cfg['test_df'])}")

    return cfg
