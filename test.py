#!/usr/bin/env python3
"""
REMA-Net — Test / Validation Script
=====================================
Evaluate a trained checkpoint on a test set.  Reports per-stream accuracy
(Spatial, Temporal, Caption, Gated, Ensemble) and saves confusion-matrix
plots to the current directory.

Usage:
    python test.py --dataset hmdb51  --checkpoint /path/to/checkpoint.pth
    python test.py --dataset ucf101  --checkpoint /path/to/checkpoint.pth
    python test.py --dataset haa500  --checkpoint /path/to/checkpoint.pth
"""

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, accuracy_score

from config.default import DEVICE
from config.datasets import get_dataset_config
from models import REMANet
from data import ActionRecognitionDataset, custom_collate_fn


# ==============================================================================
# Helpers
# ==============================================================================

def plot_confusion_matrix(cm, class_names, title, save_path):
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved → {save_path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate REMA-Net")
    parser.add_argument("--dataset",         type=str, default="hmdb51",
                        choices=["hmdb51", "ucf101", "haa500"],
                        help="Which dataset to evaluate on (default: hmdb51)")
    parser.add_argument("--checkpoint",      type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--batch_size",      type=int, default=16)
    parser.add_argument("--device",          type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # ---- resolve dataset config ----
    cfg         = get_dataset_config(args.dataset)
    data_dir    = cfg["data_dir"]
    test_df     = cfg["test_df"]
    num_classes = cfg["num_classes"]

    device = torch.device(args.device)
    print(f"Dataset : {args.dataset}  ({num_classes} classes)")
    print(f"Data dir: {data_dir}")
    print(f"Device  : {device}")

    # --- transforms (same as training) ---
    rgb_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    flow_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5]),
    ])

    # --- dataset ---
    print("Loading dataset …")
    dataset = ActionRecognitionDataset(
        data_dir=data_dir,
        annotation_file=test_df,
        num_segments=16,
        transform=rgb_transform,
        flow_transform=flow_transform,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=custom_collate_fn)

    # --- model ---
    print("Loading model …")
    model = REMANet(num_classes=num_classes, num_segments=16, pretrained=False)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # --- inference ---
    all_labels = []
    preds = {k: [] for k in ["spatial", "temporal", "caption", "gated", "ensemble"]}

    print("Running evaluation …")
    with torch.no_grad():
        for i, (rgb, flow, emb, labels) in enumerate(loader):
            rgb, flow, emb, labels = (
                rgb.to(device), flow.to(device), emb.to(device), labels.to(device),
            )

            outputs = model(rgb, flow, emb)
            if len(outputs) == 5:
                sp, tp, cp, fp, _ = outputs
            else:
                sp, tp, cp = outputs
                fp = sp  # fallback

            sp_p = F.softmax(sp, dim=1)
            tp_p = F.softmax(tp, dim=1)
            cp_p = F.softmax(cp, dim=1)
            fp_p = F.softmax(fp, dim=1)
            en_p = (sp_p + tp_p + cp_p + fp_p) / 4.0

            preds["spatial"].extend(sp_p.argmax(1).cpu().numpy())
            preds["temporal"].extend(tp_p.argmax(1).cpu().numpy())
            preds["caption"].extend(cp_p.argmax(1).cpu().numpy())
            preds["gated"].extend(fp_p.argmax(1).cpu().numpy())
            preds["ensemble"].extend(en_p.argmax(1).cpu().numpy())

            all_labels.extend(labels.cpu().numpy())

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1} batches …", end="\r")

    # --- metrics ---
    print("\n\nResults")
    print("=" * 50)

    if not all_labels:
        print("[ERROR] No data processed. Check --data_dir and --annotation_file.")
        return

    gt = np.array(all_labels)
    class_names = [str(i) for i in range(num_classes)]

    for stream, pred_list in preds.items():
        if not pred_list:
            continue
        pred_arr = np.array(pred_list)
        acc = accuracy_score(gt, pred_arr)
        cm  = confusion_matrix(gt, pred_arr)

        print(f"  {stream.capitalize():10s}  Accuracy: {acc*100:.2f}%")
        plot_confusion_matrix(
            cm, class_names,
            f"{stream.capitalize()} Stream  (Acc {acc:.2f})",
            f"cm_{stream}.png",
        )

    # --- bar chart ---
    accs = {s: accuracy_score(gt, np.array(preds[s])) for s in preds if preds[s]}
    colours = {"spatial": "red", "temporal": "green", "caption": "blue",
               "gated": "orange", "ensemble": "purple"}

    plt.figure(figsize=(12, 6))
    bars = plt.bar(
        [s.capitalize() for s in accs],
        list(accs.values()),
        color=[colours.get(s, "gray") for s in accs],
    )
    plt.title("Accuracy by Stream")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)
    for bar, v in zip(bars, accs.values()):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                 f"{v*100:.1f}%", ha="center")
    plt.tight_layout()
    plt.savefig("stream_comparison.png")
    print(f"\n  Bar chart → stream_comparison.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
