#!/usr/bin/env python3
"""
REMA-Net — Training Script
===========================
Train the multi-stream REMANet model with cross-modal knowledge
distillation and K-Fold cross-validation.

Usage:
    python train.py --dataset hmdb51
    python train.py --dataset ucf101   --epochs 50 --batch_size 16
    python train.py --dataset haa500   --epochs 30

All defaults are loaded from  config/default.py  and can be overridden
via command-line arguments.  Dataset paths are resolved automatically
from  config/datasets.py .
"""

import os
import sys
import copy
import time
import gc
import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import KFold

import wandb

# --------------- project imports ---------------
from config.default import (
    DEVICE,
    LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE, K_FOLDS,
    NUM_SEGMENTS, NUM_HEADS, NUM_WORKERS,
    DISTILL_ALPHA, DISTILL_TEMPERATURE,
    WANDB_PROJECT, WANDB_API_KEY,
)
from config.datasets import get_dataset_config
from models import REMANet
from data import ActionRecognitionDataset, custom_collate_fn, get_transforms


# ==============================================================================
# Weights & Biases
# ==============================================================================

def setup_wandb(dataset_name="hmdb51"):
    wandb.login(key=WANDB_API_KEY)
    wandb.init(
        project=WANDB_PROJECT,
        name=f"REMANet-{dataset_name}",
        config={
            "model":       "REMANet",
            "dataset":     dataset_name,
            "lr":          LEARNING_RATE,
            "alpha":       DISTILL_ALPHA,
            "temperature": DISTILL_TEMPERATURE,
        },
    )


# ==============================================================================
# Training Loop
# ==============================================================================

def train_model(model, data_loaders, num_epochs, start_epoch=0,
                checkpoint_path=None, ckpt_dir="checkpoints", best_path="best_model.pth"):
    """
    Train REMANet with cross-modal knowledge distillation.

    The caption stream acts as the *teacher* — spatial and temporal
    streams are the *students* that learn to mimic its outputs.
    """

    # --- Optional GFLOPs count ---
    try:
        from thop import profile as thop_profile
        rgb_s, flow_s, emb_s, _ = next(iter(data_loaders["val"]))
        macs, params = thop_profile(
            model,
            inputs=(rgb_s[:1].to(DEVICE), flow_s[:1].to(DEVICE), emb_s[:1].to(DEVICE)),
            verbose=False,
        )
        gflops = macs * 2 / 1e9
        print(f"Model size: {params/1e6:.2f}M parameters, {gflops:.2f} GFLOPs")
        wandb.log({"gflops": gflops, "params_M": params / 1e6})
    except ImportError:
        print("'thop' not installed — skipping GFLOPs calculation.")

    # --- losses ---
    hard_loss_fn = nn.CrossEntropyLoss()
    soft_loss_fn = nn.KLDivLoss(reduction="batchmean")

    # --- optimiser + scheduler ---
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # --- resume from checkpoint ---
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Resuming training from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", start_epoch) + 1
        print(f"Resuming from epoch {start_epoch}")
    else:
        print("No checkpoint found — starting from scratch.")

    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    # ---- epoch loop ----
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch}/{num_epochs - 1}  {'─' * 40}")

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            total_loss    = 0.0
            total_correct = 0
            total_samples = 0
            total_time    = 0.0

            for batch_idx, batch in enumerate(data_loaders[phase]):
                if batch is None:
                    continue

                rgb_in, flow_in, emb_in, labels = batch
                rgb_in  = rgb_in.to(DEVICE)
                flow_in = flow_in.to(DEVICE)
                emb_in  = emb_in.to(DEVICE)
                labels  = labels.to(DEVICE)

                optimizer.zero_grad()
                batch_start = time.time()

                with torch.set_grad_enabled(phase == "train"):
                    sp_out, tm_out, cap_out, fused_out, gate = model(
                        rgb_in, flow_in, emb_in
                    )

                    # hard loss (each stream vs true labels)
                    hard_loss = (
                        hard_loss_fn(sp_out, labels)
                        + hard_loss_fn(tm_out, labels)
                        + hard_loss_fn(cap_out, labels)
                        + hard_loss_fn(fused_out, labels)
                    )

                    # soft loss (knowledge distillation from caption stream)
                    def kd_loss(student, teacher):
                        s = F.log_softmax(student / DISTILL_TEMPERATURE, dim=1)
                        t = F.softmax(teacher / DISTILL_TEMPERATURE, dim=1)
                        return soft_loss_fn(s, t) * (DISTILL_TEMPERATURE ** 2)

                    distill_loss = (
                        kd_loss(sp_out, cap_out)
                        + kd_loss(tm_out, cap_out)
                        + kd_loss(fused_out, cap_out)
                    )

                    loss = hard_loss + DISTILL_ALPHA * distill_loss

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        wandb.log({
                            "train_loss": loss.item(),
                            "gate_weight_mean": gate.mean().item(),
                        })

                bs = rgb_in.size(0)
                total_loss    += loss.item() * bs
                total_samples += bs
                total_time    += time.time() - batch_start

                avg_probs = (
                    F.softmax(sp_out, dim=1)
                    + F.softmax(tm_out, dim=1)
                    + F.softmax(cap_out, dim=1)
                    + F.softmax(fused_out, dim=1)
                ) / 4.0
                preds = avg_probs.argmax(dim=1)
                total_correct += (preds == labels).sum().item()

                if (batch_idx + 1) % 10 == 0:
                    r_loss = total_loss / total_samples
                    r_acc  = total_correct / total_samples
                    fps    = total_samples / max(total_time, 1e-6)
                    print(
                        f"  [{phase}] Batch {batch_idx+1}/{len(data_loaders[phase])} "
                        f"| Loss: {r_loss:.4f} | Acc: {r_acc:.4f} | FPS: {fps:.1f}"
                    )

            # --- epoch summary ---
            epoch_loss = total_loss / len(data_loaders[phase].dataset)
            epoch_acc  = total_correct / len(data_loaders[phase].dataset)
            epoch_fps  = total_samples / max(total_time, 1e-6)

            print(
                f"\n  [{phase}] Epoch {epoch} Summary | "
                f"Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | FPS: {epoch_fps:.1f}"
            )
            wandb.log({f"{phase}_loss": epoch_loss, f"{phase}_acc": epoch_acc, "epoch": epoch})

            if phase == "train":
                scheduler.step()
                lr = optimizer.param_groups[0]["lr"]
                print(f"  Learning rate: {lr:.6f}")
                wandb.log({"lr": lr})

            if phase == "val":
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    best_weights = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), best_path)
                    print(f"  ✓ New best model saved! Val Acc: {best_val_acc:.4f}")

                ckpt_name = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch+1}.pth")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_acc": best_val_acc,
                    },
                    ckpt_name,
                )
                print(f"  Checkpoint saved: {ckpt_name}")

        torch.cuda.empty_cache()
        gc.collect()

    print(f"\nTraining complete! Best val Acc: {best_val_acc:.4f}")
    model.load_state_dict(best_weights)
    return model


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train REMA-Net")
    parser.add_argument("--dataset",    type=str, default="hmdb51",
                        choices=["hmdb51", "ucf101", "haa500"],
                        help="Which dataset to train on (default: hmdb51)")
    parser.add_argument("--epochs",     type=int, default=NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--k_folds",    type=int, default=K_FOLDS,    help="K-Fold splits")
    args = parser.parse_args()

    # ---- resolve dataset config ----
    cfg = get_dataset_config(args.dataset)
    data_dir    = cfg["data_dir"]
    train_df    = cfg["train_df"]
    val_df      = cfg["val_df"]
    num_classes = cfg["num_classes"]
    ckpt_dir    = cfg["ckpt_dir"]
    best_path   = cfg["best_path"]
    final_path  = cfg["final_path"]
    ckpt_path   = cfg["ckpt_path"]

    print(f"Dataset : {args.dataset}  ({num_classes} classes)")
    print(f"Data dir: {data_dir}")
    print(f"Ckpt dir: {ckpt_dir}")

    setup_wandb(dataset_name=args.dataset)

    rgb_transform, flow_transform, train_transform = get_transforms()

    # Combine train + val for K-Fold cross-validation
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    print(f"Total samples for cross-validation: {len(combined_df)}")

    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)

    for fold_num, (train_idx, val_idx) in enumerate(kfold.split(combined_df)):
        print(f"\n{'=' * 20}  Fold {fold_num + 1} / {args.k_folds}  {'=' * 20}")

        fold_train = combined_df.iloc[train_idx].reset_index(drop=True)
        fold_val   = combined_df.iloc[val_idx].reset_index(drop=True)
        print(f"Train: {len(fold_train)} | Val: {len(fold_val)}")

        ds_kwargs = dict(
            data_dir=data_dir,
            num_segments=NUM_SEGMENTS,
            flow_transform=flow_transform,
        )

        train_orig = ActionRecognitionDataset(
            annotation_file=fold_train, transform=rgb_transform, **ds_kwargs
        )
        train_aug = ActionRecognitionDataset(
            annotation_file=fold_train, transform=train_transform, **ds_kwargs
        )
        train_ds = ConcatDataset([train_orig, train_aug])

        val_ds = ActionRecognitionDataset(
            annotation_file=fold_val, transform=rgb_transform, **ds_kwargs
        )

        loaders = {
            "train": DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=True,
                num_workers=NUM_WORKERS, collate_fn=custom_collate_fn,
            ),
            "val": DataLoader(
                val_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=NUM_WORKERS, collate_fn=custom_collate_fn,
            ),
        }

        model = REMANet(
            num_classes=num_classes,
            num_segments=NUM_SEGMENTS,
            pretrained=True,
            num_heads=NUM_HEADS,
        ).to(DEVICE)

        model = train_model(
            model=model,
            data_loaders=loaders,
            num_epochs=args.epochs,
            start_epoch=0,
            checkpoint_path=ckpt_path,
            ckpt_dir=ckpt_dir,
            best_path=best_path,
        )

        torch.save(model.state_dict(), final_path)
        print(f"Final model for fold {fold_num + 1} saved to: {final_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
