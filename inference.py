#!/usr/bin/env python3
"""
REMA-Net — Real-Time Inference Demo
=====================================
Runs the trained model on a webcam or a video file with a professional
HUD overlay showing top-3 predictions, stream indicators, and semantic
captions.

Usage:
    # Webcam — full multimodal (RGB + Flow + BLIP)
    python inference.py --dataset hmdb51 --checkpoint /path/to/best_model.pth --video 0

    # Webcam — visual-only distilled (no BLIP, faster)
    python inference.py --dataset ucf101 --checkpoint /path/to/best_model.pth --video 0 --no_caption

    # Video file
    python inference.py --dataset haa500 --checkpoint /path/to/best_model.pth --video /path/to/video.mp4

Hotkeys:
    s — save a screenshot to ./real_time_vis/
    q — quit and print performance report
"""

import os
import sys
import time
import argparse
import collections

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

from config.default import DEVICE
from config.datasets import get_dataset_config
from models import REMANet
from models.streams import CaptionStream


# ==============================================================================
# Helpers
# ==============================================================================

def preprocess_frame(frame, transform):
    """Convert a BGR OpenCV frame to a transformed RGB tensor."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img   = Image.fromarray(rgb_frame)
    return transform(pil_img) if transform else transforms.ToTensor()(pil_img)


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="REMA-Net real-time inference")
    parser.add_argument("--dataset",    type=str, default="hmdb51",
                        choices=["hmdb51", "ucf101", "haa500"],
                        help="Which dataset was used for training (default: hmdb51)")
    parser.add_argument("--video",      type=str, default="0",      help="Video path or camera index")
    parser.add_argument("--checkpoint", type=str, required=True,    help="Path to .pth checkpoint")
    parser.add_argument("--num_segments", type=int, default=16)
    parser.add_argument("--stride",     type=int, default=4,        help="Sliding-window stride")
    parser.add_argument("--device",     type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_caption", action="store_true",        help="Disable BLIP (visual-only mode)")
    args = parser.parse_args()

    # ---- resolve dataset config ----
    cfg         = get_dataset_config(args.dataset)
    num_classes = cfg["num_classes"]
    # Use val_csv for class-name lookup (has label + label_original columns)
    annotation_file = cfg.get("val_csv", "")

    device = torch.device(args.device)
    print(f"Dataset : {args.dataset}  ({num_classes} classes)")
    print(f"Device  : {device}")

    # --- Model ---
    print("Loading REMANet …")
    model = REMANet(num_classes=num_classes, num_segments=args.num_segments, pretrained=False)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # --- Caption model ---
    if not args.no_caption:
        print("Loading CaptionStream (BLIP + BERT-tiny) …")
        caption_model = CaptionStream(device=device)
    else:
        print("Captioning disabled — visual-only mode.")
        caption_model = None

    # --- Transforms ---
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

    # --- Video capture ---
    src = int(args.video) if args.video.isdigit() else args.video
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Error opening video source: {src}")
        return

    # --- Class-name mapping ---
    class_names = {}
    if annotation_file and os.path.exists(annotation_file):
        try:
            df = pd.read_csv(annotation_file)
            for _, row in df.iterrows():
                class_names[int(row["label"])] = row["label_original"]
            print(f"Loaded {len(class_names)} class names from CSV.")
        except Exception as e:
            print(f"Error loading class names: {e}")

    # Fallback: build class names from dataset config's val_df (useful for UCF101)
    if not class_names and "val_df" in cfg:
        try:
            val_df = cfg["val_df"]
            for _, row in val_df.iterrows():
                idx = int(row["label"])
                if idx not in class_names and "label_original" in row:
                    class_names[idx] = row["label_original"]
            if class_names:
                print(f"Loaded {len(class_names)} class names from dataset config.")
        except Exception:
            pass

    # --- Buffers ---
    raw_buf  = collections.deque(maxlen=args.num_segments)
    rgb_buf  = collections.deque(maxlen=args.num_segments)
    flow_buf = collections.deque(maxlen=args.num_segments)
    prev_gray = None

    start_time   = time.time()
    frame_count  = 0
    infer_count  = 0
    total_infer  = 0.0
    latencies    = []
    detailed_log = []
    top_3_list   = []
    caption      = ""
    infer_duration = 0.0

    print("Starting inference. Press 'q' to quit, 's' for screenshot.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # --- RGB ---
            rgb_tensor = preprocess_frame(frame, rgb_transform)

            # --- Flow ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is None:
                prev_gray = gray
                flow_tensor = torch.zeros(2, 224, 224)
            else:
                g_r = cv2.resize(gray, (224, 224))
                p_r = cv2.resize(prev_gray, (224, 224))
                flow = cv2.calcOpticalFlowFarneback(p_r, g_r, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                fx = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                fy = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                ftx = transforms.ToTensor()(Image.fromarray(fx))
                fty = transforms.ToTensor()(Image.fromarray(fy))
                flow_tensor = torch.cat([ftx, fty], dim=0)

                for t in flow_transform.transforms:
                    if isinstance(t, transforms.Normalize):
                        flow_tensor = t(flow_tensor)
                        break
            prev_gray = gray

            raw_buf.append(frame)
            rgb_buf.append(rgb_tensor)
            flow_buf.append(flow_tensor)

            # --- Inference ---
            if len(rgb_buf) == args.num_segments and frame_count % args.stride == 0:
                t0 = time.time()

                rgb_in  = torch.stack(list(rgb_buf)).unsqueeze(0).to(device)
                flow_in = torch.stack(list(flow_buf)).unsqueeze(0).to(device)

                if not args.no_caption:
                    mid = raw_buf[args.num_segments // 2]
                    mid_pil = Image.fromarray(cv2.cvtColor(mid, cv2.COLOR_BGR2RGB))
                    with torch.no_grad():
                        emb, caption = caption_model(mid_pil)
                    if emb.dim() == 2:
                        emb_in = emb.unsqueeze(1).expand(-1, args.num_segments, -1).to(device)
                    else:
                        emb_in = emb.view(1, 1, -1).expand(-1, args.num_segments, -1).to(device)
                    with torch.no_grad():
                        _, results_probs = model.predict(rgb_in, flow_in, emb_in)
                else:
                    dummy = torch.zeros(1, args.num_segments, 128).to(device)
                    with torch.no_grad():
                        sp, tp, _, _, _ = model(rgb_in, flow_in, dummy)
                        results_probs = (F.softmax(sp, 1) + F.softmax(tp, 1)) / 2.0
                    caption = "DISABLED"

                top_prob, top_idx = torch.topk(results_probs, 3)
                top_3_list = [
                    (class_names.get(top_idx[0, i].item(), str(top_idx[0, i].item())),
                     top_prob[0, i].item())
                    for i in range(3)
                ]

                infer_duration = time.time() - t0
                total_infer   += infer_duration
                latencies.append(infer_duration)
                infer_count   += 1

                detailed_log.append({
                    "frame": frame_count,
                    "action": top_3_list[0][0],
                    "confidence": top_3_list[0][1],
                    "latency_ms": infer_duration * 1000,
                    "caption": caption,
                })

            # --- HUD visualisation ---
            disp = frame.copy()
            h, w, _ = disp.shape

            overlay = disp.copy()
            cv2.rectangle(overlay, (0, 0), (w, 140), (40, 40, 40), -1)
            cv2.addWeighted(overlay, 0.7, disp, 0.3, 0, disp)

            if infer_count > 0:
                for i, (name, prob) in enumerate(top_3_list):
                    if i == 0:
                        col, sc, th, yp = (0, 255, 0), 1.2, 2, 45
                        lbl = f"RANK 1: {name.upper()} ({prob:.1%})"
                    else:
                        col, sc, th = (200, 200, 200), 0.6, 1
                        yp = 75 + (i - 1) * 25
                        lbl = f"Rank {i+1}: {name} ({prob:.1%})"
                    cv2.putText(disp, lbl, (20, yp),
                                cv2.FONT_HERSHEY_DUPLEX, sc, col, th, cv2.LINE_AA)

                cv2.putText(disp, f"Inference Latency: {infer_duration*1000:.1f}ms",
                            (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (150, 150, 150), 1, cv2.LINE_AA)
            else:
                cv2.putText(disp, "INITIALIZING STREAMS...", (20, 60),
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

            if not args.no_caption and infer_count > 0:
                cv2.rectangle(disp, (0, h - 60), (w, h), (20, 20, 20), -1)
                cv2.addWeighted(disp, 0.8, disp, 0.2, 0, disp)
                cv2.putText(disp, f"Semantic Context: {caption}", (20, h - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            streams = [("RGB", True), ("FLOW", True), ("CAP", not args.no_caption)]
            for i, (nm, active) in enumerate(streams):
                c = (0, 255, 0) if active else (50, 50, 50)
                cv2.circle(disp, (w - 180 + i * 60, 40), 10, c, -1, cv2.LINE_AA)
                cv2.putText(disp, nm, (w - 195 + i * 60, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow("REMA-Net Professional Inference", disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                save_dir = os.path.join(os.path.dirname(__file__), "real_time_vis")
                os.makedirs(save_dir, exist_ok=True)
                tag = "distilled" if args.no_caption else "hybrid"
                ts  = time.strftime("%Y%m%d-%H%M%S")
                fp  = os.path.join(save_dir, f"REMA_Net_{tag}_{ts}.png")
                cv2.imwrite(fp, disp)
                print(f"Screenshot saved: {fp}")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

        duration = time.time() - start_time
        mode = "Visual-Only (Distilled)" if args.no_caption else "Multimodal Hybrid"

        print("\n" + "=" * 50)
        print("PERFORMANCE SUMMARY REPORT")
        print("=" * 50)
        print(f"Mode:             {mode}")
        print(f"Frames processed: {frame_count}")
        print(f"Inferences run:   {infer_count}")
        print(f"Total time:       {duration:.2f}s")

        if infer_count > 0:
            avg_lat = (total_infer / infer_count) * 1000
            avg_fps = frame_count / duration
            p50 = np.percentile(latencies, 50) * 1000
            p95 = np.percentile(latencies, 95) * 1000
            std = np.std(latencies) * 1000
            mem = torch.cuda.max_memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0

            print(f"Avg latency:      {avg_lat:.2f} ms")
            print(f"Std deviation:    {std:.2f} ms")
            print(f"Median (P50):     {p50:.2f} ms")
            print(f"P95 latency:      {p95:.2f} ms")
            print(f"Throughput:       {avg_fps:.2f} FPS")
            if torch.cuda.is_available():
                print(f"Peak GPU mem:     {mem:.2f} MB")

            # Save summary CSV
            out_dir = os.path.dirname(__file__)
            tag = "Visual-Only-Distilled" if args.no_caption else "Multimodal-Hybrid"
            pd.DataFrame([{
                "Mode": mode, "Total_Frames": frame_count,
                "Inference_Count": infer_count, "Total_Time_Sec": duration,
                "Avg_Latency_ms": avg_lat, "Std_Dev_ms": std,
                "Median_Latency_ms": p50, "P95_Latency_ms": p95,
                "Throughput_FPS": avg_fps, "Max_GPU_Mem_MB": mem,
            }]).to_csv(os.path.join(out_dir, f"{tag}.csv"), index=False)

            pd.DataFrame(detailed_log).to_csv(
                os.path.join(out_dir, f"{tag}_LOGS.csv"), index=False
            )


if __name__ == "__main__":
    main()
