#!/usr/bin/env python3
"""
Model Shape Verification
=========================
Runs a dummy forward pass through MATResNet and prints every
tensor shape to confirm the architecture is wired correctly.

Usage:
    cd rema_net
    python -m tools.verify_model_shapes
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from models import MATResNet


def verify():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_classes  = 51
    num_segments = 16
    batch_size   = 2

    print("Initialising MATResNet …")
    model = MATResNet(num_classes=num_classes, num_segments=num_segments, pretrained=False)
    model.to(device)
    model.eval()

    print("Creating dummy inputs …")
    rgb  = torch.randn(batch_size, num_segments, 3, 224, 224).to(device)
    flow = torch.randn(batch_size, num_segments, 2, 224, 224).to(device)
    emb  = torch.randn(batch_size, num_segments, 128).to(device)

    print(f"  RGB  : {rgb.shape}")
    print(f"  Flow : {flow.shape}")
    print(f"  Emb  : {emb.shape}")

    print("Running forward pass …")
    try:
        outputs = model(rgb, flow, emb)

        if len(outputs) == 5:
            sp, tp, cp, fp, gate = outputs
            print("\n✓ Forward pass successful!")
            print(f"  Spatial  output : {sp.shape}")
            print(f"  Temporal output : {tp.shape}")
            print(f"  Caption  output : {cp.shape}")
            print(f"  Fused    output : {fp.shape}")
            print(f"  Gate     weight : {gate.shape}")

            expected = (batch_size, num_classes)
            if sp.shape == expected and fp.shape == expected:
                print(f"\n✓ All shapes match expected {expected}")
            else:
                print(f"\n✗ Shape mismatch! Expected {expected}")
        else:
            print(f"✗ Unexpected number of outputs: {len(outputs)}  (expected 5)")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    verify()
