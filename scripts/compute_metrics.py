#!/usr/bin/env python3
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import sys
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def load_exr(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Failed to load: {path}")
        return None
    return cv2.cvtColor(img[:,:,:3], cv2.COLOR_BGR2RGB).astype(np.float32)

def compute_psnr(gt, pred):
    mse = np.mean((gt - pred) ** 2)
    return 20 * np.log10(1.0 / np.sqrt(mse + 1e-8))

gt_path, pred_path = sys.argv[1], sys.argv[2]

gt = load_exr(gt_path)
pred = load_exr(pred_path)

if gt is None or pred is None:
    print("Failed to load images")
    sys.exit(1)

# Normalize [0,1]
gt_norm = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)
pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

ssim_val = ssim(gt_norm, pred_norm, multichannel=True, data_range=1.0)
psnr_val = compute_psnr(gt_norm, pred_norm)

print(f"SSIM: {ssim_val:.4f} PSNR: {psnr_val:.2f}dB")
print(f"Ranges - GT: [{gt.min():.3f},{gt.max():.3f}] PRED: [{pred.min():.3f},{pred.max():.3f}]")
