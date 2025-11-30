#!/usr/bin/env python3
"""
Compute LPIPS, SSIM, and PSNR between two image folders.

Usage:
  python eval/compute_image_metrics.py \
    --pred_dir /path/to/pred_images \
    --gt_dir /path/to/gt_images \
    --resize pred_to_gt \
    --out_csv eval_results/image_metrics.csv

Notes:
- Images are matched by filename stem (basename without extension). If no stem matches
  are found, falls back to lexicographically sorted pairing.
- LPIPS is computed with the 'alex' backbone in the lpips package.
- SSIM is implemented here with a Gaussian window (11x11, sigma=1.5) on RGB and averaged.
- PSNR is computed per RGB image then averaged.
"""
import argparse
import csv
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import lpips
from PIL import Image


def list_images(dir_path: str) -> List[str]:
	exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
	files = []
	for name in os.listdir(dir_path):
		ext = os.path.splitext(name.lower())[1]
		if ext in exts:
			files.append(os.path.join(dir_path, name))
	files.sort()
	return files


def build_stem_map(paths: List[str]) -> Dict[str, str]:
	stem_map = {}
	for p in paths:
		base = os.path.basename(p)
		stem = os.path.splitext(base)[0]
		stem_map[stem] = p
	return stem_map


def load_image_rgb(path: str) -> np.ndarray:
	# Returns float32 in [0, 1], shape (H, W, 3)
	img = Image.open(path).convert("RGB")
	arr = np.asarray(img, dtype=np.uint8)
	return (arr.astype(np.float32) / 255.0)


def resize_image(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
	# size: (W, H)
	h, w = img.shape[:2]
	if (w, h) == size:
		return img
	return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def compute_psnr(img_pred: np.ndarray, img_gt: np.ndarray) -> float:
	# img_* in [0, 1], float32
	mse = np.mean((img_pred - img_gt) ** 2, dtype=np.float64)
	if mse <= 1e-12:
		return 100.0
	max_i = 1.0
	psnr = 10.0 * np.log10((max_i * max_i) / mse)
	return float(psnr)


def _gaussian_window(ksize: int = 11, sigma: float = 1.5) -> np.ndarray:
	# 2D separable Gaussian window normalized to sum=1
	g1d = cv2.getGaussianKernel(ksize, sigma)
	window = g1d @ g1d.T
	window = window / np.sum(window)
	return window.astype(np.float32)


def _ssim_single_channel(x: np.ndarray, y: np.ndarray, window: np.ndarray, c1: float, c2: float) -> float:
	# x, y: HxW in [0, 1], float32
	# Convolve with gaussian window via filter2D
	mu_x = cv2.filter2D(x, -1, window, borderType=cv2.BORDER_REFLECT)
	mu_y = cv2.filter2D(y, -1, window, borderType=cv2.BORDER_REFLECT)

	mu_x2 = mu_x * mu_x
	mu_y2 = mu_y * mu_y
	mu_xy = mu_x * mu_y

	sigma_x2 = cv2.filter2D(x * x, -1, window, borderType=cv2.BORDER_REFLECT) - mu_x2
	sigma_y2 = cv2.filter2D(y * y, -1, window, borderType=cv2.BORDER_REFLECT) - mu_y2
	sigma_xy = cv2.filter2D(x * y, -1, window, borderType=cv2.BORDER_REFLECT) - mu_xy

	num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
	den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
	ssim_map = num / (den + 1e-12)
	return float(ssim_map.mean())


def compute_ssim(img_pred: np.ndarray, img_gt: np.ndarray) -> float:
	# SSIM on RGB channels then average
	# img_* in [0, 1], float32, shape (H, W, 3)
	window = _gaussian_window(11, 1.5)
	# Constants per original SSIM paper with L=1
	L = 1.0
	k1, k2 = 0.01, 0.03
	c1 = (k1 * L) ** 2
	c2 = (k2 * L) ** 2

	ssims = []
	for c in range(3):
		ssims.append(_ssim_single_channel(img_pred[..., c], img_gt[..., c], window, c1, c2))
	return float(np.mean(ssims))


def compute_lpips(loss_fn: lpips.LPIPS, img_pred: np.ndarray, img_gt: np.ndarray, device: torch.device) -> float:
	# img_* in [0,1], float32, shape (H, W, 3). Convert to [-1,1] torch BCHW
	t1 = torch.from_numpy(img_pred).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
	t2 = torch.from_numpy(img_gt).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
	t1 = t1 * 2.0 - 1.0
	t2 = t2 * 2.0 - 1.0
	with torch.no_grad():
		val = loss_fn(t1, t2).item()
	return float(val)


def pair_images(pred_paths: List[str], gt_paths: List[str]) -> List[Tuple[str, str]]:
	pred_map = build_stem_map(pred_paths)
	gt_map = build_stem_map(gt_paths)
	common = sorted(set(pred_map.keys()) & set(gt_map.keys()))
	pairs: List[Tuple[str, str]] = []
	if common:
		for stem in common:
			pairs.append((pred_map[stem], gt_map[stem]))
		return pairs
	# fallback to sorted order if no shared stems
	n = min(len(pred_paths), len(gt_paths))
	return [(pred_paths[i], gt_paths[i]) for i in range(n)]


def main():
	parser = argparse.ArgumentParser(description="Compute LPIPS, SSIM, and PSNR between two image folders.")
	parser.add_argument("--pred_dir", type=str, required=True, help="Directory of predicted/output images.")
	parser.add_argument("--gt_dir", type=str, required=True, help="Directory of ground-truth/reference images.")
	parser.add_argument("--resize", type=str, choices=["none", "pred_to_gt", "gt_to_pred"], default="pred_to_gt",
	                    help="If shapes differ: resize which side to the other.")
	parser.add_argument("--out_csv", type=str, default="eval_results/image_metrics.csv", help="Path to CSV results.")
	parser.add_argument("--device", type=str, default="cuda", help="Torch device for LPIPS (cuda or cpu).")
	args = parser.parse_args()

	os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

	pred_paths = list_images(args.pred_dir)
	gt_paths = list_images(args.gt_dir)
	if not pred_paths:
		raise FileNotFoundError(f"No images found in pred_dir: {args.pred_dir}")
	if not gt_paths:
		raise FileNotFoundError(f"No images found in gt_dir: {args.gt_dir}")

	pairs = pair_images(pred_paths, gt_paths)
	if not pairs:
		raise RuntimeError("No image pairs found to compare.")

	device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
	loss_fn = lpips.LPIPS(net="alex").to(device)
	loss_fn.eval()

	records = []
	lpips_vals, ssim_vals, psnr_vals = [], [], []

	for pred_path, gt_path in pairs:
		img_pred = load_image_rgb(pred_path)
		img_gt = load_image_rgb(gt_path)

		hp, wp = img_pred.shape[:2]
		hg, wg = img_gt.shape[:2]
		if (hp != hg) or (wp != wg):
			if args.resize == "pred_to_gt":
				img_pred = resize_image(img_pred, (wg, hg))
			elif args.resize == "gt_to_pred":
				img_gt = resize_image(img_gt, (wp, hp))
			else:
				raise ValueError(f"Image shapes differ {img_pred.shape} vs {img_gt.shape} and resize=none.")

		lp = compute_lpips(loss_fn, img_pred, img_gt, device)
		ss = compute_ssim(img_pred, img_gt)
		pn = compute_psnr(img_pred, img_gt)

		records.append({
			"pred": os.path.basename(pred_path),
			"gt": os.path.basename(gt_path),
			"LPIPS": lp,
			"SSIM": ss,
			"PSNR": pn,
		})
		lpips_vals.append(lp)
		ssim_vals.append(ss)
		psnr_vals.append(pn)

	# Write CSV
	with open(args.out_csv, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=["pred", "gt", "LPIPS", "SSIM", "PSNR"])
		writer.writeheader()
		for r in records:
			writer.writerow(r)
		# Add mean row
		writer.writerow({
			"pred": "MEAN",
			"gt": "",
			"LPIPS": float(np.mean(lpips_vals)) if lpips_vals else float("nan"),
			"SSIM": float(np.mean(ssim_vals)) if ssim_vals else float("nan"),
			"PSNR": float(np.mean(psnr_vals)) if psnr_vals else float("nan"),
		})

	print(f"Compared {len(records)} pairs.")
	print(f"Mean LPIPS: {float(np.mean(lpips_vals)):.6f}")
	print(f"Mean SSIM : {float(np.mean(ssim_vals)):.6f}")
	print(f"Mean PSNR : {float(np.mean(psnr_vals)):.6f} dB")
	print(f"Wrote CSV to: {args.out_csv}")


if __name__ == "__main__":
	main()


