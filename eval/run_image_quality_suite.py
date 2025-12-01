#!/usr/bin/env python3
"""
Run image quality metrics (LPIPS, SSIM, PSNR) for multiple model update types
on a single sequence. This script:
  1) Extracts/collects the GT frames into output_root/gt_rgb
  2) Runs the model three times (cut3r, ttt3r, meta_beta) to save
     predictions into output_root/{model}/pred_rgb
  3) Computes LPIPS/SSIM/PSNR and writes one CSV per model in results_dir

Example:
  python eval/run_image_quality_suite.py \
    --seq_path examples/taylor.mp4 \
    --model_path src/cut3r_512_dpt_4_64.pth \
    --output_root /workspace/TTT3R/tmp/taylor_iq \
    --results_dir /workspace/TTT3R/eval_results \
    --size 512 --frame_interval 1 --device cuda
"""
import os
import sys
import shutil
import argparse
from typing import Dict, List

import numpy as np
import torch
import imageio.v2 as iio
from PIL import Image

# Ensure project root is on path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from add_ckpt_path import add_path_to_dust3r


def save_image_png(path: str, arr_uint8_hw3: np.ndarray):
	# Ensure directory exists and save as PNG
	os.makedirs(os.path.dirname(path), exist_ok=True)
	img = Image.fromarray(arr_uint8_hw3.astype(np.uint8), mode="RGB")
	img.save(path, format="PNG")


def copy_or_convert_to_png(src_path: str, dst_path: str):
	# Copy as PNG (re-encode) to have uniform extension and naming
	img = Image.open(src_path).convert("RGB")
	os.makedirs(os.path.dirname(dst_path), exist_ok=True)
	img.save(dst_path, format="PNG")


def standardize_index_name(idx: int) -> str:
	return f"{idx:06d}.png"


def build_views_for_sequence(seq_path: str, frame_interval: int, size: int, reset_interval: int):
	# Use demo utilities to parse sequence and build views
	from demo import parse_seq_path, prepare_input
	img_paths, tmpdirname = parse_seq_path(seq_path, frame_interval)
	if not img_paths:
		raise RuntimeError(f"No frames found in {seq_path}")
	img_mask = [True] * len(img_paths)
	views = prepare_input(
		img_paths=img_paths,
		img_mask=img_mask,
		size=size,
		revisit=1,
		update=True,
		reset_interval=reset_interval,
	)
	return img_paths, tmpdirname, views


def extract_gt_frames(img_paths: List[str], gt_dir: str):
	# Copy/convert input frames to a standardized gt_dir with 000000.png, ...
	if os.path.exists(gt_dir):
		shutil.rmtree(gt_dir)
	os.makedirs(gt_dir, exist_ok=True)
	for idx, p in enumerate(img_paths):
		dst = os.path.join(gt_dir, standardize_index_name(idx))
		copy_or_convert_to_png(p, dst)
	return gt_dir


def run_model_and_save_pred_rgb(model_path: str, model_update_type: str, device: str, views, pred_dir: str) -> int:
	add_path_to_dust3r(model_path)
	from src.dust3r.model import ARCroco3DStereo
	from src.dust3r.inference import inference_recurrent_lighter

	if device == "cuda" and not torch.cuda.is_available():
		print("CUDA not available. Falling back to CPU.")
		device = "cpu"

	model = ARCroco3DStereo.from_pretrained(model_path).to(device)
	model.config.model_update_type = model_update_type
	model.eval()

	(outputs, _state_args) = inference_recurrent_lighter(views, model, device)

	if os.path.exists(pred_dir):
		shutil.rmtree(pred_dir)
	os.makedirs(pred_dir, exist_ok=True)

	num_saved = 0
	for idx, pred in enumerate(outputs["pred"]):
		if "rgb" not in pred:
			continue
		rgb = pred["rgb"].detach().cpu().numpy()
		if rgb.ndim == 4:
			rgb = rgb[0]
		rgb = (0.5 * (rgb + 1.0))
		rgb = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
		save_image_png(os.path.join(pred_dir, standardize_index_name(idx)), rgb)
		num_saved += 1
	return num_saved


def compute_image_metrics_to_csv(pred_dir: str, gt_dir: str, device: str, out_csv: str, resize: str = "pred_to_gt"):
	# Reuse the metric implementations from eval/compute_image_metrics.py
	from eval.compute_image_metrics import (
		list_images,
		pair_images,
		load_image_rgb,
		resize_image,
		compute_psnr,
		compute_ssim,
		compute_lpips,
	)
	import csv
	import lpips
	import numpy as np

	os.makedirs(os.path.dirname(out_csv), exist_ok=True)

	pred_paths = list_images(pred_dir)
	gt_paths = list_images(gt_dir)
	pairs = pair_images(pred_paths, gt_paths)
	if not pairs:
		raise RuntimeError(f"No image pairs found in {pred_dir} vs {gt_dir}")

	torch_device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
	loss_fn = lpips.LPIPS(net="alex").to(torch_device)
	loss_fn.eval()

	records = []
	lpips_vals, ssim_vals, psnr_vals = [], [], []
	for pred_path, gt_path in pairs:
		img_pred = load_image_rgb(pred_path)
		img_gt = load_image_rgb(gt_path)
		hp, wp = img_pred.shape[:2]
		hg, wg = img_gt.shape[:2]
		if (hp != hg) or (wp != wg):
			if resize == "pred_to_gt":
				img_pred = resize_image(img_pred, (wg, hg))
			elif resize == "gt_to_pred":
				img_gt = resize_image(img_gt, (wp, hp))
			else:
				raise ValueError(f"Image shapes differ {img_pred.shape} vs {img_gt.shape} and resize=none.")
		lp = compute_lpips(loss_fn, img_pred, img_gt, torch_device)
		ss = compute_ssim(img_pred, img_gt)
		pn = compute_psnr(img_pred, img_gt)
		records.append({"pred": os.path.basename(pred_path), "gt": os.path.basename(gt_path), "LPIPS": lp, "SSIM": ss, "PSNR": pn})
		lpips_vals.append(lp)
		ssim_vals.append(ss)
		psnr_vals.append(pn)

	with open(out_csv, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=["pred", "gt", "LPIPS", "SSIM", "PSNR"])
		writer.writeheader()
		for r in records:
			writer.writerow(r)
		writer.writerow({
			"pred": "MEAN",
			"gt": "",
			"LPIPS": float(np.mean(lpips_vals)) if lpips_vals else float("nan"),
			"SSIM": float(np.mean(ssim_vals)) if ssim_vals else float("nan"),
			"PSNR": float(np.mean(psnr_vals)) if psnr_vals else float("nan"),
		})


def parse_args():
	parser = argparse.ArgumentParser("Run LPIPS/SSIM/PSNR for cut3r/ttt3r/meta_beta")
	parser.add_argument("--seq_path", type=str, required=True, help="Path to image folder or video.")
	parser.add_argument("--output_root", type=str, required=True, help="Root dir to store gt and predictions.")
	parser.add_argument("--results_dir", type=str, default="/workspace/TTT3R/eval_results", help="Where to write CSVs.")
	parser.add_argument("--model_path", type=str, default="src/cut3r_512_dpt_4_64.pth", help="Default checkpoint path for all modes.")
	parser.add_argument("--cut3r_path", type=str, default=None, help="Optional override checkpoint for cut3r.")
	parser.add_argument("--ttt3r_path", type=str, default=None, help="Optional override checkpoint for ttt3r.")
	parser.add_argument("--meta_beta_path", type=str, default=None, help="Optional override checkpoint for meta_beta.")
	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument("--size", type=int, default=512)
	parser.add_argument("--frame_interval", type=int, default=1)
	parser.add_argument("--reset_interval", type=int, default=1000000)
	return parser.parse_args()


def main():
	args = parse_args()

	# Build views and extract GT frames
	img_paths, tmpdirname, views = build_views_for_sequence(
		seq_path=args.seq_path,
		frame_interval=args.frame_interval,
		size=args.size,
		reset_interval=args.reset_interval,
	)
	gt_dir = os.path.join(args.output_root, "gt_rgb")
	extract_gt_frames(img_paths, gt_dir)
	if tmpdirname is not None and os.path.isdir(tmpdirname):
		shutil.rmtree(tmpdirname)

	# Decide checkpoints per mode
	ckpt_for: Dict[str, str] = {
		"cut3r": args.cut3r_path or args.model_path,
		"ttt3r": args.ttt3r_path or args.model_path,
		"meta_beta": args.meta_beta_path or args.model_path,
	}

	# Run each update type and save predictions
	model_types = ["cut3r", "ttt3r", "meta_beta"]
	for m in model_types:
		print(f"Running model_update_type={m}")
		pred_dir = os.path.join(args.output_root, m, "pred_rgb")
		num_saved = run_model_and_save_pred_rgb(
			model_path=ckpt_for[m],
			model_update_type=m,
			device=args.device,
			views=views,
			pred_dir=pred_dir,
		)
		print(f"Saved {num_saved} frames for {m} at {pred_dir}")

	# Compute metrics and write CSVs
	seq_name = os.path.splitext(os.path.basename(args.seq_path.rstrip(os.sep)))[0]
	os.makedirs(args.results_dir, exist_ok=True)
	for m in model_types:
		pred_dir = os.path.join(args.output_root, m, "pred_rgb")
		out_csv = os.path.join(args.results_dir, f"{seq_name}_metrics_pred_rgb_{m}.csv")
		print(f"Computing metrics for {m}: pred_dir={pred_dir}, gt_dir={gt_dir}")
		compute_image_metrics_to_csv(pred_dir=pred_dir, gt_dir=gt_dir, device=args.device, out_csv=out_csv, resize="pred_to_gt")
		print(f"Wrote CSV: {out_csv}")


if __name__ == "__main__":
	main()


