#!/usr/bin/env python3
"""
Save predicted RGB frames from the model to output_dir/pred_rgb for a given sequence.

This mirrors the demo's input preparation but skips visualization and only saves RGB.

Usage:
  python eval/save_pred_rgb.py \
    --model_path src/cut3r_512_dpt_4_64.pth \
    --seq_path examples/taylor.mp4 \
    --output_dir tmp/taylor \
    --size 512 \
    --model_update_type ttt3r \
    --frame_interval 1 \
    --reset_interval 100
"""
import os
import argparse
import shutil
import numpy as np
import imageio.v2 as iio
import torch
import sys

# Ensure project root is on the path so we can import add_ckpt_path and demo
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from add_ckpt_path import add_path_to_dust3r


def parse_args():
	parser = argparse.ArgumentParser("Save predicted RGBs")
	parser.add_argument("--model_path", type=str, default="src/cut3r_512_dpt_4_64.pth")
	parser.add_argument("--seq_path", type=str, required=True)
	parser.add_argument("--output_dir", type=str, required=True)
	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument("--size", type=int, default=512)
	parser.add_argument("--model_update_type", type=str, default="ttt3r", help="cut3r | ttt3r | meta_beta")
	parser.add_argument("--frame_interval", type=int, default=1)
	parser.add_argument("--reset_interval", type=int, default=1000000)
	return parser.parse_args()


def main():
	args = parse_args()
	device = args.device
	if device == "cuda" and not torch.cuda.is_available():
		print("CUDA not available. Switching to CPU.")
		device = "cpu"

	add_path_to_dust3r(args.model_path)
	# Import after ckpt path registration
	from src.dust3r.model import ARCroco3DStereo
	from src.dust3r.inference import inference_recurrent_lighter
	# Reuse demo's input preparation utilities
	from demo import parse_seq_path, prepare_input

	# Prepare input views
	img_paths, tmpdirname = parse_seq_path(args.seq_path, args.frame_interval)
	if not img_paths:
		raise RuntimeError(f"No frames found in {args.seq_path}")
	img_mask = [True] * len(img_paths)
	views = prepare_input(
		img_paths=img_paths,
		img_mask=img_mask,
		size=args.size,
		revisit=1,
		update=True,
		reset_interval=args.reset_interval,
	)
	if tmpdirname is not None:
		shutil.rmtree(tmpdirname)

	# Load model
	model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
	model.config.model_update_type = args.model_update_type
	model.eval()

	# Run inference
	(outputs, _state_args) = inference_recurrent_lighter(views, model, device)

	# Save predicted RGBs if available
	save_dir = os.path.join(args.output_dir, "pred_rgb")
	if os.path.exists(save_dir):
		shutil.rmtree(save_dir)
	os.makedirs(save_dir, exist_ok=True)

	num_saved = 0
	for idx, pred in enumerate(outputs["pred"]):
		if "rgb" not in pred:
			continue
		rgb = pred["rgb"].detach().cpu().numpy()
		# [B, H, W, 3] or [H, W, 3]
		if rgb.ndim == 4:
			rgb = rgb[0]
		rgb = (0.5 * (rgb + 1.0))  # [-1,1] -> [0,1]
		rgb = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
		iio.imwrite(os.path.join(save_dir, f"{idx:06d}.png"), rgb)
		num_saved += 1

	print(f"Saved {num_saved} predicted RGB frames to: {save_dir}")
	if num_saved == 0:
		print("Warning: No 'rgb' found in outputs['pred']. Your checkpoint might not output RGB.")


if __name__ == "__main__":
	main()


