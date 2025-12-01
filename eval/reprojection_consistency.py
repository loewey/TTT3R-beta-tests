#!/usr/bin/env python3
"""
Reprojection-consistency metric without GT.

For a run directory (e.g., tmp/taylor_*), we use:
  - color/: resized input frames (H,W,3)
  - depth/: predicted depth Z (H,W)
  - camera/: per-frame pose (4x4 cam2world) and intrinsics (3x3)

We inverse-warp frame t to t+1 using depth_{t+1} and poses, then compute SSIM/PSNR
between the warped image (from color_t) and the target image color_{t+1}.
We average metrics over all consecutive pairs.

Usage:
  python eval/reprojection_consistency.py \
    --run_dir /workspace/TTT3R/tmp/taylor_ttt3r \
    --out_csv /workspace/TTT3R/eval_results/reproj_ttt3r.csv
"""
import argparse
import os
import glob
import csv
import numpy as np
import cv2


def list_sorted(dir_path, ext):
	paths = sorted(glob.glob(os.path.join(dir_path, f"*.{ext}")))
	return paths


def load_color(path):
	img = cv2.imread(path, cv2.IMREAD_COLOR)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	return (img.astype(np.float32) / 255.0)  # H,W,3 in [0,1]


def compute_psnr(a, b):
	# a,b in [0,1], H,W,3
	mse = np.mean((a - b) ** 2, dtype=np.float64)
	if mse <= 1e-12:
		return 100.0
	return float(10.0 * np.log10(1.0 / mse))


def gaussian_window(ksize=11, sigma=1.5):
	g1d = cv2.getGaussianKernel(ksize, sigma)
	w = g1d @ g1d.T
	w /= np.sum(w)
	return w.astype(np.float32)


def ssim_channel(x, y, w, c1, c2):
	# x,y: HxW
	mu_x = cv2.filter2D(x, -1, w, borderType=cv2.BORDER_REFLECT)
	mu_y = cv2.filter2D(y, -1, w, borderType=cv2.BORDER_REFLECT)
	mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
	sigma_x2 = cv2.filter2D(x * x, -1, w, borderType=cv2.BORDER_REFLECT) - mu_x2
	sigma_y2 = cv2.filter2D(y * y, -1, w, borderType=cv2.BORDER_REFLECT) - mu_y2
	sigma_xy = cv2.filter2D(x * y, -1, w, borderType=cv2.BORDER_REFLECT) - mu_xy
	num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
	den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
	m = num / (den + 1e-12)
	return float(m.mean())


def compute_ssim(a, b):
	# a,b in [0,1], H,W,3
	w = gaussian_window()
	L = 1.0
	c1 = (0.01 * L) ** 2
	c2 = (0.03 * L) ** 2
	return float(np.mean([ssim_channel(a[..., c], b[..., c], w, c1, c2) for c in range(3)]))


def load_cam(npz_path):
	d = np.load(npz_path)
	pose = d["pose"].astype(np.float64)        # 4x4 cam2world
	K = d["intrinsics"].astype(np.float64)     # 3x3
	return pose, K


def invert_se3(T):
	R = T[:3, :3]
	t = T[:3, 3]
	Rt = R.T
	tinv = -Rt @ t
	Tinv = np.eye(4, dtype=np.float64)
	Tinv[:3, :3] = Rt
	Tinv[:3, 3] = tinv
	return Tinv


def warp_t_to_t1(color_t, depth_t1, K_t, K_t1, c2w_t, c2w_t1):
	"""
	Inverse warp: for each pixel in target (t1), backproject using depth_t1, transform to cam t, then sample color_t.
	Returns warped image aligned to t1 and a validity mask.
	"""
	h, w = depth_t1.shape
	# Build pixel grid
	u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
	z = depth_t1.astype(np.float64)
	# Backproject to cam t1
	fx1, fy1 = K_t1[0, 0], K_t1[1, 1]
	cx1, cy1 = K_t1[0, 2], K_t1[1, 2]
	Xc1 = np.stack([(u - cx1) * z / fx1, (v - cy1) * z / fy1, z, np.ones_like(z)], axis=-1)  # H,W,4
	# To world
	Xw = (c2w_t1 @ Xc1.reshape(-1, 4).T).T  # (H*W,4)
	# To cam t
	w2c_t = invert_se3(c2w_t)
	Xc0 = (w2c_t @ Xw.T).T  # (H*W,4)
	Xc0 = Xc0.reshape(h, w, 4)[..., :3]
	Z = Xc0[..., 2]
	# Project with K_t
	fx0, fy0 = K_t[0, 0], K_t[1, 1]
	cx0, cy0 = K_t[0, 2], K_t[1, 2]
	uu = fx0 * (Xc0[..., 0] / (Z + 1e-8)) + cx0
	vv = fy0 * (Xc0[..., 1] / (Z + 1e-8)) + cy0
	# Valid mask
	valid = (Z > 1e-6) & (uu >= 0) & (uu < w - 1) & (vv >= 0) & (vv < h - 1)
	# Sample color_t at (uu, vv)
	map_x = uu.astype(np.float32)
	map_y = vv.astype(np.float32)
	src_bgr = cv2.cvtColor((color_t * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
	warped_bgr = cv2.remap(src_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
	warped = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
	return warped, valid


def main():
	parser = argparse.ArgumentParser("Reprojection-consistency metric (no GT)")
	parser.add_argument("--run_dir", type=str, required=True, help="Directory containing color/, depth/, camera/")
	parser.add_argument("--out_csv", type=str, required=True)
	parser.add_argument("--step", type=int, default=1, help="Temporal step (default: 1 for t->t+1)")
	args = parser.parse_args()

	color_dir = os.path.join(args.run_dir, "color")
	depth_dir = os.path.join(args.run_dir, "depth")
	cam_dir = os.path.join(args.run_dir, "camera")
	assert os.path.isdir(color_dir) and os.path.isdir(depth_dir) and os.path.isdir(cam_dir), "Missing subfolders."

	color_paths = list_sorted(color_dir, "png")
	depth_paths = list_sorted(depth_dir, "npy")
	cam_paths = list_sorted(cam_dir, "npz")
	n = min(len(color_paths), len(depth_paths), len(cam_paths))
	assert n >= 2, "Not enough frames."

	os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

	records = []
	psnr_vals, ssim_vals = [], []

	for i in range(0, n - args.step):
		j = i + args.step
		# Load data
		color_i = load_color(color_paths[i])
		color_j = load_color(color_paths[j])
		depth_j = np.load(depth_paths[j]).astype(np.float32)
		pose_i, K_i = load_cam(cam_paths[i])
		pose_j, K_j = load_cam(cam_paths[j])

		warped_i_to_j, valid = warp_t_to_t1(color_i, depth_j, K_i, K_j, pose_i, pose_j)
		# Fill invalid with target to avoid penalizing
		if not np.all(valid):
			mask3 = np.repeat(valid[..., None], 3, axis=-1)
			warped_i_to_j = np.where(mask3, warped_i_to_j, color_j)

		psnr = compute_psnr(warped_i_to_j, color_j)
		ssim = compute_ssim(warped_i_to_j, color_j)
		records.append({"pair": f"{i:06d}->{j:06d}", "PSNR": psnr, "SSIM": ssim})
		psnr_vals.append(psnr)
		ssim_vals.append(ssim)

	with open(args.out_csv, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=["pair", "PSNR", "SSIM"])
		writer.writeheader()
		for r in records:
			writer.writerow(r)
		writer.writerow({
			"pair": "MEAN",
			"PSNR": float(np.mean(psnr_vals)) if psnr_vals else float("nan"),
			"SSIM": float(np.mean(ssim_vals)) if ssim_vals else float("nan"),
		})

	print(f"Pairs: {len(records)}")
	print(f"Mean PSNR: {float(np.mean(psnr_vals)):.4f} dB")
	print(f"Mean SSIM: {float(np.mean(ssim_vals)):.6f}")
	print(f"CSV: {args.out_csv}")


if __name__ == "__main__":
	main()


