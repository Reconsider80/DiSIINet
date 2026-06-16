#!/usr/bin/env python3
"""Inference script for DiSIINet."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from disiinet.data import build_dataset
from disiinet.models.disiinet import DiSIINet
from disiinet.utils import evaluate_batch, load_config


def parse_args():
    parser = argparse.ArgumentParser(description="DiSIINet inference")
    parser.add_argument("--config", type=str, default="configs/acdc.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_vis", action="store_true", help="Save visualizations")
    return parser.parse_args()


def save_image(tensor: torch.Tensor, path: str):
    arr = tensor.squeeze().detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def save_mask(tensor: torch.Tensor, path: str):
    t = tensor[0] if tensor.dim() == 4 else tensor
    pred = t.argmax(dim=0).cpu().numpy().astype(np.uint8)
    scale = 255 // max(int(pred.max()), 1)
    Image.fromarray(pred * scale).save(path)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset = build_dataset(
        cfg["dataset"]["name"],
        cfg["dataset"]["root"],
        args.split,
        image_size=cfg["dataset"]["image_size"],
        num_classes=cfg["dataset"]["num_classes"],
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model_cfg = cfg["model"]
    diff_cfg = cfg["diffusion"]
    model = DiSIINet(
        in_channels=1,
        num_classes=cfg["dataset"]["num_classes"],
        latent_channels=model_cfg["latent_channels"],
        base_channels=model_cfg["base_channels"],
        channel_mult=tuple(model_cfg["channel_mult"]),
        num_res_blocks=model_cfg["num_res_blocks"],
        attention_resolutions=tuple(model_cfg["attention_resolutions"]),
        num_heads=model_cfg["num_heads"],
        dropout=model_cfg["dropout"],
        num_train_timesteps=diff_cfg["num_train_timesteps"],
        schedule=diff_cfg["schedule"],
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    output_dir = cfg["inference"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    enh_dir = os.path.join(output_dir, "enhanced")
    seg_dir = os.path.join(output_dir, "segmentation")
    os.makedirs(enh_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)

    metrics_sum = {"psnr": 0.0, "ssim": 0.0, "dice": 0.0, "miou": 0.0}
    count = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            x_lq = batch["x_lq"].to(device)
            x_gt = batch["x_gt"].to(device)
            mask = batch["mask"].to(device)
            name = batch["name"][0]

            outputs = model.sample(
                x_lq,
                num_inference_steps=diff_cfg["num_inference_steps"],
                eta=diff_cfg["eta_infer"],
            )

            batch_metrics = evaluate_batch(
                outputs, x_gt, mask, cfg["dataset"]["num_classes"]
            )
            for k in metrics_sum:
                metrics_sum[k] += batch_metrics[k]
            count += 1

            if args.save_vis:
                save_image(outputs["ip"], os.path.join(enh_dir, f"{name}.png"))
                save_mask(outputs["sp"], os.path.join(seg_dir, f"{name}.png"))

    if count > 0:
        print("\n=== Evaluation Results ===")
        for k, v in metrics_sum.items():
            print(f"  {k.upper()}: {v / count:.4f}")

    results_path = os.path.join(output_dir, "metrics.txt")
    with open(results_path, "w") as f:
        for k, v in metrics_sum.items():
            f.write(f"{k}: {v / max(count, 1):.6f}\n")
    print(f"Metrics saved to {results_path}")


if __name__ == "__main__":
    main()
