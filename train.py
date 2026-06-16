#!/usr/bin/env python3
"""Train DiSIINet for joint medical image enhancement and segmentation."""

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from disiinet.data import build_dataset
from disiinet.losses import DiSIINetLoss
from disiinet.models.disiinet import DiSIINet
from disiinet.utils import load_config, save_checkpoint, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train DiSIINet")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/acdc.yaml",
        help="Path to config file",
    )
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_ds = build_dataset(
        cfg["dataset"]["name"],
        cfg["dataset"]["root"],
        "train",
        image_size=cfg["dataset"]["image_size"],
        num_classes=cfg["dataset"]["num_classes"],
    )
    val_ds = build_dataset(
        cfg["dataset"]["name"],
        cfg["dataset"]["root"],
        "val",
        image_size=cfg["dataset"]["image_size"],
        num_classes=cfg["dataset"]["num_classes"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
    )

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

    criterion = DiSIINetLoss(
        beta=cfg["loss"]["beta"],
        lambda_seg=cfg["loss"]["lambda_seg"],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1

    output_dir = cfg["train"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(output_dir, "logs"))

    global_step = 0
    for epoch in range(start_epoch, cfg["train"]["num_epochs"]):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['train']['num_epochs']}")
        for batch in pbar:
            x_lq = batch["x_lq"].to(device)
            x_gt = batch["x_gt"].to(device)
            mask = batch["mask"].to(device)

            outputs = model(x_lq, x_gt, mask)
            losses = criterion(outputs, x_gt, mask)

            optimizer.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += losses["loss"].item()
            global_step += 1

            if global_step % cfg["train"]["log_interval"] == 0:
                writer.add_scalar("train/loss", losses["loss"].item(), global_step)
                writer.add_scalar("train/ddim_enh", losses["ddim_enh"].item(), global_step)
                writer.add_scalar("train/ddim_seg", losses["ddim_seg"].item(), global_step)
                writer.add_scalar("train/enh", losses["enh"].item(), global_step)
                writer.add_scalar("train/seg", losses["seg"].item(), global_step)

            pbar.set_postfix(loss=f"{losses['loss'].item():.4f}")

        avg_loss = epoch_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch+1} - avg train loss: {avg_loss:.4f}")

        if len(val_loader) > 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x_lq = batch["x_lq"].to(device)
                    x_gt = batch["x_gt"].to(device)
                    mask = batch["mask"].to(device)
                    outputs = model(x_lq, x_gt, mask)
                    losses = criterion(outputs, x_gt, mask)
                    val_loss += losses["loss"].item()
            avg_val = val_loss / len(val_loader)
            writer.add_scalar("val/loss", avg_val, epoch)
            print(f"Epoch {epoch+1} - avg val loss: {avg_val:.4f}")

        if (epoch + 1) % cfg["train"]["save_interval"] == 0:
            ckpt_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            save_checkpoint(model, optimizer, epoch, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    final_path = os.path.join(output_dir, "checkpoint_final.pth")
    save_checkpoint(model, optimizer, cfg["train"]["num_epochs"] - 1, final_path)
    writer.close()
    print(f"Training complete. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
