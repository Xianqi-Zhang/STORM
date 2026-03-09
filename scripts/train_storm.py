#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from storm.config.loader import load_config
from storm.data.synthetic import SyntheticConfig, SyntheticStormDataset
from storm.models.storm_v1 import StormV1
from storm.training.trainer import StormTrainer
from storm.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train STORM V1 (smoke)")
    p.add_argument("--config", type=str, default="src/storm/config/default.yaml")
    p.add_argument("--output", type=str, default="outputs/storm_v1")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(cfg["seed"])

    device = torch.device("cuda" if cfg["device"] == "cuda" and torch.cuda.is_available() else "cpu")

    mcfg = cfg["model"]
    dcfg = cfg["data"]

    train_ds = SyntheticStormDataset(
        SyntheticConfig(
            size=dcfg["train_size"],
            sequence_length=dcfg["sequence_length"],
            text_len=mcfg["text_len"],
            text_vocab_size=mcfg["text_vocab_size"],
            obj_dim=mcfg["obj_dim"],
            human_dim=mcfg["human_dim"],
            robot_dim=mcfg["robot_dim"],
            graph_nodes=dcfg["graph_nodes"],
            graph_node_dim=mcfg["graph"]["node_dim"],
            contact_distance=cfg["constraints"]["contact_distance"],
        ),
        seed=cfg["seed"],
    )

    val_ds = SyntheticStormDataset(
        SyntheticConfig(
            size=dcfg["val_size"],
            sequence_length=dcfg["sequence_length"],
            text_len=mcfg["text_len"],
            text_vocab_size=mcfg["text_vocab_size"],
            obj_dim=mcfg["obj_dim"],
            human_dim=mcfg["human_dim"],
            robot_dim=mcfg["robot_dim"],
            graph_nodes=dcfg["graph_nodes"],
            graph_node_dim=mcfg["graph"]["node_dim"],
            contact_distance=cfg["constraints"]["contact_distance"],
        ),
        seed=cfg["seed"] + 1,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
    )

    model = StormV1(cfg)
    trainer = StormTrainer(model=model, cfg=cfg, device=device)
    history = trainer.fit(train_loader, val_loader)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg}, out_dir / "storm_v1.pt")

    with (out_dir / "history.txt").open("w", encoding="utf-8") as f:
        for row in history:
            f.write(f"{row.epoch}\t{row.train_loss:.6f}\t{row.val_loss:.6f}\n")

    print(f"Saved checkpoint to {out_dir / 'storm_v1.pt'}")


if __name__ == "__main__":
    main()
