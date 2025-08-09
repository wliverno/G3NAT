#!/usr/bin/env python3
"""
Lightweight ablation runner for G3NAT.

Usage (PowerShell examples):
  python ablate.py --config configs/ablate.json --out out/ablate_run
  # or with YAML (if PyYAML is installed)
  python ablate.py --config configs/ablate.yaml --out out/ablate_run

If no --config is provided, a small default set of variants is used.

Outputs:
  - results.csv: per-variant metrics
  - results.md: markdown table summary
  - per-variant JSON files with full metrics
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from models import (
    DNATransportGNN,
    DNATransportHamiltonianGNN,
    train_model,
)
from data_generator import create_sample_data
from dataset import create_dna_dataset

# Import helpers from main without side effects
try:
    from main import split_dataset, LengthBucketBatchSampler
except Exception:
    # Fallbacks if import path changes; define minimal split
    from sklearn.model_selection import train_test_split
    def split_dataset(dataset, train_split: float = 0.8):
        from torch.utils.data import Subset
        dataset_size = len(dataset)
        train_indices, val_indices = train_test_split(
            range(dataset_size), test_size=1-train_split, random_state=42
        )
        return Subset(dataset, train_indices), Subset(dataset, val_indices)
    LengthBucketBatchSampler = None  # Not strictly required if sequence lengths are constant


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class DataSpec:
    num_samples: int = 1000
    train_seq_length: int = 8
    eval_seq_length: Optional[int] = None  # if None, use train_seq_length
    min_length: int = -1  # set = train_seq_length for uniform-length batches
    num_energy_points: int = 100


@dataclass
class TrainSpec:
    epochs: int = 60
    batch_size: int = 32
    learning_rate: float = 1e-3
    device: str = "auto"


@dataclass
class Variant:
    name: str
    model: str  # "standard" | "hamiltonian"
    params: Dict[str, Any] = field(default_factory=dict)


def build_model(variant: Variant, energy_grid: np.ndarray) -> torch.nn.Module:
    if variant.model == "standard":
        return DNATransportGNN(
            hidden_dim=variant.params.get("hidden_dim", 128),
            num_layers=variant.params.get("num_layers", 4),
            num_heads=variant.params.get("num_heads", 4),
            output_dim=len(energy_grid),
            dropout=variant.params.get("dropout", 0.1),
        )
    elif variant.model == "hamiltonian":
        return DNATransportHamiltonianGNN(
            hidden_dim=variant.params.get("hidden_dim", 128),
            num_layers=variant.params.get("num_layers", 4),
            num_heads=variant.params.get("num_heads", 4),
            energy_grid=energy_grid,
            dropout=variant.params.get("dropout", 0.1),
            n_orb=variant.params.get("n_orb", 1),
            enforce_hermiticity=variant.params.get("enforce_hermiticity", True),
            solver_type=variant.params.get("solver_type", "frobenius"),
            use_log_outputs=variant.params.get("use_log_outputs", True),
            log_floor=variant.params.get("log_floor", 1e-16),
            complex_eta=variant.params.get("complex_eta", 1e-12),
        )
    else:
        raise ValueError(f"Unknown model type: {variant.model}")


def create_loaders(
    sequences: List[str],
    sequences_complementary: List[str],
    dos_data: List[np.ndarray],
    trans_data: List[np.ndarray],
    energy_grid: np.ndarray,
    batch_size: int,
    is_hamiltonian: bool,
):
    dataset = create_dna_dataset(
        sequences=sequences,
        dos_data=np.array(dos_data),
        transmission_data=np.array(trans_data),
        energy_grid=energy_grid,
        complementary_sequences=sequences_complementary,
    )
    train_ds, val_ds = split_dataset(dataset, train_split=0.8)

    if is_hamiltonian and LengthBucketBatchSampler is not None:
        train_sampler = LengthBucketBatchSampler(train_ds, batch_size=batch_size, shuffle=True)
        val_sampler = LengthBucketBatchSampler(val_ds, batch_size=batch_size, shuffle=False)
        train_loader = DataLoader(train_ds, batch_sampler=train_sampler)
        val_loader = DataLoader(val_ds, batch_sampler=val_sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


@torch.no_grad()
def evaluate_metrics(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    dos_preds, trans_preds, dos_tgts, trans_tgts = [], [], [], []
    finite_dos, finite_trans = 0, 0
    total_dos, total_trans = 0, 0

    for batch in loader:
        batch = batch.to(device)
        dos_pred, trans_pred = model(batch)
        dos_preds.append(dos_pred.detach().cpu().numpy())
        trans_preds.append(trans_pred.detach().cpu().numpy())
        # Targets
        bs = dos_pred.size(0)
        ne = dos_pred.size(1)
        dos_tgts.append(batch.dos.view(bs, ne).detach().cpu().numpy())
        trans_tgts.append(batch.transmission.view(bs, ne).detach().cpu().numpy())

        finite_dos += np.isfinite(dos_pred.detach().cpu().numpy()).sum()
        finite_trans += np.isfinite(trans_pred.detach().cpu().numpy()).sum()
        total_dos += bs * ne
        total_trans += bs * ne

    dos_pred_arr = np.concatenate(dos_preds, axis=0)
    trans_pred_arr = np.concatenate(trans_preds, axis=0)
    dos_tgt_arr = np.concatenate(dos_tgts, axis=0)
    trans_tgt_arr = np.concatenate(trans_tgts, axis=0)

    def mae(a, b):
        return float(np.mean(np.abs(a - b)))

    def nrmse(a, b):
        rmse = np.sqrt(np.mean((a - b) ** 2))
        denom = np.maximum(np.abs(b).mean(), 1e-12)
        return float(rmse / denom)

    metrics = {
        "dos_mae": mae(dos_pred_arr, dos_tgt_arr),
        "dos_nrmse": nrmse(dos_pred_arr, dos_tgt_arr),
        "trans_mae": mae(trans_pred_arr, trans_tgt_arr),
        "trans_nrmse": nrmse(trans_pred_arr, trans_tgt_arr),
        "dos_finite_ratio": float(finite_dos / max(1, total_dos)),
        "trans_finite_ratio": float(finite_trans / max(1, total_trans)),
    }
    return metrics


def run_variant(
    variant: Variant,
    data_spec: DataSpec,
    train_spec: TrainSpec,
    out_dir: str,
    seed: int,
) -> Dict[str, Any]:
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    # Data generation (train)
    train_min_length = data_spec.min_length if data_spec.min_length != -1 else data_spec.train_seq_length
    seqs, seqs_comp, dos_list, trans_list, energy_grid = create_sample_data(
        num_samples=data_spec.num_samples,
        seq_length=data_spec.train_seq_length,
        num_energy_points=data_spec.num_energy_points,
        min_length=train_min_length,
    )

    # Data generation (eval/generalization)
    eval_seq_length = data_spec.eval_seq_length or data_spec.train_seq_length
    eval_min_length = train_min_length if data_spec.eval_seq_length is None else (
        data_spec.eval_seq_length if data_spec.min_length == -1 else data_spec.min_length
    )
    seqs_eval, seqs_comp_eval, dos_list_eval, trans_list_eval, _ = create_sample_data(
        num_samples=max(200, data_spec.num_samples // 5),
        seq_length=eval_seq_length,
        num_energy_points=data_spec.num_energy_points,
        min_length=eval_min_length,
    )

    is_ham = (variant.model == "hamiltonian")
    train_loader, val_loader = create_loaders(
        seqs, seqs_comp, dos_list, trans_list, energy_grid, train_spec.batch_size, is_ham
    )
    eval_loader, _ = create_loaders(
        seqs_eval, seqs_comp_eval, dos_list_eval, trans_list_eval, energy_grid, train_spec.batch_size, is_ham
    )

    device = torch.device('cuda' if (train_spec.device == 'auto' and torch.cuda.is_available()) else (
        train_spec.device if train_spec.device != 'auto' else 'cpu'))

    # Build model
    model = build_model(variant, energy_grid)

    # Train
    t0 = time.time()
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=train_spec.epochs,
        learning_rate=train_spec.learning_rate,
        device=str(device),
        checkpoint_dir=None,
        checkpoint_frequency=10,
    )
    train_time = time.time() - t0

    # Evaluate on validation split
    t1 = time.time()
    val_metrics = evaluate_metrics(model.to(device), val_loader, device)
    val_time = time.time() - t1

    # Evaluate on eval set (potentially different length)
    t2 = time.time()
    gen_metrics = evaluate_metrics(model.to(device), eval_loader, device)
    gen_time = time.time() - t2

    # Aggregate
    result = {
        "name": variant.name,
        "model": variant.model,
        "params": variant.params,
        "epochs": train_spec.epochs,
        "batch_size": train_spec.batch_size,
        "learning_rate": train_spec.learning_rate,
        "train_seq_length": data_spec.train_seq_length,
        "eval_seq_length": eval_seq_length,
        "num_energy_points": data_spec.num_energy_points,
        "train_time_sec": train_time,
        "val_time_sec": val_time,
        "gen_time_sec": gen_time,
        "train_loss_last": float(train_losses[-1]) if len(train_losses) > 0 else None,
        "val_loss_last": float(val_losses[-1]) if len(val_losses) > 0 else None,
    }
    result.update({f"val_{k}": v for k, v in val_metrics.items()})
    result.update({f"gen_{k}": v for k, v in gen_metrics.items()})

    # Persist per-variant JSON
    with open(os.path.join(out_dir, f"{variant.name}.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def write_summaries(results: List[Dict[str, Any]], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    # CSV
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "results.csv"), index=False)

    # Markdown summary (selected columns)
    cols = [
        "name", "model", "epochs", "train_seq_length", "eval_seq_length",
        "val_dos_mae", "val_trans_mae", "val_dos_nrmse", "val_trans_nrmse",
        "gen_dos_mae", "gen_trans_mae", "gen_dos_nrmse", "gen_trans_nrmse",
        "train_time_sec"
    ]
    cols = [c for c in cols if c in df.columns]
    with open(os.path.join(out_dir, "results.md"), "w") as f:
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "---|" * len(cols) + "\n")
        for _, row in df.iterrows():
            f.write("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |\n")


def load_config(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return {}
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r") as f:
        if ext in (".yml", ".yaml"):
            try:
                import yaml  # optional dep
                return yaml.safe_load(f)
            except Exception:
                raise RuntimeError("PyYAML not installed; install pyyaml or use JSON config.")
        else:
            return json.load(f)


def build_default_variants() -> List[Variant]:
    return [
        Variant(
            name="std_direct",
            model="standard",
            params=dict(hidden_dim=128, num_layers=4, num_heads=4, dropout=0.1),
        ),
        Variant(
            name="ham_frob_herm_log",
            model="hamiltonian",
            params=dict(
                hidden_dim=128, num_layers=4, num_heads=4, dropout=0.1,
                n_orb=1, enforce_hermiticity=True, solver_type="frobenius",
                use_log_outputs=True, log_floor=1e-16,
            ),
        ),
        Variant(
            name="ham_complex_herm_log",
            model="hamiltonian",
            params=dict(
                hidden_dim=128, num_layers=4, num_heads=4, dropout=0.1,
                n_orb=1, enforce_hermiticity=True, solver_type="complex",
                use_log_outputs=True, log_floor=1e-16, complex_eta=1e-12,
            ),
        ),
        Variant(
            name="ham_frob_noherm_log",
            model="hamiltonian",
            params=dict(
                hidden_dim=128, num_layers=4, num_heads=4, dropout=0.1,
                n_orb=1, enforce_hermiticity=False, solver_type="frobenius",
                use_log_outputs=True, log_floor=1e-16,
            ),
        ),
        Variant(
            name="ham_frob_herm_nolog",
            model="hamiltonian",
            params=dict(
                hidden_dim=128, num_layers=4, num_heads=4, dropout=0.1,
                n_orb=1, enforce_hermiticity=True, solver_type="frobenius",
                use_log_outputs=False,
            ),
        ),
    ]


def main():
    parser = argparse.ArgumentParser(description="Ablation runner for G3NAT")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON/YAML config of variants")
    parser.add_argument("--out", type=str, default="./outputs/ablate", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Data spec
    data_cfg = cfg.get("data", {})
    data_spec = DataSpec(
        num_samples=int(data_cfg.get("num_samples", 1000)),
        train_seq_length=int(data_cfg.get("train_seq_length", 8)),
        eval_seq_length=(None if data_cfg.get("eval_seq_length", None) is None else int(data_cfg.get("eval_seq_length"))),
        min_length=int(data_cfg.get("min_length", -1)),
        num_energy_points=int(data_cfg.get("num_energy_points", 100)),
    )

    # Train spec
    train_cfg = cfg.get("train", {})
    train_spec = TrainSpec(
        epochs=int(train_cfg.get("epochs", 60)),
        batch_size=int(train_cfg.get("batch_size", 32)),
        learning_rate=float(train_cfg.get("learning_rate", 1e-3)),
        device=str(train_cfg.get("device", "auto")),
    )

    # Variants
    variants_cfg = cfg.get("variants", None)
    if variants_cfg is None:
        variants = build_default_variants()
    else:
        variants = [Variant(name=v["name"], model=v["model"], params=v.get("params", {})) for v in variants_cfg]

    # Run
    results: List[Dict[str, Any]] = []
    os.makedirs(args.out, exist_ok=True)
    for variant in variants:
        print(f"\n=== Running variant: {variant.name} ({variant.model}) ===")
        variant_dir = os.path.join(args.out, variant.name)
        os.makedirs(variant_dir, exist_ok=True)
        res = run_variant(variant, data_spec, train_spec, variant_dir, args.seed)
        results.append(res)

    write_summaries(results, args.out)
    print(f"\nAblation finished. Results saved to: {args.out}")


if __name__ == "__main__":
    main()


