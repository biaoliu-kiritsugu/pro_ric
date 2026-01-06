#!/usr/bin/env python3
"""
python /data/liubiao/llm/a800_2/RiC/pro_ric/analyze_score_distribution.py \
  --dataset_path ./datasets/summary_pref1faithfuldeberta_messages.hf \
  --temperature 0.5 \
  --rate 10 \
  --plot_dir ./score_dist_plots \

Analyze score distributions in a dataset.

Supports:
- HuggingFace `load_from_disk()` dataset directories
- JSON/JSONL files via `load_dataset("json")`

Outputs:
- Raw score mean/std/min/max per score column
- "Normalized" mean: mean of softmax(scores / temperature) across samples
  - also reports scaled-by-rate mean (normalized_probs * rate)
- Optional histogram plots (on a sampled subset)
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from datasets import load_dataset, load_from_disk


def _detect_score_columns(column_names: List[str]) -> List[str]:
    # Prefer canonical offline columns: score1, score2, ...
    score_cols = [c for c in column_names if c.startswith("score") and c[5:].isdigit()]
    score_cols.sort(key=lambda x: int(x[5:]))
    if score_cols:
        return score_cols

    # Fallback: obtained_score1, obtained_score2, ...
    score_cols = [c for c in column_names if c.startswith("obtained_score") and c[len("obtained_score") :].isdigit()]
    score_cols.sort(key=lambda x: int(x[len("obtained_score") :]))
    return score_cols


def _load_any_dataset(path: str):
    if os.path.isdir(path):
        return load_from_disk(path)
    lower = path.lower()
    if lower.endswith(".json") or lower.endswith(".jsonl"):
        return load_dataset("json", data_files=path, split="train")
    # Best-effort fallback: try load_from_disk, else json
    try:
        return load_from_disk(path)
    except Exception:
        return load_dataset("json", data_files=path, split="train")


def _reservoir_update(reservoir: np.ndarray, seen: int, values: np.ndarray, rng: np.random.Generator) -> int:
    """
    Reservoir sampling update.
    - reservoir: shape (k, d)
    - values: shape (m, d)
    Returns updated `seen` count.
    """
    k = reservoir.shape[0]
    for v in values:
        seen += 1
        if seen <= k:
            reservoir[seen - 1] = v
        else:
            j = rng.integers(0, seen)
            if j < k:
                reservoir[j] = v
    return seen


def analyze(
    dataset,
    score_cols: List[str],
    temperature: float,
    rate: float,
    max_samples: Optional[int],
    reservoir_size: int,
    seed: int,
) -> Dict:
    n_total = len(dataset)
    n = min(n_total, max_samples) if max_samples is not None else n_total
    d = len(score_cols)

    # Running stats
    s1 = np.zeros(d, dtype=np.float64)
    s2 = np.zeros(d, dtype=np.float64)
    vmin = np.full(d, np.inf, dtype=np.float64)
    vmax = np.full(d, -np.inf, dtype=np.float64)

    # Normalized mean (softmax)
    p_sum = np.zeros(d, dtype=np.float64)

    # For distribution plots/quantiles: reservoir sample of vectors
    rng = np.random.default_rng(seed)
    k = min(reservoir_size, n)
    reservoir = np.zeros((k, d), dtype=np.float32)
    seen = 0

    # Iterate without materializing giant arrays
    # datasets supports slicing; keep it simple and chunked
    batch_size = 2048
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = dataset.select(range(start, end))
        mat = np.stack([np.asarray(batch[c], dtype=np.float32) for c in score_cols], axis=1)  # (bs, d)

        s1 += mat.sum(axis=0)
        s2 += (mat.astype(np.float64) ** 2).sum(axis=0)
        vmin = np.minimum(vmin, mat.min(axis=0))
        vmax = np.maximum(vmax, mat.max(axis=0))

        # softmax(scores / temperature)
        x = mat.astype(np.float64) / float(temperature)
        x = x - x.max(axis=1, keepdims=True)  # stable
        ex = np.exp(x)
        p = ex / ex.sum(axis=1, keepdims=True)
        p_sum += p.sum(axis=0)

        seen = _reservoir_update(reservoir, seen, mat.astype(np.float32), rng)

    mean = s1 / n
    var = np.maximum(s2 / n - mean**2, 0.0)
    std = np.sqrt(var)
    p_mean = p_sum / n
    scaled_mean = p_mean * float(rate)

    # Quantiles from reservoir
    q = {}
    for qi in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
        q[str(qi)] = np.quantile(reservoir[: min(seen, k)], qi, axis=0).astype(np.float64)

    return {
        "n_total": n_total,
        "n_used": n,
        "score_cols": score_cols,
        "raw": {"mean": mean, "std": std, "min": vmin, "max": vmax, "quantiles": q},
        "normalized": {"temperature": float(temperature), "rate": float(rate), "mean_prob": p_mean, "mean_scaled": scaled_mean},
        "reservoir": reservoir[: min(seen, k)],
    }


def maybe_plot(out_dir: str, score_cols: List[str], reservoir: np.ndarray, temperature: float, rate: float, bins: int) -> None:
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    d = len(score_cols)

    # Raw histograms
    for i, col in enumerate(score_cols):
        plt.figure(figsize=(7, 4))
        plt.hist(reservoir[:, i], bins=bins, alpha=0.85)
        plt.title(f"Raw distribution: {col}")
        plt.xlabel(col)
        plt.ylabel("count (sampled)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"hist_raw_{col}.png"))
        plt.close()

    # Normalized/scaled distributions
    x = reservoir.astype(np.float64) / float(temperature)
    x = x - x.max(axis=1, keepdims=True)
    ex = np.exp(x)
    p = ex / ex.sum(axis=1, keepdims=True)
    scaled = p * float(rate)
    for i, col in enumerate(score_cols):
        plt.figure(figsize=(7, 4))
        plt.hist(scaled[:, i], bins=bins, alpha=0.85)
        plt.title(f"Normalized (softmax/t={temperature}) * rate={rate}: {col}")
        plt.xlabel(f"{col}_scaled")
        plt.ylabel("count (sampled)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"hist_norm_scaled_{col}.png"))
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="HF dataset dir (load_from_disk) or .json/.jsonl file")
    parser.add_argument("--temperature", type=float, default=0.5, help="softmax temperature for normalization")
    parser.add_argument("--rate", type=float, default=10.0, help="scale factor applied after softmax")
    parser.add_argument("--max_samples", type=int, default=None, help="optional cap to only analyze first N samples")
    parser.add_argument("--reservoir_size", type=int, default=200000, help="sample size for quantiles/histograms")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot_dir", type=str, default=None, help="if set, save histogram PNGs to this directory")
    parser.add_argument("--bins", type=int, default=60)
    args = parser.parse_args()

    ds = _load_any_dataset(args.dataset_path)
    score_cols = _detect_score_columns(list(ds.column_names))
    if not score_cols:
        raise ValueError(f"No score columns found. columns={ds.column_names}")

    res = analyze(
        dataset=ds,
        score_cols=score_cols,
        temperature=args.temperature,
        rate=args.rate,
        max_samples=args.max_samples,
        reservoir_size=args.reservoir_size,
        seed=args.seed,
    )

    # Pretty print
    raw = res["raw"]
    norm = res["normalized"]
    cols = res["score_cols"]

    print(f"dataset_path: {args.dataset_path}")
    print(f"n_total={res['n_total']}, n_used={res['n_used']}")
    print(f"score_cols={cols}")
    print("")
    print("== Raw score stats ==")
    for i, c in enumerate(cols):
        print(
            f"{c:>16s}: mean={raw['mean'][i]: .4f} std={raw['std'][i]: .4f} "
            f"min={raw['min'][i]: .4f} max={raw['max'][i]: .4f} median={raw['quantiles']['0.5'][i]: .4f}"
        )
    print("")
    print("== Normalized mean (softmax(scores/temperature)) ==")
    print(f"temperature={norm['temperature']}, rate={norm['rate']}")
    print("mean_prob (sums to 1):")
    for i, c in enumerate(cols):
        print(f"{c:>16s}: {norm['mean_prob'][i]: .6f}")
    print("mean_scaled (mean_prob * rate):")
    for i, c in enumerate(cols):
        print(f"{c:>16s}: {norm['mean_scaled'][i]: .6f}")

    if args.plot_dir:
        maybe_plot(args.plot_dir, cols, res["reservoir"], args.temperature, args.rate, args.bins)
        print("")
        print(f"Saved plots to: {args.plot_dir}")


if __name__ == "__main__":
    main()


