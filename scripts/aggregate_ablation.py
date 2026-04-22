"""Aggregate multi-variant eval_rollout outputs into paper-ready tables/figures.

Inputs:
    One or more run directories produced by scripts/eval_rollout.py, each
    containing episodes.csv / summary.json / traces.pt.

Usage:
    python scripts/aggregate_ablation.py \\
        --runs exp_results/eval/far_jump/baseline_seed0_* \\
               exp_results/eval/far_jump/diff_seed0_* \\
               exp_results/eval/far_jump/stl_seed0_* \\
               exp_results/eval/far_jump/diff_stl_seed0_* \\
        --out_dir exp_results/eval/far_jump/aggregate_<stamp>

Outputs (in --out_dir):
    all_episodes.csv      – concatenated per-episode rows (tagged by variant)
    table.md / table.tex  – mean ± std per variant, per metric
    delta.md              – relative change vs baseline
    fig_base_z.pdf        – successful-episode base_z(t) overlaid per variant
    fig_mpjpe_violin.pdf  – per-episode MPJPE distribution
    fig_bar_success.pdf   – success / completion rate bars
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

# Paper-style column ordering.  First element of each tuple is the DataFrame
# column name; second is the display label; third is arrow ('↑'/'↓'); fourth
# is a formatting precision (number of decimals).
METRIC_COLUMNS = [
    ("success",                 "Success",              "↑", 3),
    ("completion",              "Completion",           "↑", 3),
    ("body_err_mean_mm",        "MPJPE (mm)",           "↓", 1),
    ("body_err_peak_mm",        "MPJPE-peak (mm)",      "↓", 1),
    ("joint_err_mean_rad",      "E_joint (rad)",        "↓", 3),
    ("stl_rho_mean",            "STL rho (mean)",       "↑", 3),
    ("stl_rho_min",             "STL rho (min)",        "↑", 3),
    ("apex_base_z_peak_m",      "Apex z (m)",           "↑", 3),
    ("airborne_feet_force_peak_N",
                                "Air Fz peak (N)",      "↓", 1),
    ("land_body_xy_err_min_m",  "Land xy err (m)",      "↓", 3),
]

VARIANT_ORDER = ["baseline", "diff", "stl", "diff_stl"]


# ---------------------------------------------------------------------------
def _resolve_runs(patterns: List[str]) -> List[str]:
    runs: List[str] = []
    for p in patterns:
        matches = sorted(glob.glob(p))
        if not matches:
            print(f"[aggregate] WARN: pattern matched nothing: {p}", file=sys.stderr)
        runs.extend(matches)
    uniq: List[str] = []
    for r in runs:
        if r not in uniq:
            uniq.append(r)
    return uniq


def _load_runs(run_dirs: List[str]) -> pd.DataFrame:
    frames = []
    for rd in run_dirs:
        csv = os.path.join(rd, "episodes.csv")
        if not os.path.exists(csv):
            print(f"[aggregate] skip (no episodes.csv): {rd}", file=sys.stderr)
            continue
        df = pd.read_csv(csv)
        if "tag" not in df.columns:
            df["tag"] = os.path.basename(rd).split("_seed")[0]
        df["run_dir"] = rd
        frames.append(df)
    if not frames:
        raise RuntimeError("No runs loaded.")
    return pd.concat(frames, ignore_index=True)


def _order_variants(tags: List[str]) -> List[str]:
    ordered = [v for v in VARIANT_ORDER if v in tags]
    extras = [t for t in tags if t not in ordered]
    return ordered + extras


# ---------------------------------------------------------------------------
def build_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame (index=variant, columns=display labels) of 'mean±std'."""
    variants = _order_variants(sorted(df["tag"].unique()))
    rows = []
    for tag in variants:
        sub = df[df["tag"] == tag]
        row = {"variant": tag, "n_episodes": int(len(sub))}
        for col, label, _arrow, prec in METRIC_COLUMNS:
            if col not in sub.columns:
                row[label] = "—"
                continue
            vals = sub[col].astype(float)
            if col in ("success",):
                # boolean-like; just the mean
                m = vals.mean(skipna=True)
                row[label] = f"{m:.{prec}f}"
            else:
                m = vals.mean(skipna=True)
                s = vals.std(ddof=0, skipna=True)
                if np.isnan(m):
                    row[label] = "—"
                else:
                    row[label] = f"{m:.{prec}f} ± {s:.{prec}f}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("variant")


def build_delta_table(df: pd.DataFrame, baseline: str = "baseline") -> pd.DataFrame:
    """Percentage change of means vs baseline. Positive = improvement
    w.r.t. the arrow direction (↑ metric higher / ↓ metric lower)."""
    if baseline not in df["tag"].unique():
        return pd.DataFrame()
    base = df[df["tag"] == baseline]
    variants = [v for v in _order_variants(sorted(df["tag"].unique())) if v != baseline]
    rows = []
    for tag in variants:
        sub = df[df["tag"] == tag]
        row = {"variant": tag}
        for col, label, arrow, _prec in METRIC_COLUMNS:
            if col not in sub.columns or col not in base.columns:
                row[label] = "—"
                continue
            b = float(base[col].mean(skipna=True))
            m = float(sub[col].mean(skipna=True))
            if np.isnan(b) or np.isnan(m) or abs(b) < 1e-12:
                row[label] = "—"
                continue
            rel = (m - b) / abs(b) * 100.0
            # flip sign so "+" is always an improvement
            signed = rel if arrow == "↑" else -rel
            row[label] = f"{signed:+.1f}%"
        rows.append(row)
    return pd.DataFrame(rows).set_index("variant")


def to_markdown(table: pd.DataFrame, title: str) -> str:
    headers = list(table.columns)
    lines = [f"## {title}", ""]
    lines.append("| variant | " + " | ".join(headers) + " |")
    lines.append("|" + "---|" * (len(headers) + 1))
    for idx, row in table.iterrows():
        lines.append("| " + str(idx) + " | " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines) + "\n"


def to_latex(table: pd.DataFrame, label: str = "tab:ablation") -> str:
    cols = list(table.columns)
    out = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Ablation of diffusion reference (Diff) and STL reward (STL) on \texttt{far\_jump}. Mean $\pm$ std over per-episode rollouts.}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{l" + "c" * len(cols) + "}",
        r"\toprule",
        "Variant & " + " & ".join(c.replace("%", r"\%") for c in cols) + r" \\",
        r"\midrule",
    ]
    for idx, row in table.iterrows():
        cells = [str(row[c]).replace("±", r"$\pm$") for c in cols]
        out.append(f"{idx} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
def _lazy_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def plot_success_bar(df: pd.DataFrame, out_path: str) -> None:
    plt = _lazy_mpl()
    if plt is None:
        return
    variants = _order_variants(sorted(df["tag"].unique()))
    succ = [df[df["tag"] == v]["success"].astype(float).mean() for v in variants]
    comp = [df[df["tag"] == v]["completion"].astype(float).mean() for v in variants]
    x = np.arange(len(variants))
    w = 0.35
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.bar(x - w / 2, succ, w, label="Success")
    ax.bar(x + w / 2, comp, w, label="Completion")
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_mpjpe_violin(df: pd.DataFrame, out_path: str) -> None:
    plt = _lazy_mpl()
    if plt is None or "body_err_mean_mm" not in df.columns:
        return
    variants = _order_variants(sorted(df["tag"].unique()))
    data = [df[df["tag"] == v]["body_err_mean_mm"].astype(float).dropna().values for v in variants]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.violinplot(data, showmeans=True, showextrema=False)
    ax.set_xticks(np.arange(1, len(variants) + 1))
    ax.set_xticklabels(variants)
    ax.set_ylabel("MPJPE (mm)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_base_z_traces(run_dirs: List[str], out_path: str) -> None:
    plt = _lazy_mpl()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for rd in run_dirs:
        tp = os.path.join(rd, "traces.pt")
        if not os.path.exists(tp):
            continue
        blob = torch.load(tp, map_location="cpu")
        tag = blob.get("tag", os.path.basename(rd).split("_seed")[0])
        traces = blob.get("traces", [])
        succ = [tr for tr in traces if tr.get("success", False)]
        pick = succ[:3] if succ else traces[:3]
        for tr in pick:
            t = tr["trace"]["t"]
            z = tr["trace"]["base_z"]
            ax.plot(t, z, alpha=0.7, label=f"{tag}" if tr is pick[0] else None)
    ax.set_xlabel("motion time (s)")
    ax.set_ylabel("base z (m)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="Run directories or glob patterns (episodes.csv must exist).")
    ap.add_argument("--out_dir", type=str,
                    default=f"aggregate_{time.strftime('%Y%m%d_%H%M%S')}")
    ap.add_argument("--baseline_tag", type=str, default="baseline")
    args = ap.parse_args()

    runs = _resolve_runs(args.runs)
    if not runs:
        sys.exit("[aggregate] no runs resolved")
    print(f"[aggregate] loading {len(runs)} runs")
    df = _load_runs(runs)

    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(os.path.join(args.out_dir, "all_episodes.csv"), index=False)

    main_table = build_table(df)
    delta_table = build_delta_table(df, baseline=args.baseline_tag)

    md = to_markdown(main_table, title="Ablation (mean ± std per episode)")
    if not delta_table.empty:
        md += "\n" + to_markdown(delta_table, title=f"Relative improvement vs {args.baseline_tag}")
    with open(os.path.join(args.out_dir, "table.md"), "w") as f:
        f.write(md)

    with open(os.path.join(args.out_dir, "table.tex"), "w") as f:
        f.write(to_latex(main_table))

    plot_success_bar(df, os.path.join(args.out_dir, "fig_bar_success.pdf"))
    plot_mpjpe_violin(df, os.path.join(args.out_dir, "fig_mpjpe_violin.pdf"))
    plot_base_z_traces(runs, os.path.join(args.out_dir, "fig_base_z.pdf"))

    summary = {
        "runs": runs,
        "variants": sorted(df["tag"].unique().tolist()),
        "n_episodes_total": int(len(df)),
        "n_episodes_per_variant": df.groupby("tag").size().to_dict(),
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[aggregate] wrote {args.out_dir}")
    print(md)


if __name__ == "__main__":
    main()
