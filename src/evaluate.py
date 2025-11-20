# src/evaluate.py
"""Independent evaluation & visualisation – no training interactions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from omegaconf import OmegaConf
from scipy import stats

sns.set_theme(style="whitegrid")


def _save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


# -----------------------------------------------------------------------------
#                         per-run processing                                     
# -----------------------------------------------------------------------------

def _process_run(entity: str, project: str, run_id: str, out_dir: Path) -> Dict[str, float]:
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    hist = run.history(keys=["step", "train_loss", "dev_exact_match", "val_loss", "lr"], pandas=True)
    summary = run.summary._json_dict
    config = dict(run.config)

    # save raw artefacts -------------------------------------------------------
    metrics_out = out_dir / "metrics.json"
    _save_json({"history": hist.to_dict("list"), "summary": summary, "config": config}, metrics_out)

    # learning curve -----------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(hist["step"], hist["train_loss"], label="train_loss", color="tab:blue")
    ax1.set_ylabel("Train loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    if "dev_exact_match" in hist.columns:
        ax2 = ax1.twinx()
        ax2.plot(hist["step"], hist["dev_exact_match"], label="dev_EM", color="tab:green")
        ax2.set_ylabel("Dev EM", color="tab:green")
        ax2.tick_params(axis="y", labelcolor="tab:green")
    fig.tight_layout()
    lc_path = out_dir / f"{run_id}_learning_curve.pdf"
    fig.savefig(lc_path)
    plt.close(fig)

    # confusion matrix dummy ---------------------------------------------------
    total_dev = summary.get("dev_set_size", 1319)
    correct = int(summary.get("final_dev_exact_match", 0.0) * total_dev)
    incorrect = total_dev - correct
    cm = [[correct, incorrect], [0, 0]]
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    cm_path = out_dir / f"{run_id}_confusion_matrix.pdf"
    fig_cm.tight_layout()
    fig_cm.savefig(cm_path)
    plt.close(fig_cm)

    print(lc_path)
    print(cm_path)
    print(metrics_out)

    return {
        "dev_exact_match": summary.get("final_dev_exact_match", 0.0),
        "joules_to_55_em": summary.get("joules_to_55_em", float("nan")),
        "inference_flops": summary.get("inference_flops", float("nan")),
    }


# -----------------------------------------------------------------------------
#                       aggregated comparison                                   
# -----------------------------------------------------------------------------

def _aggregate(all_metrics: Dict[str, Dict[str, float]], results_dir: Path):
    comp_dir = results_dir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    metric_names = {mn for m in all_metrics.values() for mn in m.keys()}
    metrics_nested: Dict[str, Dict[str, float]] = {mn: {} for mn in metric_names}
    for run_id, m in all_metrics.items():
        for mn in metric_names:
            metrics_nested[mn][run_id] = m.get(mn, float("nan"))

    primary_metric = (
        "1. GSM8K dev exact-match.\n2. Joules-to-55 % EM (from NVIDIA-Smi).\n3. Inference FLOPs & parameter count after training."
    )

    proposed_best = max(
        ((rid, m["dev_exact_match"]) for rid, m in all_metrics.items() if "proposed" in rid or "gnarum" in rid),
        key=lambda x: x[1],
        default=(None, -1.0),
    )
    baseline_best = max(
        ((rid, m["dev_exact_match"]) for rid, m in all_metrics.items() if "baseline" in rid or "comparative" in rid),
        key=lambda x: x[1],
        default=(None, -1.0),
    )
    gap = 0.0
    if baseline_best[1] > 0:
        gap = (proposed_best[1] - baseline_best[1]) / baseline_best[1] * 100

    prop_vals = [m["dev_exact_match"] for rid, m in all_metrics.items() if "proposed" in rid or "gnarum" in rid]
    base_vals = [m["dev_exact_match"] for rid, m in all_metrics.items() if "baseline" in rid or "comparative" in rid]
    p_val = float("nan")
    if len(prop_vals) >= 2 and len(base_vals) >= 2:
        _, p_val = stats.ttest_ind(prop_vals, base_vals, equal_var=False)

    agg = {
        "primary_metric": primary_metric,
        "metrics": metrics_nested,
        "best_proposed": {"run_id": proposed_best[0], "value": proposed_best[1]},
        "best_baseline": {"run_id": baseline_best[0], "value": baseline_best[1]},
        "gap": gap,
        "p_value": p_val,
    }
    agg_path = comp_dir / "aggregated_metrics.json"
    _save_json(agg, agg_path)
    print(agg_path)

    # bar chart ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    run_ids = list(all_metrics.keys())
    vals = [all_metrics[r]["dev_exact_match"] for r in run_ids]
    sns.barplot(x=run_ids, y=vals, ax=ax, palette="viridis")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
    ax.set_ylabel("Dev Exact-Match")
    ax.set_xlabel("Run ID")
    ax.set_xticklabels(run_ids, rotation=45, ha="right")
    fig.tight_layout()
    bar_path = comp_dir / "comparison_accuracy_bar_chart.pdf"
    fig.savefig(bar_path)
    plt.close(fig)
    print(bar_path)

    # boxplot -----------------------------------------------------------------
    cats = ["proposed" if ("proposed" in rid or "gnarum" in rid) else "baseline" for rid in run_ids]
    fig_b, ax_b = plt.subplots(figsize=(4, 4))
    sns.boxplot(x=cats, y=vals, ax=ax_b)
    ax_b.set_ylabel("Dev Exact-Match")
    fig_b.tight_layout()
    box_path = comp_dir / "comparison_exact_match_boxplot.pdf"
    fig_b.savefig(box_path)
    plt.close(fig_b)
    print(box_path)


# -----------------------------------------------------------------------------
#                                    CLI                                        
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Aggregate WandB runs for GSM8K experiments")
    ap.add_argument("results_dir", type=str)
    ap.add_argument("run_ids", type=str, help='JSON list, e.g. "[\"run1\", \"run2\"]"')
    args = ap.parse_args()

    run_ids: List[str] = json.loads(args.run_ids)
    results_dir = Path(args.results_dir).expanduser().resolve()

    cfg_global = OmegaConf.load(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    entity, project = cfg_global.wandb.entity, cfg_global.wandb.project

    all_metrics: Dict[str, Dict[str, float]] = {}
    for rid in run_ids:
        out_dir = results_dir / rid
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics = _process_run(entity, project, rid, out_dir)
        all_metrics[rid] = metrics

    _aggregate(all_metrics, results_dir)


if __name__ == "__main__":
    main()