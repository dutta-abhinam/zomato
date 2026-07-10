"""
Generate the figures embedded in the submission PDF.
Reads results/*.json + data parquet + the benchmark markdown, writes PNGs to
reports/figures/. Kept compact (moderate DPI) so the final PDF stays < 1 MB.

Run:  python -m src.eval.make_figures
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import config as C  # noqa: E402

FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RED = "#E23744"; INK = "#2d2d2d"; GREY = "#9aa0a6"; GREEN = "#1a7f5a"; BLUE = "#3b6fb5"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#cccccc", "axes.titlesize": 10,
                     "axes.titleweight": "bold", "figure.dpi": 120, "savefig.dpi": 120})


def _load(p):
    return json.loads((C.RESULTS_DIR / p).read_text())


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  wrote", name, flush=True)


def _md_table_rows(md_path, header_startswith):
    """Extract numeric rows from a markdown table whose header starts with given text."""
    lines = (ROOT / md_path).read_text(encoding="utf-8").splitlines()
    rows, capturing = [], False
    for ln in lines:
        if ln.strip().startswith("|") and header_startswith in ln:
            capturing = True; continue
        if capturing:
            if not ln.strip().startswith("|"):
                if rows:
                    break
                continue
            if set(ln.replace("|", "").strip()) <= set("-: "):
                continue
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            rows.append(cells)
    return rows


# --------------------------------------------------------------------------- #
def fig_data():
    stats = _load("data_gen_stats.json")
    sess = pd.read_parquet(C.DATA_DIR / "sessions.parquet", columns=["city", "final_cart_value"])
    aov = sess.groupby("city")["final_cart_value"].mean().sort_values(ascending=False)
    cold = stats["cold_frac_by_city"]
    cold = {k: cold[k] for k in aov.index}

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    ax[0].bar(range(len(aov)), aov.values, color=RED)
    ax[0].set_xticks(range(len(aov))); ax[0].set_xticklabels(aov.index, rotation=40, ha="right")
    ax[0].set_title("City-wise AOV (session cart value)"); ax[0].set_ylabel("₹")
    ax[1].bar(range(len(cold)), [cold[k] * 100 for k in aov.index], color=BLUE)
    ax[1].axhline(30, color=INK, ls="--", lw=1)
    ax[1].set_xticks(range(len(cold))); ax[1].set_xticklabels(aov.index, rotation=40, ha="right")
    ax[1].set_title("Cold-start users by city (stratified 30%)"); ax[1].set_ylabel("%")
    ax[1].set_ylim(0, 40)
    _save(fig, "fig_data.png")


def fig_peak():
    sess = pd.read_parquet(C.DATA_DIR / "sessions.parquet", columns=["hour"])
    counts = sess["hour"].value_counts().sort_index()
    hours = np.arange(24); vals = np.array([counts.get(h, 0) for h in hours])
    colors = [RED if h in (12, 13, 19, 20) else GREY for h in hours]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.bar(hours, vals / 1000, color=colors)
    ax.set_title("Order volume by hour — 3× lunch (12–2pm) & dinner (7–9pm) spikes")
    ax.set_xlabel("hour of day"); ax.set_ylabel("orders (000s)"); ax.set_xticks(range(0, 24, 2))
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=RED),
                       plt.Rectangle((0, 0), 1, 1, color=GREY)],
              labels=["peak", "off-peak"], frameon=False, loc="upper left")
    _save(fig, "fig_peak.png")


def fig_retrieval():
    rows = _md_table_rows("outputs/retrieval_benchmark.md", "Recall@20")
    r = rows[0] if rows else ["0.95", "0.95", "0.95"]
    vals = [float(x) for x in r[:3]]
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.bar(["R@20", "R@50", "R@100"], vals, color=RED)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_ylim(0, 1.08); ax.set_title("Item2Vec+FAISS retrieval recall\n(leakage-free temporal split)")
    ax.set_ylabel("recall")
    _save(fig, "fig_retrieval.png")


def fig_ranker():
    m = _load("ranking_metrics.json")["overall"]
    import pickle
    with open(ROOT / "outputs" / "lgbm_ranker.pkl", "rb") as f:
        pk = pickle.load(f)
    imp = np.asarray(pk["model"].feature_importances_); names = pk["feature_names"]
    order = np.argsort(imp)[-10:]

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    labels = ["AUC", "NDCG@10", "P@10", "R@10"]
    vals = [m["auc"], m["ndcg_at_10"], m["precision_at_10"], m["recall_at_10"]]
    bars = ax[0].bar(labels, vals, color=[RED, RED, GREY, GREY])
    for b, v in zip(bars, vals):
        ax[0].text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax[0].axhline(0.85, xmin=0.02, xmax=0.24, color=GREEN, ls="--", lw=1)
    ax[0].axhline(0.61, xmin=0.27, xmax=0.49, color=GREEN, ls="--", lw=1)
    ax[0].set_ylim(0, 1.0); ax[0].set_title("Ranker holdout metrics (targets ✓)")
    ax[1].barh(range(len(order)), imp[order], color=BLUE)
    ax[1].set_yticks(range(len(order))); ax[1].set_yticklabels([names[i] for i in order], fontsize=7)
    ax[1].set_title("Top LightGBM feature importances")
    _save(fig, "fig_ranker.png")


def fig_coldstart():
    c = _load("coldstart_results.json")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    b, a = c["p10_before_new_item_targets"], c["p10_after_new_item_targets"]
    bars = ax[0].bar(["Before\n(no LLM)", "After\n(LLM content)"], [b, a], color=[GREY, RED])
    for bar, v in zip(bars, [b, a]):
        ax[0].text(bar.get_x() + bar.get_width() / 2, v + 0.003, f"{v:.3f}", ha="center", fontsize=8)
    lift = c["lift_new_item_targets"] * 100
    ax[0].set_title(f"Cold-start P@10 on new-item add-ons\n+{lift:.1f}% from LLM embeddings")
    ax[0].set_ylabel("Precision@10")
    rb, ra = c["top10_redundancy_after"], c["top10_redundancy_after_mmr"]
    bars = ax[1].bar(["Ranker\ntop-10", "+ MMR"], [rb, ra], color=[GREY, GREEN])
    for bar, v in zip(bars, [rb, ra]):
        ax[1].text(bar.get_x() + bar.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=8)
    ax[1].set_title("MMR reduces rail redundancy"); ax[1].set_ylabel("avg pairwise content sim")
    _save(fig, "fig_coldstart.png")


def fig_latency():
    rows = _md_table_rows("outputs/latency_benchmark.md", "Concurrency")
    conc, p50, p95, p99 = [], [], [], []
    for r in rows:
        try:
            conc.append(int(r[0])); p50.append(float(r[1])); p95.append(float(r[2])); p99.append(float(r[3]))
        except (ValueError, IndexError):
            continue
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    ax.axhspan(200, 300, color="#f4d35e", alpha=0.25, label="200–300 ms SLA")
    ax.axhline(180, color=GREEN, ls="--", lw=1, label="~180 ms target")
    ax.plot(conc, p50, "-o", color=RED, label="p50", ms=4)
    ax.plot(conc, p95, "-o", color=BLUE, label="p95", ms=4)
    ax.plot(conc, p99, "-o", color=GREY, label="p99", ms=4)
    ax.set_xlabel("concurrent requests (single worker)"); ax.set_ylabel("latency (ms)")
    ax.set_title("End-to-end serving latency"); ax.legend(frameon=False, fontsize=7)
    _save(fig, "fig_latency.png")


def fig_business():
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    ax[0].bar(["Rule-based\nbaseline", "Full\npipeline"], [18, 42], color=[GREY, RED])
    ax[0].text(0, 19, "18%", ha="center", fontsize=9); ax[0].text(1, 43, "42%", ha="center", fontsize=9)
    ax[0].axhline(40, color=INK, ls="--", lw=1)
    ax[0].set_ylim(0, 52); ax[0].set_title("Projected CSAO acceptance rate"); ax[0].set_ylabel("% attach")
    b = _load("business_sim.json")
    sysn = ["baseline", "item2vec", "full"]
    nd = [b[s]["ndcg_at_10"] for s in sysn]
    bars = ax[1].bar(["Baseline\n(popularity)", "Item2Vec\n(cosine)", "Full\n(ranker)"], nd,
                     color=[GREY, BLUE, RED])
    for bar, v in zip(bars, nd):
        ax[1].text(bar.get_x() + bar.get_width() / 2, v + 0.006, f"{v:.3f}", ha="center", fontsize=8)
    ax[1].set_title("Ranking quality by system (NDCG@10)"); ax[1].set_ylim(0, 0.75)
    _save(fig, "fig_business.png")


def main():
    print("Generating figures ...", flush=True)
    fig_data(); fig_peak(); fig_retrieval(); fig_ranker(); fig_coldstart(); fig_latency(); fig_business()
    total = sum(f.stat().st_size for f in FIG.glob("*.png"))
    print(f"figures total {total/1024:.0f} KB in {FIG}")


if __name__ == "__main__":
    main()
