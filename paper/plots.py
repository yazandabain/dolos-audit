"""Rebuild standalone figures from the saved experimental summaries."""

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".local/matplotlib-cache"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DEST = ROOT / "paper/figures"
INPUTS = [
    "results/summaries/discovery-001.json",
    "results/confirmation-main-analysis.json",
    "results/weight-scan-001.json",
]


def load(path):
    return json.loads((ROOT / path).read_text())


def save(fig, name):
    for suffix in ["png", "pdf", "svg"]:
        metadata = (
            {"CreationDate": None, "ModDate": None}
            if suffix == "pdf"
            else ({"Date": None} if suffix == "svg" else None)
        )
        fig.savefig(DEST / f"{name}.{suffix}", dpi=300, bbox_inches="tight", metadata=metadata)
    plt.close(fig)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "dolos-audit-figures",
            "axes.axisbelow": True,
        }
    )
    groups = {
        (r["model"], r["family"]): r for r in load(INPUTS[0])["groups"] if r["condition"] == "all"
    }
    families = ["arithmetic", "extraction", "incentive_choice", "pattern"]
    strict = [
        100 * (groups["dolos", f]["strict_correct"] - groups["qwen", f]["strict_correct"]) / 64
        for f in families
    ]
    relaxed = [
        100 * (groups["dolos", f]["relaxed_correct"] - groups["qwen", f]["relaxed_correct"]) / 64
        for f in families
    ]
    fig, ax = plt.subplots(figsize=(6.5, 2.55), layout="constrained")
    x = np.arange(4)
    w = 0.32
    ax.bar(x - w / 2, strict, w, label="Strict JSON success", color="#ADB9C5")
    ax.bar(x + w / 2, relaxed, w, label="Complete JSON fences accepted", color="#245A81")
    for xx, values in [(x - w / 2, strict), (x + w / 2, relaxed)]:
        for a, b in zip(xx, values, strict=True):
            ax.text(
                a,
                b + (2 if b >= 0 else -2),
                f"{b:+.1f}",
                ha="center",
                va="bottom" if b >= 0 else "top",
                fontsize=8,
            )
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylim(-70, 85)
    ax.set_xticks(x, ["Arithmetic", "Extraction", "Incentive choice", "Sequence"])
    ax.set_ylabel("Dolos − Qwen (percentage points)")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=8)
    save(fig, "figure-1-scoring")
    result = load(INPUTS[1])["contrasts"]
    names = [
        "format_dolos_minus_qwen",
        "accepted_dolos_minus_qwen",
        "format_restore_minus_identity",
        "format_restore_minus_random_mean",
        "capability_restore_minus_identity",
        "capability_random_mean_minus_identity",
    ]
    labels = [
        "D−Q: strict format",
        "D−Q: accepted answer",
        "Restored−identity: format",
        "Restored−random: format",
        "Restored−identity: capability",
        "Random−identity: capability",
    ]
    fig, ax = plt.subplots(figsize=(6.5, 2.9), layout="constrained")
    for i, name in enumerate(names):
        r = result[name]
        point = 100 * r["estimate"]
        lo, hi = [100 * v for v in r["interval"]]
        ax.errorbar(
            point,
            i,
            xerr=[[max(0, point - lo)], [max(0, hi - point)]],
            fmt="o",
            capsize=3,
            color="#245A81" if i < 4 else "#69737C",
            markersize=4,
            linewidth=1.3,
        )
    ax.set_yticks(range(6), labels)
    ax.invert_yaxis()
    ax.axvline(0, color="#777777", linestyle="--", linewidth=0.8)
    ax.set_xlim(-13, 13)
    ax.set_xlabel("Paired difference (percentage points)")
    ax.grid(axis="x", alpha=0.18)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "figure-2-confirmation")
    groups = load(INPUTS[2])["groups"]
    attention = [
        100 * groups[f"model.layers.{i}.self_attn"]["relative_delta_l2"] for i in range(28)
    ]
    mlp = [100 * groups[f"model.layers.{i}.mlp"]["relative_delta_l2"] for i in range(28)]
    fig, ax = plt.subplots(figsize=(6.5, 2.55), layout="constrained")
    ax.plot(
        range(28),
        attention,
        color="#245A81",
        label="Attention projections and biases",
        linewidth=1.6,
    )
    ax.plot(range(28), mlp, color="#929DA6", label="MLP", linewidth=1.4)
    ax.scatter(
        [27], [attention[27]], color="#A44935", s=30, zorder=3, label="Selected attention group"
    )
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Relative weight drift (%)")
    ax.set_xticks([0, 5, 10, 15, 20, 25, 27])
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.16)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    save(fig, "figure-a1-weight-drift")
    manifest = {
        "input_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in INPUTS},
        "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "figures": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(DEST.iterdir())
            if p.suffix in {".png", ".pdf", ".svg"}
        },
    }
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Rebuilt three figures in PNG, PDF, and SVG with input hashes")


if __name__ == "__main__":
    main()
