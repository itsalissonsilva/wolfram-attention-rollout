"""Recompute the maps independently and render the figures used in the post."""
import hashlib
import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from analyze_heads_layers import analyze

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--image", choices=('samoyed', 'samoyed-option-4', 'samoyed-option-5', 'samoyed-option-1'), default="samoyed")
parser.add_argument("--output", type=Path, help="Output directory; defaults to generated/<image>.")
args = parser.parse_args()
SOURCE = ROOT / f"results/{args.image}-native-001"
OUT = args.output or ROOT / "generated" / args.image
OUT.mkdir(parents=True, exist_ok=True)
run = json.loads((SOURCE / "result.json").read_text(encoding="utf-8-sig"))
assert run["Success"]
path = SOURCE / "attention.f32"
assert hashlib.file_digest(path.open("rb"), "sha256").hexdigest() == run["AttentionSHA256"]
attention = np.fromfile(path, "<f4").reshape(run["AttentionShape"]).astype(float)
independent = analyze(attention, 16)
data = json.loads((SOURCE / "analysis.json").read_text(encoding="utf-8-sig"))
errors = {}
for key in ("HeadPatchMaps", "MeanPatchMaps", "RolloutPatchMaps", "PreviousCLSFootprint"):
    errors[key] = float(np.max(np.abs(np.asarray(data[key]) - independent[key])))
for key in ("HeadEntropies", "HeadPatchMasses", "SpatialHeadMixtureWeights",
            "WeightedHeadEntropy", "MeanHeadEntropy", "HeadJensenShannon",
            "RolloutEntropy", "RolloutPatchMass", "MaxRowTVFromCLS", "MeanRowTVFromCLS"):
    errors[key] = float(np.max(np.abs(np.asarray([x[key] for x in data["LayerMetrics"]]) -
                                    [x[key] for x in independent["LayerMetrics"]])))
assert max(errors.values()) < 1e-12
crop = np.asarray(Image.open(SOURCE / "model-crop.png").convert("RGB")) / 255
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 11})

def overlay(ax, values, title):
    values = np.asarray(values)
    scaled = (values - values.min()) / np.ptp(values) if np.ptp(values) else np.zeros_like(values)
    large = np.asarray(Image.fromarray(scaled.astype("float32")).resize(
        (224, 224), Image.Resampling.BILINEAR))
    ax.imshow(.55 * crop + .45 * plt.get_cmap("inferno")(large)[..., :3])
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_axis_off()

fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.8))
overlay(axes[0], data["MeanPatchMaps"][-1], "Attention heads\nMean of the final layer")
overlay(axes[1], data["RolloutPatchMaps"][-1], "Attention rollout\nAccumulated through all 12 layers")
fig.subplots_adjust(left=.025, right=.975, top=.84, bottom=.025, wspace=.07)
fig.savefig(OUT / "comparison.png", dpi=140, facecolor="white")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8.0, 3.8))
layers = np.arange(1, 13)
head_h = np.asarray([x["HeadEntropies"] for x in data["LayerMetrics"]])
ax.fill_between(layers, head_h.min(axis=1), head_h.max(axis=1),
                color="#b4d4e5", alpha=.6, label="Range across individual heads")
for key, label, color in (
    ("MeanHeadEntropy", "Mean-head map", "#356ba0"),
    ("RolloutEntropy", "Attention rollout", "#c66a34")
):
    ax.plot(layers, [m[key] for m in data["LayerMetrics"]],
            "-o", label=label, color=color, markersize=4, linewidth=2)
ax.set(xlim=(1, 12), ylim=(0, 1.04), xticks=layers, xlabel="Layer",
       ylabel="Normalized spatial entropy\n1 = uniform across patches")
ax.grid(alpha=.17)
ax.legend(loc="lower left", frameon=False, fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "layers.png", dpi=140, facecolor="white")
plt.close(fig)

# Inspection aid only; this additional pair is not embedded in the post.
selected = [int(head_h[-1].argmin()), int(head_h[-1].argmax())]
fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.8))
for ax, h, label in zip(axes, selected, ("Most concentrated", "Least concentrated")):
    overlay(ax, data["HeadPatchMaps"][-1][h], f"Head {h+1}: {label}\nH = {head_h[-1,h]:.3f}")
fig.subplots_adjust(left=.025, right=.975, top=.84, bottom=.025, wspace=.07)
fig.savefig(OUT / "head-inspection.png", dpi=125, facecolor="white")
plt.close(fig)

summary = {"Success": True, "Model": "B16", "Image": args.image, "Top3": data["Top3"],
           "NativeNumPyMaximumError": max(errors.values()), "Errors": errors,
           "MostConcentratedHead": selected[0]+1, "LeastConcentratedHead": selected[1]+1,
           "LastLayer": data["LayerMetrics"][-1],
           "FinalMixture": {k:v for k,v in data["FinalMixture"].items() if k != "SpatialMixtureWeights"}}
(OUT / "result.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({k:v for k,v in summary.items() if k not in ("Errors","LastLayer")}, indent=2))
