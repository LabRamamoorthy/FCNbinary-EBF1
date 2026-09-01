import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Sample Data (RAW COUNTS) ──────────────────────────────────────────────────
datasets     = [ "pro-B", "3T3L1", "OP9"]
group_labels = ["Predicted Positive", "Predicted Negative"]
categories = ["Category 1", "Category 2", "Category 3"]

colors_group = [
    ["blue", "#CB6CE6", "#7ED957"],   # Group 1: 3 stacks
    ["blue", "#CB6CE6", "#A6A6A6"],   # Group 2: 3 stacks
]

raw_counts = np.array([
    [[1151, 13, 10438], [4, 1, 390]],
    [[1151, 2116, 5648], [4, 81, 2031]],
    [[1148, 2114, 5675], [5, 76, 413]],
])

# Convert counts → percentages
dataset_totals = raw_counts.sum(axis=(1, 2), keepdims=True)
data_pct = raw_counts / dataset_totals * 100
 
# ── Layout ────────────────────────────────────────────────────────────────────
bar_width   = 0.30
group_gap   = 0.08
dataset_gap = 0.90
 
fig, ax = plt.subplots(figsize=(12, 6)) 
x_tick_positions = []
 
for d_idx in range(len(datasets)):
    cluster_centre = d_idx * dataset_gap
 
    for g_idx in range(2):
        x      = cluster_centre + (g_idx - 0.5) * (bar_width + group_gap / 2)
        values = data_pct[d_idx, g_idx]
        clrs   = colors_group[g_idx]
 
        bottom = 0.0
        for val, clr in zip(values, clrs):
            ax.bar(x, val, bottom=bottom, width=bar_width,
                   color=clr, edgecolor="white", linewidth=0.6)
            if val >= 7:
                ax.text(x, bottom + val / 2, f"{val:.1f}%",
                        ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")
            bottom += val
 
        ax.text(x, -4, group_labels[g_idx],
                ha="center", va="top", fontsize=7.5, color="#555555")
 
    x_tick_positions.append(cluster_centre)
 
# ── Axes Styling ──────────────────────────────────────────────────────────────
ax.set_ylim(-8, 108)
ax.set_ylabel("Percentage (%)", fontsize=11, labelpad=8)
ax.set_xticks(x_tick_positions)
ax.set_xticklabels(datasets, fontsize=10)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{int(y)}%"))
ax.tick_params(axis="y", labelsize=9)
for spine in ["top", "right", "bottom"]:
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_color("#CCCCCC")
ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.7, linestyle="--")
ax.set_axisbelow(True)
 
# ── Legend — one entry per group+category combo ───────────────────────────────
patches = []
for g_idx, g_label in enumerate(group_labels):
    for c_idx, cat in enumerate(categories):
        patches.append(mpatches.Patch(
            color=colors_group[g_idx][c_idx],
            label=f"{g_label} – {cat}"
        ))
ax.legend(handles=patches, loc="upper right", fontsize=8,
          framealpha=0.85, ncol=2, title="Group – Category", title_fontsize=8)
 
ax.set_title("Grouped Stacked Bar Chart – Percentage Breakdown by Dataset",
             fontsize=13, fontweight="bold", pad=14, color="#222222")
 
plt.tight_layout()
plt.savefig("grouped_stacked_bar_bfcn.pdf", dpi=150, bbox_inches="tight")
plt.show()
print("Chart saved.")