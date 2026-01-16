import os
from matplotlib import pyplot as plt
from matplotlib_venn import venn2
import subprocess

folder = "test_predict/output/focal_loss_0.25/bedtools_results/"

# Paths
true_bed = os.path.join(folder,"true_noheader.bed")
pred_bed = os.path.join(folder, "pred_noheader.bed")
overlap_bed = os.path.join(folder, "True_in_Pred.bed")

# true_bed = os.path.join("test_predict/output/WT2_binary/WT3/true_noheader.bed")
# pred_bed = os.path.join("test_predict/output/WT2_binary/WT3/pred_noheader.bed")
# overlap_bed = os.path.join("test_predict/output/WT2_binary/WT3/True_in_Pred.bed")

# Count lines
def count_lines(path):
    with open(path) as f:
        return sum(1 for _ in f)

# Counts
true_total = count_lines(true_bed)
pred_total = count_lines(pred_bed)
overlap = count_lines(overlap_bed)

true_only = true_total - overlap
pred_only = pred_total - overlap

# Plot Venn
plt.figure(figsize=(5,5))
v = venn2(subsets=(true_only, pred_only, overlap),
          set_labels=("True", "Predicted"),
          set_colors=("lightgreen", "skyblue"),
          alpha=0.7)

# Add counts inside circles
v.get_label_by_id('10').set_text(str(true_only))
v.get_label_by_id('01').set_text(str(pred_only))
v.get_label_by_id('11').set_text(str(overlap))

plt.title("True vs Predicted Intervals")
plt.tight_layout()
plt.savefig(os.path.join(folder, "venn_counts.png"), dpi=300)
plt.show()

# Print summary
precision = overlap / pred_total if pred_total else 0
recall = overlap / true_total if true_total else 0
f1 = 2*precision*recall/(precision+recall) if (precision+recall) else 0

print(f"True total: {true_total}")
print(f"Predicted total: {pred_total}")
print(f"Overlap: {overlap}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-score: {f1:.3f}")
