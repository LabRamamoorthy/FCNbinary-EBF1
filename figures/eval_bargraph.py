# import numpy as np
# import matplotlib.pyplot as plt

# plt.rcParams["font.family"] = "serif"

# models = [
#     'FCNBinary',
#     'BertSNR2',
#     'Fused Model',
#     'FCNBinary + ATAC',
#     'Fused Model + ATAC'
# ]

# parameters = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'AUPR']

# Shape: (models, parameters)
# data = np.array([
#     [0.848, 0.849, 0.857, 0.853, 0.918, 0.928],
#     [0.946, 0.935, 0.959, 0.947, 0.984, 0.982],
#     [0.908, 0.914, 0.901, 0.907, 0.968, 0.964],
#     [0.938, 0.957, 0.921, 0.938, 0.982, 0.986],
#     [0.946, 0.944, 0.946, 0.986, 0.984, 0.984]
# ])
# # data = np.array([
# #     [0.863 , 0.723, 0.791, 0.756, 0.9208, 0.828],
# #     [0.994, 0.728, 0.615, 0.667, 0.993, 0.740],
# #     [0.822, 0.453, 0.897, 0.452, 0.981, 0.862], 
# #     [0.90, 0.795, 0.845, 0.819, 0.9587, 0.900],
# #     [0.839, 0.944, 0.949, 0.946, 0.9906, 0.906]
# # ])

# # Manual thesis-style colors
# colors = [
#     "#4C72B0",  # muted blue
#     "#DD8452",  # soft orange
#     "#55A868",  # green
#     "#C44E52",  # red
#     "#8172B2"   # purple  
# ]

# n_models = len(models)
# n_params = len(parameters)

# x = np.arange(n_params)
# width = 0.15

# fig, ax = plt.subplots(figsize=(10, 6))

# for i in range(n_models):
#     offsets = x + (i - n_models/2) * width + width/2

#     ax.bar(
#         offsets,
#         data[i],
#         width,
#         label=models[i],
#         color=colors[i]
#     )

# # Clean axis style (thesis/paper friendly)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# # Labels
# ax.set_xticks(x)
# ax.set_xticklabels(parameters)
# ax.set_ylabel('Score')
# ax.set_title('Model Comparison Across Evaluation Metrics (base Level)', fontsize=14)

# # Legend outside plot
# ax.legend(
#     loc='center left',
#     bbox_to_anchor=(1, 0.5),
#     frameon=False
# )

# plt.tight_layout()
# plt.subplots_adjust(right=0.8)
# plt.grid(True)
# # Save high-quality outputs
# plt.savefig("eval_bar_seq_run2.png", dpi=300, bbox_inches="tight")
# plt.savefig("eval_bar_seq_run2.pdf", bbox_inches="tight")

# plt.show()

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"

models = [
    '500',
    '1000',
    '2000',
    '5000'
]

parameters = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'AUPR']

# Shape: (models, parameters)
# different training length
data = np.array([
    [0.938, 0.957, 0.921, 0.938, 0.982, 0.986],
    [0.948, 0.951, 0.949, 0.950, 0.982, 0.987],
    [0.906, 0.905, 0.938, 0.921, 0.966, 0.979],
    [0.882, 0.889, 0.927, 0.908, 0.952, 0.974]
])

# data = np.array([
#     [0.8998, 0.7951, 0.8450, 0.8193, 0.9587],
#     [0.9231, 0.7651, 0.8697, 0.8140, 0.9714],
#     [0.9386, 0.6874, 0.8299, 0.7519, 0.9698],
#     [0.9529, 0.5421, 0.8609, 0.6653, 0.9763]
# ])

# Manual thesis-style colors
colors = [
    "#4C72B0",  # muted blue
    "#DD8452",  # soft orange
    "#55A868",  # green
    "#C44E52",  # red
]

n_models = len(models)
n_params = len(parameters)

x = np.arange(n_params)
width = 0.15

fig, ax = plt.subplots(figsize=(10, 6))

for i in range(n_models):
    offsets = x + (i - n_models/2) * width + width/2

    ax.bar(
        offsets,
        data[i],
        width,
        label=models[i],
        color=colors[i]
    )

# Clean axis style (thesis/paper friendly)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Labels
ax.set_xticks(x)
ax.set_xticklabels(parameters)
ax.set_ylabel('Score')
ax.set_title('FCNBinary + ATAC : Comparing Models With Different Training Sequence Length', fontsize=14)

# Legend outside plot
ax.legend(
    loc='center left',
    bbox_to_anchor=(1, 0.5),
    frameon=False
)

plt.tight_layout()
plt.subplots_adjust(right=0.8)
plt.grid(True)
# Save high-quality outputs
plt.savefig("train_seq.png", dpi=300, bbox_inches="tight")
plt.savefig("train_seq.pdf", bbox_inches="tight")

plt.show()

