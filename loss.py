import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryLoss(nn.Module):
    """Binary cross-entropy loss for base-resolution TFBS classification."""
    def __init__(self, use_logits=False, pos_weight=None):
        super(BinaryLoss, self).__init__()
        self.use_logits = use_logits

        if use_logits:
            # Use logits + BCEWithLogitsLoss for imbalanced data
            self.criteria = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            # Use standard BCE for balanced data (model outputs sigmoid probs)
            self.criteria = nn.BCELoss()

    def forward(self, prediction, target):
        prediction = prediction.view(-1)
        target = target.view(-1).float()
        return self.criteria(prediction, target)
        
# class FocalLoss(nn.Module):
    
#     def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
#         super().__init__()

#         if not (0 <= alpha <= 1) and alpha != -1:
#             raise ValueError(
#                 f"Invalid alpha value: {alpha}. alpha must be in [0,1] or -1."
#             )
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction

#     def forward(self, inputs, targets):
#         # Binary cross-entropy (per-element)
#         ce_loss = F.binary_cross_entropy_with_logits(
#             inputs, targets, reduction="none"
#         )

#         # p = sigmoid(logits)
#         p = torch.sigmoid(inputs)

#         # p_t = p if target == 1 else (1 - p)
#         p_t = p * targets + (1 - p) * (1 - targets)

#         # Focal modulation
#         loss = ce_loss * ((1 - p_t) ** self.gamma)

#         # Alpha balancing 
#         if self.alpha >= 0:
#             alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
#             loss = alpha_t * loss

#         # Reduction
#         if self.reduction == "mean":
#             loss = loss.mean()
#         elif self.reduction == "sum":
#             loss = loss.sum()
#         elif self.reduction == "none":
#             pass
#         else:
#             raise ValueError(
#                 f"Invalid reduction: {self.reduction}. "
#                 "Use 'none', 'mean', or 'sum'."
#             )

#         return loss




