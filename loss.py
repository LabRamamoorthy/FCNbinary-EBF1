import torch
import torch.nn as nn

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