import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryLoss(nn.Module):
    """Binary cross-entropy loss for base-resolution TFBS classification."""
    def __init__(self, use_logits=False, pos_weight=None):
        super(BinaryLoss, self).__init__()
        self.use_logits = use_logits
        if use_logits:
            self.criteria = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.criteria = nn.BCELoss()

    def forward(self, prediction, target):
        prediction = prediction.view(-1)
        target     = target.view(-1).float()
        return self.criteria(prediction, target)


class SeqLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight
        self.criteria   = None

    def forward(self, logits_seq, seq_labels):
        if self.criteria is None:
            pw = torch.tensor([self.pos_weight],
                              dtype=torch.float32,
                              device=seq_labels.device) \
                if self.pos_weight is not None else None
            self.criteria = nn.BCEWithLogitsLoss(pos_weight=pw)
        return self.criteria(logits_seq.squeeze(-1), seq_labels.float())


class TokenLoss(nn.Module):
    def __init__(self, bert_pos_weight=None, fcn_pos_weight=None,
                 alpha=0.5, beta=0.5):
        super().__init__()
        self.bert_pos_weight = bert_pos_weight
        self.fcn_pos_weight  = fcn_pos_weight
        self.alpha           = alpha
        self.beta            = beta
        self.criteria_bert   = None
        self.criteria_fcn    = None

    def forward(self, fused_logits, bert_labels, fcn_signal):
        if self.criteria_bert is None:
            pw = torch.tensor([self.bert_pos_weight], dtype=torch.float32,
                              device=bert_labels.device) \
                if self.bert_pos_weight is not None else None
            self.criteria_bert = nn.BCEWithLogitsLoss(pos_weight=pw)

        if self.criteria_fcn is None:
            pw = torch.tensor([self.fcn_pos_weight], dtype=torch.float32,
                              device=bert_labels.device) \
                if self.fcn_pos_weight is not None else None
            self.criteria_fcn = nn.BCEWithLogitsLoss(pos_weight=pw)

        total = torch.tensor(0.0, device=bert_labels.device, requires_grad=True)
        for b, logits in enumerate(fused_logits):
            L    = logits.size(0)
            pred = logits.squeeze(-1)

            loss_bert = self.criteria_bert(pred, bert_labels[b, :L].float())
            loss_fcn  = self.criteria_fcn(pred,  fcn_signal[b,  :L].float())

            total = total + self.alpha * loss_bert + self.beta * loss_fcn

        return total / len(fused_logits)


class FusedLoss(nn.Module):
    """
    total = (1 - t_seq) * token_loss
          +  t_seq       * seq_loss

    token_loss = alpha * BCE(fused_logits, bert_labels)   # BERT supervision
               + beta  * BCE(fused_logits, fcn_signal)    # FCN supervision

    t_seq must be < 1.0 so token_loss gets positive weight.
    alpha + beta must sum to 1.0.
    """
    def __init__(self, t_seq=0.2,
                 bert_pos_weight=None,
                 fcn_pos_weight=None,
                 seq_pos_weight=None,
                 alpha=0.5, beta=0.5):
        super().__init__()
        assert t_seq < 1.0, \
            "t_seq must be < 1.0 so token_loss gets positive weight"
        assert abs(alpha + beta - 1.0) < 1e-6, \
            "alpha + beta must sum to 1.0"

        self.t_seq      = t_seq
        self.token_loss = TokenLoss(bert_pos_weight=bert_pos_weight,
                                    fcn_pos_weight=fcn_pos_weight,
                                    alpha=alpha, beta=beta)
        self.seq_loss   = SeqLoss(pos_weight=seq_pos_weight)

    def forward(self, fused_logits, logits_seq,
                bert_labels, seq_labels, fcn_signal):

        t_loss = self.token_loss(fused_logits, bert_labels, fcn_signal)
        s_loss = self.seq_loss(logits_seq,     seq_labels)

        total = (
            (1 - self.t_seq) * t_loss +
             self.t_seq       * s_loss
        )
        return total, {
            'token' : t_loss.item(),
            'seq'   : s_loss.item(),
            'total' : total.item(),
        }