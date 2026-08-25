# import os
# import os.path as osp
# from copy import deepcopy

# import numpy as np
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# from tqdm import tqdm
# from loss import FusedLoss

# plt.rcParams["font.family"] = "serif"


# # EARLY STOPPING

# class EarlyStopping:

#     def __init__(self, patience=5, min_delta=1e-4, verbose=True):
#         self.patience   = patience
#         self.min_delta  = min_delta
#         self.verbose    = verbose
#         self.best_loss  = None
#         self.best_epoch = 0
#         self._counter   = 0

#     def __call__(self, val_loss, epoch):
#         """Returns True when training should stop."""
#         if self.best_loss is None:
#             self.best_loss  = val_loss
#             self.best_epoch = epoch
#             if self.verbose:
#                 print(f"  EarlyStopping: initial val_loss={val_loss:.6f}")

#         elif val_loss < self.best_loss - self.min_delta:
#             if self.verbose:
#                 print(f"  EarlyStopping: improved {self.best_loss:.6f} → {val_loss:.6f}")
#             self.best_loss  = val_loss
#             self.best_epoch = epoch
#             self._counter   = 0

#         else:
#             self._counter += 1
#             if self.verbose:
#                 print(f"  EarlyStopping: no improvement "
#                       f"({self._counter}/{self.patience})")
#             if self._counter >= self.patience:
#                 if self.verbose:
#                     print(f"  EarlyStopping triggered! "
#                           f"Best epoch={self.best_epoch}, "
#                           f"best_loss={self.best_loss:.6f}")
#                 return True

#         return False


# # TRAINER

# class Trainer:

#     def __init__(
#         self,
#         model,
#         train_loader,
#         val_loader,
#         optimizer,
#         loss_fn,
#         device,
#         checkpoint_dir   = "./checkpoints",
#         scheduler        = None,
#         max_epoch        = 20,
#         threshold        = 0.5,
#         # unfreeze_bert_at = 3,
#         clip_grad_norm   = 1.0,
#         early_stopping   = None,
#     ):
#         self.model            = model.to(device)
#         self.train_loader     = train_loader
#         self.val_loader       = val_loader
#         self.optimizer        = optimizer
#         self.loss_fn          = loss_fn
#         self.device           = device
#         self.scheduler        = scheduler
#         self.max_epoch        = max_epoch
#         self.threshold        = threshold
#         # self.unfreeze_bert_at = unfreeze_bert_at
#         self.clip_grad_norm   = clip_grad_norm
#         self.early_stopping   = early_stopping
#         self.checkpoint_dir   = checkpoint_dir

#         os.makedirs(checkpoint_dir, exist_ok=True)

#         # state
#         self.epoch         = 0
#         self.best_val_loss = float('inf')
#         self.state_best    = None

#         self.train_losses = []
#         self.val_losses   = []

#     # INTERNAL HELPERS

#     def _to_device(self, batch):
#         return {k: v.to(self.device) for k, v in batch.items()}

#     def _step(self, batch):
#         logits_seq, fused_logits = self.model(
#             fcn_input      = batch['one_hot'],
#             input_ids      = batch['input_ids'],
#             attention_mask = batch['attention_mask'],
#             offset_mapping = batch['offset_mapping'],
#         )

#         total, breakdown = self.loss_fn(
#             fused_logits = fused_logits,
#             logits_seq   = logits_seq,
#             bert_labels  = batch['bert_labels'],
#             seq_labels   = batch['seq_label'],
#             fcn_signal   = batch['fcn_signal'],
#         )

#         return total, breakdown

#     # CLASS-WEIGHT HELPER

#     def compute_pos_weights(self):
#         pos_bert = neg_bert = 0   # for bert_pos_weight (bert_labels)
#         pos_fcn  = neg_fcn  = 0   # for fcn_pos_weight  (fcn_signal)
#         pos_seq  = neg_seq  = 0   # for seq_pos_weight  (seq_label)

#         print("Computing class distributions from training data...")
#         for batch in self.train_loader:
#             bert = batch['bert_labels'].numpy()
#             pos_bert += int((bert == 1).sum())
#             neg_bert += int((bert == 0).sum())

#             fcn = batch['fcn_signal'].numpy()
#             pos_fcn += int((fcn == 1).sum())
#             neg_fcn += int((fcn == 0).sum())

#             seq = batch['seq_label'].numpy()
#             pos_seq += int((seq == 1).sum())
#             neg_seq += int((seq == 0).sum())

#         def _pw(pos, neg, name):
#             if pos == 0 or neg == 0:
#                 print(f"  Warning: only one class in {name}. Using pos_weight=1.0")
#                 return 1.0
#             pw = neg / pos
#             print(f"  {name:20s} pos={pos:,}  neg={neg:,}  pos_weight={pw:.3f}")
#             return pw

#         bert_pw = _pw(pos_bert, neg_bert, "bert_labels (bert)")
#         fcn_pw  = _pw(pos_fcn,  neg_fcn,  "fcn_signal  (fcn)")
#         seq_pw  = _pw(pos_seq,  neg_seq,  "seq_label   (seq)")

#         return (
#             torch.tensor(bert_pw, dtype=torch.float32).to(self.device),
#             torch.tensor(fcn_pw,  dtype=torch.float32).to(self.device),
#             torch.tensor(seq_pw,  dtype=torch.float32).to(self.device),
#         )

#     # TRAIN ONE EPOCH

#     def _train_epoch(self):
#         self.model.train()
#         agg      = {'token': 0.0, 'seq': 0.0, 'total': 0.0}
#         progress = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")

#         for batch in progress:
#             batch = self._to_device(batch)
#             self.optimizer.zero_grad()

#             total, breakdown = self._step(batch)

#             if not torch.isfinite(total):
#                 print("  [warn] non-finite loss — skipping batch")
#                 continue

#             total.backward()
#             if self.clip_grad_norm:
#                 nn.utils.clip_grad_norm_(self.model.parameters(),
#                                          self.clip_grad_norm)
#             self.optimizer.step()

#             if self.scheduler is not None:
#                 self.scheduler.step()

#             for k in agg:
#                 agg[k] += breakdown[k]

#             progress.set_description(
#                 f"Epoch {self.epoch} | "
#                 f"loss={breakdown['total']:.4f} "
#                 f"(token={breakdown['token']:.4f}, "
#                 f"seq={breakdown['seq']:.4f})"
#             )

#         n   = len(self.train_loader)
#         avg = {k: v / n for k, v in agg.items()}
#         self.train_losses.append(avg['total'])
#         return avg

#     # VALIDATE ONE EPOCH

#     @torch.no_grad()
#     def _val_epoch(self):
#         self.model.eval()
#         agg      = {'token': 0.0, 'seq': 0.0, 'total': 0.0}
#         progress = tqdm(self.val_loader, desc="Validating")

#         for batch in progress:
#             batch = self._to_device(batch)
#             _, breakdown = self._step(batch)

#             for k in agg:
#                 agg[k] += breakdown[k]

#         n   = len(self.val_loader)
#         avg = {k: v / n for k, v in agg.items()}
#         self.val_losses.append(avg['total'])

#         print(f"  [val] loss_total={avg['total']:.4f}  "
#               f"token={avg['token']:.4f}  seq={avg['seq']:.4f}")

#         # save best state by val_loss
#         if avg['total'] < self.best_val_loss:
#             self.best_val_loss = avg['total']
#             self.state_best    = deepcopy(self.model.state_dict())
#             self._save_checkpoint(self.epoch, avg['total'])

#         return avg

#     # CHECKPOINT

#     def _save_checkpoint(self, epoch, val_loss):
#         path = osp.join(self.checkpoint_dir, "best_model.pt")
#         torch.save({
#             'epoch'    : epoch,
#             'val_loss' : val_loss,
#             'model'    : self.model.state_dict(),
#             'optimizer': self.optimizer.state_dict(),
#         }, path)
#         print(f"  checkpoint saved  (val_loss={val_loss:.4f})")

#     def load_best_checkpoint(self):
#         path = osp.join(self.checkpoint_dir, "best_model.pt")
#         ckpt = torch.load(path, map_location=self.device)
#         self.model.load_state_dict(ckpt['model'])
#         print(f"  loaded checkpoint from epoch {ckpt['epoch']}")

#     # FIT

#     def fit(self):
#         early_stopped = False

#         for epoch in range(1, self.max_epoch + 1):
#             self.epoch = epoch

#             # unfreeze BERT after warm-up epochs
#             # if epoch == self.unfreeze_bert_at + 1:
#             #     print(f"\n[Epoch {epoch}] Unfreezing BERT with reduced LR...")
#             #     for p in self.model.bert.parameters():
#             #         p.requires_grad = True
#             #     for pg in self.optimizer.param_groups[:2]:   # first two groups are BERT
#             #         pg['lr'] = pg['lr'] * 0.1
#             #         print(f"  BERT lr reduced to {pg['lr']:.2e}")

#             print(f"\n{'═' * 60}")
#             print(f"Epoch {epoch} / {self.max_epoch}")
#             print(f"{'═' * 60}")

#             self._train_epoch()
#             val_avg = self._val_epoch()

#             if self.early_stopping is not None:
#                 if self.early_stopping(val_avg['total'], epoch):
#                     early_stopped = True
#                     break

#         print("\nTraining finished. Plotting loss curve...")
#         self.plot_results()

#         status = f"stopped early at epoch {epoch}" if early_stopped \
#                  else f"completed {self.max_epoch} epochs"
#         print(f"Training {status}  |  best_val_loss={self.best_val_loss:.4f}")

#         return self.best_val_loss, self.state_best

#     # PLOTTING

#     def plot_results(self):
#         cp = self.checkpoint_dir

#         plt.figure(figsize=(8, 5))
#         plt.plot(self.train_losses, label="Train loss")
#         plt.plot(self.val_losses,   label="Val loss", linewidth=2)
#         plt.xlabel("Epoch")
#         plt.ylabel("Loss")
#         plt.title("Training and Validation Loss")
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(osp.join(cp, "loss_curve.png"))
#         plt.close()
#         print(f"Saved loss curve → {osp.join(cp, 'loss_curve.png')}")


import os
import os.path as osp
from copy import deepcopy

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    average_precision_score, precision_recall_curve,
)
from loss import FusedLoss

plt.rcParams["font.family"] = "serif"


# ─────────────────────────────────────────────
# EARLY STOPPING
# ─────────────────────────────────────────────

class EarlyStopping:

    def __init__(self, patience=5, min_delta=1e-4, verbose=True):
        self.patience   = patience
        self.min_delta  = min_delta
        self.verbose    = verbose
        self.best_loss  = None
        self.best_epoch = 0
        self._counter   = 0

    def __call__(self, val_loss, epoch):
        """Returns True when training should stop."""
        if self.best_loss is None:
            self.best_loss  = val_loss
            self.best_epoch = epoch
            if self.verbose:
                print(f"  EarlyStopping: initial val_loss={val_loss:.6f}")

        elif val_loss < self.best_loss - self.min_delta:
            if self.verbose:
                print(f"  EarlyStopping: improved {self.best_loss:.6f} → {val_loss:.6f}")
            self.best_loss  = val_loss
            self.best_epoch = epoch
            self._counter   = 0

        else:
            self._counter += 1
            if self.verbose:
                print(f"  EarlyStopping: no improvement "
                      f"({self._counter}/{self.patience})")
            if self._counter >= self.patience:
                if self.verbose:
                    print(f"  EarlyStopping triggered! "
                          f"Best epoch={self.best_epoch}, "
                          f"best_loss={self.best_loss:.6f}")
                return True

        return False


# ─────────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────────

class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        device,
        checkpoint_dir   = "./checkpoints",
        scheduler        = None,
        max_epoch        = 20,
        threshold        = 0.5,
        clip_grad_norm   = 1.0,
        early_stopping   = None,
    ):
        self.model            = model.to(device)
        self.train_loader     = train_loader
        self.val_loader       = val_loader
        self.optimizer        = optimizer
        self.loss_fn          = loss_fn
        self.device           = device
        self.scheduler        = scheduler
        self.max_epoch        = max_epoch
        self.threshold        = threshold
        self.clip_grad_norm   = clip_grad_norm
        self.early_stopping   = early_stopping
        self.checkpoint_dir   = checkpoint_dir

        os.makedirs(checkpoint_dir, exist_ok=True)

        # scalar tracking
        self.epoch         = 0
        self.best_val_loss = float('inf')
        self.state_best    = None
        self.train_losses  = []
        self.val_losses    = []

        # prediction stores (filled in collect_final_predictions)
        self.val_token_probs = []   # fused token head probs (single set)
        self.val_bert_labels = []   # ground truth: bert_labels
        self.val_fcn_labels  = []   # ground truth: fcn_signal
        self.val_seq_probs   = []
        self.val_seq_labels  = []

    # ── internal helpers ──────────────────────

    def _to_device(self, batch):
        return {k: v.to(self.device) for k, v in batch.items()}

    def _step(self, batch):
        logits_seq, fused_logits = self.model(
            fcn_input      = batch['one_hot'],
            input_ids      = batch['input_ids'],
            attention_mask = batch['attention_mask'],
            offset_mapping = batch['offset_mapping'],
        )

        total, breakdown = self.loss_fn(
            fused_logits = fused_logits,
            logits_seq   = logits_seq,
            bert_labels  = batch['bert_labels'],
            seq_labels   = batch['seq_label'],
            fcn_signal   = batch['fcn_signal'],
        )

        return total, breakdown, logits_seq, fused_logits

    # ── class-weight helper ───────────────────

    def compute_pos_weights(self):
        pos_bert = neg_bert = 0
        pos_fcn  = neg_fcn  = 0
        pos_seq  = neg_seq  = 0

        print("Computing class distributions from training data...")
        for batch in self.train_loader:
            bert = batch['bert_labels'].numpy()
            pos_bert += int((bert == 1).sum())
            neg_bert += int((bert == 0).sum())

            fcn = batch['fcn_signal'].numpy()
            pos_fcn += int((fcn == 1).sum())
            neg_fcn += int((fcn == 0).sum())

            seq = batch['seq_label'].numpy()
            pos_seq += int((seq == 1).sum())
            neg_seq += int((seq == 0).sum())

        def _pw(pos, neg, name):
            if pos == 0 or neg == 0:
                print(f"  Warning: only one class in {name}. Using pos_weight=1.0")
                return 1.0
            pw = neg / pos
            print(f"  {name:20s} pos={pos:,}  neg={neg:,}  pos_weight={pw:.3f}")
            return pw

        bert_pw = _pw(pos_bert, neg_bert, "bert_labels (bert)")
        fcn_pw  = _pw(pos_fcn,  neg_fcn,  "fcn_signal  (fcn)")
        seq_pw  = _pw(pos_seq,  neg_seq,  "seq_label   (seq)")

        return (
            torch.tensor(bert_pw, dtype=torch.float32).to(self.device),
            torch.tensor(fcn_pw,  dtype=torch.float32).to(self.device),
            torch.tensor(seq_pw,  dtype=torch.float32).to(self.device),
        )

    # ── train one epoch ───────────────────────

    def _train_epoch(self):
        self.model.train()
        agg      = {'token': 0.0, 'seq': 0.0, 'total': 0.0}
        progress = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")

        for batch in progress:
            batch = self._to_device(batch)
            self.optimizer.zero_grad()

            total, breakdown, _, _ = self._step(batch)

            if not torch.isfinite(total):
                print("  [warn] non-finite loss — skipping batch")
                continue

            total.backward()
            if self.clip_grad_norm:
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.clip_grad_norm)
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            for k in agg:
                agg[k] += breakdown[k]

            progress.set_description(
                f"Epoch {self.epoch} | "
                f"loss={breakdown['total']:.4f} "
                f"(token={breakdown['token']:.4f}, "
                f"seq={breakdown['seq']:.4f})"
            )

        n   = len(self.train_loader)
        avg = {k: v / n for k, v in agg.items()}
        self.train_losses.append(avg['total'])
        return avg

    # ── validate one epoch ────────────────────

    @torch.no_grad()
    def _val_epoch(self):
        self.model.eval()
        agg      = {'token': 0.0, 'seq': 0.0, 'total': 0.0}
        progress = tqdm(self.val_loader, desc="Validating")

        for batch in progress:
            batch = self._to_device(batch)
            _, breakdown, _, _ = self._step(batch)

            for k in agg:
                agg[k] += breakdown[k]

        n   = len(self.val_loader)
        avg = {k: v / n for k, v in agg.items()}
        self.val_losses.append(avg['total'])

        print(f"  [val] loss_total={avg['total']:.4f}  "
              f"token={avg['token']:.4f}  seq={avg['seq']:.4f}")

        if avg['total'] < self.best_val_loss:
            self.best_val_loss = avg['total']
            self.state_best    = deepcopy(self.model.state_dict())
            self._save_checkpoint(self.epoch, avg['total'])

        return avg

    # ── checkpoint ────────────────────────────

    def _save_checkpoint(self, epoch, val_loss):
        path = osp.join(self.checkpoint_dir, "best_model.pt")
        torch.save({
            'epoch'    : epoch,
            'val_loss' : val_loss,
            'model'    : self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
        print(f"  checkpoint saved  (val_loss={val_loss:.4f})")

    def load_best_checkpoint(self):
        path = osp.join(self.checkpoint_dir, "best_model.pt")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        print(f"  loaded checkpoint from epoch {ckpt['epoch']}")

    # ── collect final predictions ─────────────

    @torch.no_grad()
    def collect_final_predictions(self):
        """
        Run the validation loader one final time and store per-sample arrays.

        Model returns:
          logits_seq   : [B, 1]              sequence head (CLS token)
          fused_logits : list of [L_i, 1]   token head, one tensor per sample
                         (variable length because sequences differ in length)

        The single fused token head is compared against BOTH bert_labels and
        fcn_signal to produce two per-base CMs whose average is reported.

        All stored arrays are 1-D numpy arrays (flattened per sample so that
        variable-length sequences can be concatenated later).
        """
        print("\nCollecting final validation predictions...")
        self.model.eval()

        # fused token head probs (one set, compared against two label sources)
        self.val_token_probs = []   # list of 1-D float arrays
        self.val_bert_labels = []   # ground truth: bert_labels
        self.val_fcn_labels  = []   # ground truth: fcn_signal

        # sequence head
        self.val_seq_probs   = []   # list of scalars / 1-D arrays
        self.val_seq_labels  = []

        for batch in tqdm(self.val_loader, desc="Collecting preds"):
            batch = self._to_device(batch)

            # logits_seq   : [B, 1]
            # fused_logits : list of B tensors, each [L_i, 1]
            logits_seq, fused_logits = self.model(
                fcn_input      = batch['one_hot'],
                input_ids      = batch['input_ids'],
                attention_mask = batch['attention_mask'],
                offset_mapping = batch['offset_mapping'],
            )

            # ── sequence head ──────────────────────────────────────────────
            # logits_seq is [B, 1]; squeeze to [B]
            seq_probs  = torch.sigmoid(logits_seq).squeeze(-1).cpu().numpy()
            seq_labels = batch['seq_label'].cpu().numpy()            # [B]
            self.val_seq_probs.append(seq_probs)
            self.val_seq_labels.append(seq_labels)

            # ── token (fused) head — per sample ───────────────────────────
            # bert_labels : [B, max_len]  (padded to tokeniser max_len)
            # fcn_signal  : [B, L]        (original sequence length)
            # fused_logits[i] : [L_i, 1]  (actual nucleotide length after
            #                              BERT alignment, ≤ min(max_len, L))
            bert_labels_batch = batch['bert_labels'].cpu().numpy()   # [B, max_len]
            fcn_labels_batch  = batch['fcn_signal'].cpu().numpy()    # [B, L]

            for i, logit_i in enumerate(fused_logits):
                # logit_i : [L_i, 1]
                L_i   = logit_i.size(0)
                probs = torch.sigmoid(logit_i).squeeze(-1).cpu().numpy()  # [L_i]

                # Align labels to exactly L_i positions
                bert_lbl = bert_labels_batch[i, :L_i]   # [L_i]
                fcn_lbl  = fcn_labels_batch[i,  :L_i]   # [L_i]

                self.val_token_probs.append(probs)
                self.val_bert_labels.append(bert_lbl)
                self.val_fcn_labels.append(fcn_lbl)

    # ── fit ───────────────────────────────────

    def fit(self):
        early_stopped = False

        for epoch in range(1, self.max_epoch + 1):
            self.epoch = epoch

            print(f"\n{'═' * 60}")
            print(f"Epoch {epoch} / {self.max_epoch}")
            print(f"{'═' * 60}")

            self._train_epoch()
            val_avg = self._val_epoch()

            if self.early_stopping is not None:
                if self.early_stopping(val_avg['total'], epoch):
                    early_stopped = True
                    break

        print("\nTraining finished. Generating final outputs...")
        self.collect_final_predictions()
        self.plot_results()
        self.save_val_outputs()

        status = f"stopped early at epoch {epoch}" if early_stopped \
                 else f"completed {self.max_epoch} epochs"
        print(f"Training {status}  |  best_val_loss={self.best_val_loss:.4f}")

        return self.best_val_loss, self.state_best

    # ── plotting helpers ──────────────────────

    @staticmethod
    def _cm_metrics(y_true, y_pred, y_prob=None):
        """Return dict of scalar classification metrics."""
        return {
            'acc' : accuracy_score(y_true, y_pred),
            'prec': precision_score(y_true, y_pred, zero_division=0),
            'rec' : recall_score(y_true, y_pred, zero_division=0),
            'f1'  : f1_score(y_true, y_pred, zero_division=0),
            'auc' : roc_auc_score(y_true, y_prob) if y_prob is not None
                    and len(np.unique(y_true)) > 1 else np.nan,
        }

    def _save_cm(self, cm, title, fname_stem, fmt='d'):
        """Save a ConfusionMatrixDisplay as PNG and PDF."""
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues", values_format=fmt)
        plt.title(title)
        plt.tight_layout()
        for ext in ('png', 'pdf'):
            plt.savefig(osp.join(self.checkpoint_dir, f"{fname_stem}.{ext}"))
        plt.close()

    def _averaged_per_base_cm(self):
        """
        Compute the averaged per-base confusion matrix.
        The single fused token head's probabilities are compared against
        two different ground truths (bert_labels and fcn_signal) to produce
        two CMs, which are then row-normalised and averaged.
        """
        token_prob = np.concatenate(self.val_token_probs).flatten()
        token_pred = (token_prob >= self.threshold).astype(int)

        bert_true = np.concatenate(self.val_bert_labels).flatten()
        fcn_true  = np.concatenate(self.val_fcn_labels).flatten()

        cm_bert = confusion_matrix(bert_true, token_pred).astype(float)
        cm_fcn  = confusion_matrix(fcn_true,  token_pred).astype(float)

        def _row_norm(cm):
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            return cm / row_sums

        cm_avg_pct = (_row_norm(cm_bert) + _row_norm(cm_fcn)) / 2.0 * 100

        bert_metrics = self._cm_metrics(bert_true, token_pred, token_prob)
        fcn_metrics  = self._cm_metrics(fcn_true,  token_pred, token_prob)

        return cm_bert, cm_fcn, cm_avg_pct, bert_metrics, fcn_metrics

    def _seq_level_cm(self):
        """
        Compute sequence-level confusion matrix directly from the seq head output.
        """
        seq_true = np.concatenate(self.val_seq_labels).flatten().astype(int)
        seq_prob = np.concatenate(self.val_seq_probs).flatten()
        seq_pred = (seq_prob >= self.threshold).astype(int)

        cm  = confusion_matrix(seq_true, seq_pred).astype(float)
        metrics = self._cm_metrics(seq_true, seq_pred, seq_prob)

        return cm, seq_true, seq_pred, seq_prob, metrics

    # ── main plot/save routine ────────────────

    def plot_results(self):
        cp = self.checkpoint_dir

        # ── loss curve ──
        plt.figure(figsize=(8, 5))
        plt.plot(self.train_losses, label="Train loss")
        plt.plot(self.val_losses,   label="Val loss", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(osp.join(cp, "loss_curve.png"))
        plt.close()
        print(f"Saved loss curve → {osp.join(cp, 'loss_curve.png')}")

        if not self.val_token_probs:
            print("No validation predictions collected — skipping CM plots.")
            return

        # ── per-base averaged CM ──────────────────────────────────────────────
        cm_bert, cm_fcn, cm_avg_pct, bert_m, fcn_m = self._averaged_per_base_cm()

        # individual CMs (counts)
        self._save_cm(cm_bert.astype(int),
                      "Per-base CM — BERT head (counts)",
                      "cm_per_base_bert_counts")
        self._save_cm(cm_fcn.astype(int),
                      "Per-base CM — FCN head (counts)",
                      "cm_per_base_fcn_counts")

        # averaged CM (percentages)
        self._save_cm(cm_avg_pct,
                      "Per-base CM — Averaged BERT & FCN (%)",
                      "cm_per_base_avg_pct",
                      fmt=".2f")

        # also save a percentage version for each head individually
        def _pct(cm):
            rs = cm.sum(axis=1, keepdims=True)
            rs[rs == 0] = 1
            return cm / rs * 100

        self._save_cm(_pct(cm_bert),
                      "Per-base CM — BERT head (%)",
                      "cm_per_base_bert_pct",
                      fmt=".2f")
        self._save_cm(_pct(cm_fcn),
                      "Per-base CM — FCN head (%)",
                      "cm_per_base_fcn_pct",
                      fmt=".2f")

        # per-base PR curve — averaged labels, single token prob
        token_prob = np.concatenate(self.val_token_probs).flatten()
        bert_true  = np.concatenate(self.val_bert_labels).flatten()
        fcn_true   = np.concatenate(self.val_fcn_labels).flatten()

        if bert_true.shape == fcn_true.shape:
            avg_true = ((bert_true + fcn_true) / 2.0 >= 0.5).astype(int)
            prec_arr, rec_arr, _ = precision_recall_curve(avg_true, token_prob)
            ap = average_precision_score(avg_true, token_prob)
            plt.figure(figsize=(6, 5))
            plt.plot(rec_arr, prec_arr, linewidth=2)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Per-base PR Curve — Averaged BERT & FCN (AP={ap:.3f})")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(osp.join(cp, "pr_curve_per_base_avg.png"))
            plt.savefig(osp.join(cp, "pr_curve_per_base_avg.pdf"))
            plt.close()

        # ── sequence-level CM ─────────────────────────────────────────────────
        cm_seq, seq_true, seq_pred, seq_prob, seq_m = self._seq_level_cm()

        self._save_cm(cm_seq.astype(int),
                      "Sequence-level CM — seq head (counts)",
                      "cm_per_sequence_counts")
        self._save_cm(_pct(cm_seq),
                      "Sequence-level CM — seq head (%)",
                      "cm_per_sequence_pct",
                      fmt=".2f")

        # seq-level PR curve
        if len(np.unique(seq_true)) > 1:
            prec_arr, rec_arr, _ = precision_recall_curve(seq_true, seq_prob)
            ap_seq = average_precision_score(seq_true, seq_prob)
            plt.figure(figsize=(6, 5))
            plt.plot(rec_arr, prec_arr, linewidth=2)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Sequence-level PR Curve (AP={ap_seq:.3f})")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(osp.join(cp, "pr_curve_per_sequence.png"))
            plt.savefig(osp.join(cp, "pr_curve_per_sequence.pdf"))
            plt.close()

        # ── metrics summary text file ─────────────────────────────────────────
        metrics_file = osp.join(cp, "metrics_summary.txt")
        with open(metrics_file, "w") as f:
            f.write("=== PER-BASE METRICS — BERT HEAD ===\n")
            for k, v in bert_m.items():
                f.write(f"  {k:6s}: {v:.4f}\n")

            f.write("\n=== PER-BASE METRICS — FCN HEAD ===\n")
            for k, v in fcn_m.items():
                f.write(f"  {k:6s}: {v:.4f}\n")

            # simple average of the two heads' scalar metrics
            avg_m = {k: (bert_m[k] + fcn_m[k]) / 2.0 for k in bert_m}
            f.write("\n=== PER-BASE METRICS — AVERAGED (BERT + FCN) ===\n")
            for k, v in avg_m.items():
                f.write(f"  {k:6s}: {v:.4f}\n")

            f.write("\n=== SEQUENCE-LEVEL METRICS — SEQ HEAD ===\n")
            for k, v in seq_m.items():
                f.write(f"  {k:6s}: {v:.4f}\n")

        print(f"Saved metric summary → {metrics_file}")

    # ── save raw outputs ──────────────────────

    def save_val_outputs(self):
        """Save all validation predictions and labels to a compressed .npz file."""
        if not self.val_token_probs:
            print("No validation outputs to save.")
            return

        out_path = osp.join(self.checkpoint_dir, "val_outputs.npz")
        np.savez_compressed(
            out_path,
            # fused token head (single prob array, two label arrays)
            val_token_probs  = np.concatenate(self.val_token_probs,  axis=0),
            val_bert_labels  = np.concatenate(self.val_bert_labels,  axis=0),
            val_fcn_labels   = np.concatenate(self.val_fcn_labels,   axis=0),
            # seq head
            val_seq_probs    = np.concatenate(self.val_seq_probs,    axis=0),
            val_seq_labels   = np.concatenate(self.val_seq_labels,   axis=0),
            # loss curves
            train_losses     = np.array(self.train_losses),
            val_losses       = np.array(self.val_losses),
        )
        print(f"Validation outputs saved → {out_path}")