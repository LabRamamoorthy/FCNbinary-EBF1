import os
import os.path as osp
import numpy as np
from copy import deepcopy
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score
)

import matplotlib.pyplot as plt
import torch
from loss import BinaryLoss # FocalLoss

plt.rcParams["font.family"] = "serif"


class Trainer(object):
    """Trainer for base-resolution binary classification with class imbalance handling"""

    def __init__(self, model, optimizer, criterion, device, checkpoint,
                 max_epoch, train_loader, test_loader, scheduler,
                 threshold=0.5, seq_min_run=6):
        """
        Args:
            signal_thresh: minimal value to consider a position as positive
            seq_min_run: minimal number of consecutive bases to call a sequence positive
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.scheduler = scheduler
        self.max_epoch = max_epoch
        self.checkpoint = checkpoint
        self.threshold = threshold
        self.seq_min_run = seq_min_run

        self.epoch = 0
        self.best_f1 = 0
        self.state_best = None

        # Tracking variables
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []  # (acc, prec, rec, f1, auc, val_loss)
        self.train_preds = []
        self.train_truths = []
        self.val_preds = []
        self.val_truths = []

        # # sequence-level metrics container
        # self.sequence_metrics = None
        # self.criterion = criterion

        if isinstance(criterion, BinaryLoss) and hasattr(criterion, 'use_logits'):
            pos_weight = self.compute_pos_weight()
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            print("Using BCEWithLogitsLoss with computed pos_weight")
        else:
            self.criterion = criterion

    def compute_pos_weight(self):
        """Compute pos_weight for BCEWithLogitsLoss based on class distribution"""
        total_pos = 0
        total_neg = 0

        print("Computing class distribution from training data...")
        for sample_batch in self.train_loader:
            signal = sample_batch["signal"].float()
            if signal.ndim == 3:
                signal = signal.squeeze(1)
            
            signal_np = signal.cpu().numpy()  
            
            total_pos += np.sum(signal_np == 1)
            total_neg += np.sum(signal_np == 0)

        if total_pos == 0 or total_neg == 0:
            print("Warning: only one class present in training data. Using pos_weight = 1.0")
            pos_weight_value = 1.0
        else:
            # Weight inversely proportional to class frequency
            pos_weight_value = total_neg / total_pos
            print(f"Class distribution: Positives = {total_pos:,}, Negatives = {total_neg:,}")
            print(f"Computed pos_weight = {pos_weight_value:.3f}")

        pos_weight_tensor = torch.tensor(pos_weight_value, dtype=torch.float32).to(self.device)
        return pos_weight_tensor


    # def compute_pos_weight(self):
    #     """Use a light constant pos_weight for balanced dataset."""
    #     pos_weight_value = 1.5   # slightly more weight for positives
    #     pos_weight_tensor = torch.tensor(pos_weight_value, dtype=torch.float32).to(self.device)
    #     print(f"Using constant pos_weight = {pos_weight_value:.3f}")
    #     return pos_weight_tensor

    def has_run_of_ones(self, arr, min_run=None):
        """
        Return True if arr (1D array-like of 0/1) contains >= min_run consecutive ones.
        Accepts numpy arrays or lists. Treats non-zero as 1.
        """
        if min_run is None:
            min_run = self.seq_min_run
        count = 0
        for v in np.asarray(arr).astype(int):
            if v == 1:
                count += 1
                if count >= min_run:
                    return True
            else:
                count = 0
        return False

    def compute_sequence_metrics(self, min_run=None):
        """
        Compute per-sequence metrics using the >=min_run consecutive-ones rule.
        Uses self.val_truths and self.val_preds which are lists of numpy arrays.
        Returns: (acc, prec, rec, f1, auc, seq_true_list, seq_pred_list)
        """
        if min_run is None:
            min_run = self.seq_min_run

        seq_true = []
        seq_pred = []

        # iterate through stored batches
        for truth_batch, pred_batch in zip(self.val_truths, self.val_preds):
            # ensure shape is (batch, seq_len)
            truth_batch = np.asarray(truth_batch)
            pred_batch = np.asarray(pred_batch)

            # If shapes are (N, L) already, fine. If (N, 1, L) should have been squeezed earlier.
            if truth_batch.ndim == 1:
                truth_batch = truth_batch.reshape(1, -1)
            if pred_batch.ndim == 1:
                pred_batch = pred_batch.reshape(1, -1)

            # reshape safety: some code stores (batch, seq_len) while others store (batch, seq_len, 1)
            if truth_batch.ndim == 3 and truth_batch.shape[1] == 1:
                truth_batch = truth_batch.reshape(truth_batch.shape[0], truth_batch.shape[2])
            if pred_batch.ndim == 3 and pred_batch.shape[1] == 1:
                pred_batch = pred_batch.reshape(pred_batch.shape[0], pred_batch.shape[2])

            for t, p in zip(truth_batch, pred_batch):
                t_bin = np.asarray(t).astype(int)
                p_bin = (np.asarray(p) >= self.threshold).astype(int)

                truth_has = self.has_run_of_ones(t_bin, min_run=min_run)
                pred_has = self.has_run_of_ones(p_bin, min_run=min_run)

                seq_true.append(1 if truth_has else 0)
                seq_pred.append(1 if pred_has else 0)

        # metrics
        if len(seq_true) == 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan, seq_true, seq_pred

        seq_acc = accuracy_score(seq_true, seq_pred)
        seq_prec = precision_score(seq_true, seq_pred, zero_division=0)
        seq_rec = recall_score(seq_true, seq_pred, zero_division=0)
        seq_f1 = f1_score(seq_true, seq_pred, zero_division=0)
        try:
            seq_auc = roc_auc_score(seq_true, seq_pred)
        except Exception:
            seq_auc = np.nan

        return seq_acc, seq_prec, seq_rec, seq_f1, seq_auc, seq_true, seq_pred

    def train(self):
        """Training the model"""
        self.model.to(self.device)

        for epoch in range(self.max_epoch):
            self.model.train()
            self.epoch = epoch
            epoch_loss = 0

            for i_batch, sample_batch in enumerate(self.train_loader):
                X_data = sample_batch["data"].float().to(self.device)
                signal = sample_batch["signal"].float().to(self.device)

                if signal.ndim == 3:
                    signal = signal.squeeze(1)

                pred = self.model(X_data)
                loss = self.criterion(pred, signal)

                if np.isnan(loss.item()) or not np.isfinite(loss.item()):
                    print("loss is nan or inf while training")
                    return self.best_f1, self.state_best

                self.optimizer.zero_grad()
                loss.backward()
                # for name, param in self.model.named_parameters():
                #     if param.grad is not None:
                #         print(name, param.grad.norm().item())
                self.optimizer.step()

                epoch_loss += loss.item()

                if epoch == self.max_epoch - 1:
                    probs = torch.sigmoid(pred)  
                    self.train_preds.append(probs.detach().cpu().numpy()) 
                    self.train_truths.append(signal.cpu().numpy())

                print(f"epoch/i_batch: {epoch}/{i_batch} --- "
                      f"loss: {loss.item():.4f} --- "
                      f"lr: {self.optimizer.param_groups[0]['lr']:.5f}")

            avg_epoch_loss = epoch_loss / len(self.train_loader)
            self.train_losses.append(avg_epoch_loss)

            # Validation
            self.test()
            self.scheduler.step()

        # Plot and save results
        self.plot_results()
        self.save_val_outputs()

        return self.best_f1, self.state_best

    def test(self):
        """Validate model performance on test data"""
        self.model.eval()
        p_all, t_all = [], []
        val_loss = 0

        for sample_batch in self.test_loader:
            X_data = sample_batch["data"].float().to(self.device)
            signal = sample_batch["signal"].float().to(self.device)

            if signal.ndim == 3:
                signal = signal.squeeze(1)

            with torch.no_grad():
                logits = self.model(X_data)
                loss = self.criterion(logits, signal)
                probs = torch.sigmoid(logits)

            val_loss += loss.item()
            p_all.append(probs.cpu().numpy())
            t_all.append(signal.cpu().numpy())

        val_loss /= len(self.test_loader)
        self.val_losses.append(val_loss)

        # Flatten for per-base metrics
        y_true = np.concatenate(t_all).flatten()
        y_pred_prob = np.concatenate(p_all).flatten()
        y_pred = (y_pred_prob >= self.threshold).astype(int)

        # Per-base Metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_pred_prob)
        except ValueError:
            auc = np.nan

        self.val_metrics.append((acc, prec, rec, f1, auc, val_loss))

        if self.epoch == self.max_epoch - 1:
            # store batch-wise arrays for plotting and sequence-level evaluation
            self.val_preds = p_all
            self.val_truths = t_all

        if f1 > self.best_f1:
            self.best_f1 = f1
            self.state_best = deepcopy(self.model.state_dict())

        print(f"ACC: {acc:.3f}\tPREC: {prec:.3f}\tREC: {rec:.3f}\t"
              f"F1: {f1:.3f}\tAUC: {auc:.3f}")

        return f1

    def plot_results(self):
        """Plot training/validation loss, confusion matrix, and sample predictions"""
        # --- Extract validation losses ---
        val_losses = [x[5] for x in self.val_metrics] if len(self.val_metrics) > 0 else []

        # --- Plot Training vs Validation Loss ---
        plt.figure(figsize=(8, 5))
        if len(self.train_losses) > 0:
            plt.plot(self.train_losses, label="Training Loss ")
        if len(val_losses) > 0:
            plt.plot(val_losses, label="Validation Loss ", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(osp.join(self.checkpoint, "loss_curve.png"))
        plt.close()

        # --- Plot True vs Predicted signals (10 samples) ---
        def plot_signals(truth_batches, pred_batches, title, filename):
            if len(truth_batches) == 0 or len(pred_batches) == 0:
                return
            all_truths = np.concatenate([t.reshape(-1, t.shape[-1]) for t in truth_batches], axis=0)
            all_preds = np.concatenate([p.reshape(-1, p.shape[-1]) for p in pred_batches], axis=0)
            all_preds_binary = (all_preds >= self.threshold).astype(int)
            num_samples = min(10, len(all_truths))

            for i in range(num_samples):
                x = all_truths[i]
                y = all_preds_binary[i]
                plt.figure(figsize=(10, 3))
                plt.plot(x, label="True (0/1)", linewidth=1)
                plt.plot(y, label="Pred Binary", linewidth=1)
                plt.xlabel("Position")
                plt.ylabel("0 / 1")
                plt.title(f"{title} Sample {i+1}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(osp.join(self.checkpoint, f"{filename}_{i+1}.pdf"))
                plt.savefig(osp.join(self.checkpoint, f"{filename}_{i+1}.png"))
                plt.close()

        plot_signals(self.train_truths, self.train_preds, "Train", "train_true_vs_pred")
        plot_signals(self.val_truths, self.val_preds, "Validation", "val_true_vs_pred")

        # --- Sequence-level Confusion Matrix & Per-base PR Curve ---
        # --- TRAINING CONFUSION MATRIX ---
        if len(self.train_preds) > 0:
            y_true_train = np.concatenate(self.train_truths).flatten()
            y_pred_prob_train = np.concatenate(self.train_preds).flatten()
            y_pred_train = (y_pred_prob_train >= self.threshold).astype(int)

            cm_train = confusion_matrix(y_true_train, y_pred_train)
            disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train)
            disp_train.plot(cmap="Blues", values_format="d")
            plt.title("Training Confusion Matrix (Per-base)")
            plt.tight_layout()
            plt.savefig(osp.join(self.checkpoint, "confusion_matrix_train_per_base.png"))
            plt.savefig(osp.join(self.checkpoint, "confusion_matrix_train_per_base.pdf"))
            plt.close()
        # --- VALIDATION CONFUSION MATRIX ---
        if len(self.val_preds) > 0:
            # per-base arrays
            y_true_base = np.concatenate(self.val_truths).flatten()
            y_pred_prob_base = np.concatenate(self.val_preds).flatten()
            y_pred_base = (y_pred_prob_base >= self.threshold).astype(int)

            # Per-base Confusion Matrix 
            cm_base = confusion_matrix(y_true_base, y_pred_base)
            disp_base = ConfusionMatrixDisplay(confusion_matrix=cm_base)
            disp_base.plot(cmap="Blues", values_format="d")
            plt.title("Per-base Confusion Matrix")
            plt.tight_layout()
            plt.savefig(osp.join(self.checkpoint, "confusion_matrix_per_base.png"))
            plt.savefig(osp.join(self.checkpoint, "confusion_matrix_per_base.pdf"))
            plt.close()

            # Precision-Recall Curve (per-base)
            precision, recall, _ = precision_recall_curve(y_true_base, y_pred_prob_base)
            avg_prec = average_precision_score(y_true_base, y_pred_prob_base)
            plt.figure(figsize=(6, 5))
            plt.plot(recall, precision, linewidth=2)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Precision-Recall Curve (Per-base, AP = {avg_prec:.3f})")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(osp.join(self.checkpoint, "precision_recall_curve_per_base.png"))
            plt.savefig(osp.join(self.checkpoint, "precision_recall_curve_per_base.pdf"))
            plt.close()

            # Sequence-level metrics & confusion matrix
            seq_acc, seq_prec, seq_rec, seq_f1, seq_auc, seq_true_list, seq_pred_list = self.compute_sequence_metrics()

            # Sequence-level confusion matrix
            cm_seq = confusion_matrix(seq_true_list, seq_pred_list)
            disp_seq = ConfusionMatrixDisplay(confusion_matrix=cm_seq)
            disp_seq.plot(cmap="Blues", values_format="d")
            plt.title(f"Sequence-level Confusion Matrix (>= {self.seq_min_run}bp run)")
            plt.tight_layout()
            plt.savefig(osp.join(self.checkpoint, "confusion_matrix_per_sequence.png"))
            plt.savefig(osp.join(self.checkpoint, "confusion_matrix_per_sequence.pdf"))
            plt.close()

            # store sequence metrics
            self.sequence_metrics = {
                "acc": seq_acc,
                "prec": seq_prec,
                "rec": seq_rec,
                "f1": seq_f1,
                "auc": seq_auc
            }

        else:
            # no validation preds stored
            self.sequence_metrics = {
                "acc": np.nan,
                "prec": np.nan,
                "rec": np.nan,
                "f1": np.nan,
                "auc": np.nan
            }

        # ----- SAVE METRICS TO TEXT FILE -----
        metrics_file = osp.join(self.checkpoint, "metrics_summary.txt")

        # Per-base final epoch metrics (if available)
        if len(self.val_metrics) > 0:
            base_acc, base_prec, base_rec, base_f1, base_auc, _ = self.val_metrics[-1]
        else:
            base_acc = base_prec = base_rec = base_f1 = base_auc = np.nan

        with open(metrics_file, "w") as f:
            f.write("=== PER-BASE METRICS (FINAL EPOCH) ===\n")
            f.write(f"Accuracy:  {base_acc:.4f}\n")
            f.write(f"Precision: {base_prec:.4f}\n")
            f.write(f"Recall:    {base_rec:.4f}\n")
            f.write(f"F1-score:  {base_f1:.4f}\n")
            f.write(f"AUC:       {base_auc:.4f}\n\n")

            f.write(f"=== PER-SEQUENCE METRICS (>= {self.seq_min_run} bp rule) ===\n")
            f.write(f"Accuracy:  {self.sequence_metrics['acc']:.4f}\n")
            f.write(f"Precision: {self.sequence_metrics['prec']:.4f}\n")
            f.write(f"Recall:    {self.sequence_metrics['rec']:.4f}\n")
            f.write(f"F1-score:  {self.sequence_metrics['f1']:.4f}\n")
            f.write(f"AUC:       {self.sequence_metrics['auc']:.4f}\n")

        print(f"Saved metric summary to {metrics_file}")

    def save_val_outputs(self):
        """Save validation predictions, metrics, and losses"""
        if len(self.val_preds) == 0 or len(self.val_truths) == 0:
            print("No validation outputs to save.")
            return

        val_preds_array = np.concatenate(self.val_preds, axis=0)
        val_truths_array = np.concatenate(self.val_truths, axis=0)
        val_preds_binary = (val_preds_array >= self.threshold).astype(int)

        val_losses_array = np.array(self.val_losses) if hasattr(self, "val_losses") else np.array([])

        out_path = osp.join(self.checkpoint, "val_outputs.npz")
        np.savez_compressed(
            out_path,
            val_preds=val_preds_array,
            val_preds_binary=val_preds_binary,
            val_truths=val_truths_array,
            train_losses=np.array(self.train_losses),
            val_losses=val_losses_array,
            val_metrics=np.array(self.val_metrics),
        )
        print(f"Validation outputs saved to: {out_path}")