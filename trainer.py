import os
import os.path as osp
import numpy as np
from copy import deepcopy

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score
)

import matplotlib.pyplot as plt
import torch
from loss import BinaryLoss

plt.rcParams["font.family"] = "serif"


class Trainer(object):
    """Trainer for base-resolution binary classification with class imbalance handling"""

    def __init__(self, model, optimizer, criterion, device, checkpoint,
                 max_epoch, train_loader, test_loader, scheduler,
                 threshold=0.5, signal_thresh=1e-4):
        """
        Args:
            signal_thresh: minimal value to consider a position as positive
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
        self.signal_thresh = signal_thresh

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

        # --- Handle class imbalance with pos_weight ---
        if isinstance(criterion, BinaryLoss) and hasattr(criterion, 'use_logits'):
            pos_weight = self.compute_pos_weight()
            self.criterion = BinaryLoss(pos_weight=pos_weight)
        else:
            self.criterion = criterion

    def threshold_signal(self, signal):
        """Convert small signals to binary 0/1 using signal_thresh"""
        return (signal >= self.signal_thresh).float()

    def compute_pos_weight(self):
        """Compute pos_weight from the training dataset for imbalanced binary labels"""
        all_signals = []
        for batch in self.train_loader:
            y = batch["signal"]
            y_binary = self.threshold_signal(y)
            all_signals.append(y_binary.view(-1))

        all_signals = torch.cat(all_signals)
        num_pos = (all_signals == 1).sum().item()
        num_neg = (all_signals == 0).sum().item()

        pos_weight_value = num_neg / (num_pos + 1e-8)
        pos_weight_value = min(max(pos_weight_value, 1.0), 10.0)

        pos_weight_tensor = torch.tensor(pos_weight_value, dtype=torch.float32).to(self.device)
        print(f"Computed pos_weight = {pos_weight_value:.3f}")
        return pos_weight_tensor

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
                self.optimizer.step()

                epoch_loss += loss.item()

                if epoch == self.max_epoch - 1:
                    self.train_preds.append(pred.detach().cpu().numpy())
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
                pred = self.model(X_data)
                loss = self.criterion(pred, signal)

            val_loss += loss.item()
            p_all.append(pred.cpu().numpy())
            t_all.append(signal.cpu().numpy())

        val_loss /= len(self.test_loader)
        self.val_losses.append(val_loss)

        # Flatten for metrics
        y_true = np.concatenate(t_all).flatten()
        y_pred_prob = np.concatenate(p_all).flatten()
        y_pred = (y_pred_prob >= self.threshold).astype(int)

        # Metrics
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
        val_losses = [x[5] for x in self.val_metrics]

        # --- Smooth validation loss (moving average) ---
        def smooth_curve(y, window=3):
            if len(y) < window:
                return np.array(y)
            return np.convolve(y, np.ones(window) / window, mode='valid')

        val_losses_smooth = smooth_curve(val_losses, window=3)

        # --- Plot Training vs Validation Loss ---
        plt.figure(figsize=(8, 5))
        plt.plot(self.train_losses, label="Training Loss (raw)")
        x_smooth = np.arange(len(val_losses_smooth)) + (len(val_losses) - len(val_losses_smooth))
        plt.plot(x_smooth, val_losses_smooth, label="Validation Loss (smoothed)", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(osp.join(self.checkpoint, "loss_curve.png"))
        plt.close()

        # --- Plot True vs Predicted signals (10 samples) ---
        def plot_signals(truth_batches, pred_batches, title, filename):
            all_truths = np.concatenate([t.reshape(-1, t.shape[-1]) for t in truth_batches], axis=0)
            all_preds = np.concatenate([p.reshape(-1, p.shape[-1]) for p in pred_batches], axis=0)
            all_preds_binary = (all_preds >= self.threshold).astype(int)
            num_samples = min(10, len(all_truths))

            for i in range(num_samples):
                x = all_truths[i]
                y = all_preds_binary[i]
                plt.figure(figsize=(10, 3))
                plt.plot(x, label="True (0/1)", color="black", linewidth=1)
                plt.plot(y, label="Pred Binary", color="orange", linewidth=1)
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

        # --- Confusion Matrix & PR Curve ---
        if len(self.val_preds) > 0:
            y_true = np.concatenate(self.val_truths).flatten()
            y_pred_prob = np.concatenate(self.val_preds).flatten()
            y_pred = (y_pred_prob >= self.threshold).astype(int)

            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap="Blues", values_format="d")
            plt.title("Confusion Matrix")
            plt.tight_layout()
            plt.savefig(osp.join(self.checkpoint, "confusion_matrix.png"))
            plt.savefig(osp.join(self.checkpoint, "confusion_matrix.pdf"))
            plt.close()

            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
            avg_prec = average_precision_score(y_true, y_pred_prob)
            plt.figure(figsize=(6, 5))
            plt.plot(recall, precision, linewidth=2)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Precision-Recall Curve (AP = {avg_prec:.3f})")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(osp.join(self.checkpoint, "precision_recall_curve.png"))
            plt.savefig(osp.join(self.checkpoint, "precision_recall_curve.pdf"))
            plt.close()

    def save_val_outputs(self):
        """Save validation predictions, metrics, and losses"""
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


    # def plot_results(self):
    #     # --- Plot Training Loss and Validation F1 ---
    #     val_losses = [x[5] for x in self.val_metrics]  # index 5 is val_loss
    #     plt.figure(figsize=(8, 5))
    #     plt.plot(self.train_losses, label='Training Loss', color='blue')
    #     plt.plot(val_losses, label='Validation Loss', color='orange')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.title('Training and Validation Loss')
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(osp.join(self.checkpoint, 'loss_curve.png'))
    #     plt.close()

    #     # --- Plot True vs Predicted Probabilities (10 samples) ---
    #     def plot_signals(truth_batches, pred_batches, title, filename):
    #         all_truths = np.concatenate([t.reshape(-1, t.shape[-1]) for t in truth_batches], axis=0)
    #         all_preds = np.concatenate([p.reshape(-1, p.shape[-1]) for p in pred_batches], axis=0)
    #         all_preds_binary = (all_preds >= self.threshold).astype(int)
    #         num_samples = min(10, len(all_truths))

    #         for i in range(num_samples):
    #             x = all_truths[i]
    #             y = all_preds_binary[i] 
    #             plt.figure(figsize=(10, 3))
    #             plt.plot(x, label='True (0/1)', color='black', linewidth=1)
    #             plt.plot(y, label='Pred Binary', color='orange', linewidth=1)
    #             plt.xlabel('Position')
    #             plt.ylabel('0 / 1')
    #             plt.title(f'{title} Sample {i+1}')
    #             plt.legend()
    #             plt.tight_layout()
    #             plt.savefig(osp.join(self.checkpoint, f'{filename}_{i+1}.pdf'))
    #             plt.savefig(osp.join(self.checkpoint, f'{filename}_{i+1}.png'))
    #             plt.close()

    #     plot_signals(self.train_truths, self.train_preds, "Train", "train_true_vs_pred")
    #     plot_signals(self.val_truths, self.val_preds, "Validation", "val_true_vs_pred")

    #     # --- Confusion Matrix and Precision-Recall Curve ---
    #     if len(self.val_preds) > 0:
    #         y_true = np.concatenate(self.val_truths).flatten()
    #         y_pred_prob = np.concatenate(self.val_preds).flatten()
    #         y_pred = (y_pred_prob >= self.threshold).astype(int)

    #         # Confusion Matrix
    #         cm = confusion_matrix(y_true, y_pred)
    #         disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    #         disp.plot(cmap='Blues', values_format='d')
    #         plt.title('Confusion Matrix')
    #         plt.tight_layout()
    #         plt.savefig(osp.join(self.checkpoint, 'confusion_matrix.png'))
    #         plt.savefig(osp.join(self.checkpoint, 'confusion_matrix.pdf'))
    #         plt.close()

    #         # Precision-Recall Curve
    #         precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    #         avg_prec = average_precision_score(y_true, y_pred_prob)
    #         plt.figure(figsize=(6, 5))
    #         plt.plot(recall, precision, color='blue', linewidth=2)
    #         plt.xlabel('Recall')
    #         plt.ylabel('Precision')
    #         plt.title(f'Precision-Recall Curve (AP = {avg_prec:.3f})')
    #         plt.grid(True)
    #         plt.tight_layout()
    #         plt.savefig(osp.join(self.checkpoint, 'precision_recall_curve.png'))
    #         plt.savefig(osp.join(self.checkpoint, 'precision_recall_curve.pdf'))
    #         plt.close()

    #     # --- Threshold Sweep: F1 / Precision / Recall vs Threshold ---
    #     thresholds = np.linspace(0.0, 1.0, 101)
    #     f1s, precisions, recalls = [], [], []

    #     for th in thresholds:
    #         y_pred_th = (y_pred_prob >= th).astype(int)
    #         precisions.append(precision_score(y_true, y_pred_th, zero_division=0))
    #         recalls.append(recall_score(y_true, y_pred_th, zero_division=0))
    #         f1s.append(f1_score(y_true, y_pred_th, zero_division=0))

    #     plt.figure(figsize=(7, 5))
    #     plt.plot(thresholds, precisions, label='Precision', linestyle='--')
    #     plt.plot(thresholds, recalls, label='Recall', linestyle='-.')
    #     plt.plot(thresholds, f1s, label='F1 Score', linewidth=2)
    #     plt.xlabel('Threshold')
    #     plt.ylabel('Score')
    #     plt.title('Precision, Recall, and F1 vs Threshold')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(osp.join(self.checkpoint, 'threshold_sweep.png'))
    #     plt.savefig(osp.join(self.checkpoint, 'threshold_sweep.pdf'))
    #     plt.close()

    #     # Report best threshold for info
    #     best_idx = np.argmax(f1s)
    #     best_th = thresholds[best_idx]
    #     best_f1_val = f1s[best_idx]
    #     self.threshold = best_th 
    #     print(f"\nBest threshold = {best_th:.2f} with F1 = {best_f1_val:.3f}\n")

    