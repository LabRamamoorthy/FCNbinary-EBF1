import os
import math
import datetime
import numpy as np
import os.path as osp
from copy import deepcopy
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import torch

class Trainer(object):
    """build a trainer"""
    def __init__(self, model, optimizer, criterion, device, checkpoint, max_epoch,
                 train_loader, test_loader, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.scheduler = scheduler
        self.max_epoch = max_epoch
        self.checkpoint = checkpoint
        self.epoch = 0
        self.rmse_lowest = 10000
        self.pr_best = 0
        self.state_best = None
        # Tracking variables
        self.train_losses = []
        self.val_rmse = []
        self.val_pr = []
        self.train_preds = []
        self.train_truths = []
        self.val_preds = []
        self.val_truths = []


    def train(self):
        """training the model"""
        self.model.to(self.device)
        for epoch in range(self.max_epoch):
            # set training mode during the training process
            self.model.train()
            self.epoch = epoch
            epoch_loss = 0 
            for i_batch, sample_batch in enumerate(self.train_loader):
                X_data = sample_batch["data"].float().to(self.device)
                signal = sample_batch["signal"].float().to(self.device)
                pred = self.model(X_data)
                loss = self.criterion(pred, signal)
                if np.isnan(loss.item()) or not np.isfinite(loss.item()):
                    print('loss is nan or inf while training')
                    return self.rmse_lowest, self.pr_best, self.state_best
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                # Collect true/predicted for plotting in final epoch
                if epoch == self.max_epoch - 1:
                    self.train_preds.append(pred.view(-1).detach().cpu().numpy())
                    self.train_truths.append(signal.view(-1).cpu().numpy())
                print("epoch/i_batch: {}/{}---loss: {:.4f}---lr: {:.5f}".format(self.epoch, i_batch,
                                                    loss.item(), self.optimizer.param_groups[0]['lr']))
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            self.train_losses.append(avg_epoch_loss)
            # validation and save the model with higher accuracy
            self.test()
            self.scheduler.step()
        self.plot_results()
        return self.rmse_lowest, self.pr_best, self.state_best

    def test(self):
        """validate the performance of the trained model."""
        self.model.eval()
        p_all = []
        t_all = []
        for i_batch, sample_batch in enumerate(self.test_loader):
            X_data = sample_batch["data"].float().to(self.device)
            signal = sample_batch["signal"].float()
            with torch.no_grad():
                pred = self.model(X_data)
            p_all.append(pred.view(-1).data.cpu().numpy())
            t_all.append(signal.view(-1).data.numpy())
        total_rmse = []
        total_pr = []
        for t_one, p_one in zip(t_all, p_all):
            mse = mean_squared_error(t_one, p_one)
            total_rmse.append(mse)

            # Safe Pearson
            if np.std(t_one) != 0 and np.std(p_one) != 0:
                pr_val = pearsonr(t_one, p_one)[0]
                total_pr.append(pr_val)
            else:
                total_pr.append(np.nan)

        rmse = np.mean(total_rmse)
        valid_pr = [x for x in total_pr if not np.isnan(x)]
        pr = np.mean(valid_pr) if valid_pr else 0.0
        self.val_rmse.append(rmse)
        self.val_pr.append(pr)
        if self.epoch == self.max_epoch - 1:
            self.val_preds = p_all
            self.val_truths = t_all
        if self.rmse_lowest > rmse:
            self.rmse_lowest = rmse
            self.pr_best = pr
            self.state_best = deepcopy(self.model.state_dict())
        print("rmse: {:.3f}\tpearsonr: {:.3f}\n".format(rmse, pr))
        return rmse
    
    def plot_results(self):
        # --- Plot Training and Validation Loss ---
        plt.figure(figsize=(8, 5))
        plt.plot(self.train_losses, label='Training Loss', color='blue')
        plt.plot(self.val_rmse, label='Validation RMSE', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(osp.join(self.checkpoint, 'loss_curve.png'))
        plt.close()

        # --- Plot true vs predicted signals (up to 10 samples) ---
        def plot_signals(truth_batches, pred_batches, title, filename):
            all_truths = np.concatenate([t.reshape(-1, t.shape[-1]) for t in truth_batches], axis=0)
            all_preds = np.concatenate([p.reshape(-1, p.shape[-1]) for p in pred_batches], axis=0)
            num_samples = min(10, len(all_truths))

            for i in range(num_samples):
                x = all_truths[i]
                y = all_preds[i]

                if np.std(x) == 0 or np.std(y) == 0:
                    pr_val = np.nan
                else:
                    pr_val = pearsonr(x, y)[0]

                plt.figure(figsize=(10, 3))
                plt.plot(x, label='True Signal', color='black')
                plt.plot(y, label='Predicted Signal', color='orange')
                plt.xlabel('Position')
                plt.ylabel('Signal')
                plt.title(f'{title} Sample {i+1} - Pearson r: {pr_val:.3f}')
                plt.legend()
                plt.tight_layout()
                plt.savefig(osp.join(self.checkpoint, f'{filename}_{i+1}.png'))
                plt.close()

        # Plot for final epoch
        plot_signals(self.train_truths, self.train_preds, "Train", "train_true_vs_pred")
        plot_signals(self.val_truths, self.val_preds, "Validation", "val_true_vs_pred")

