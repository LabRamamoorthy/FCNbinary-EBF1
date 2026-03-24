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
from loss import FusedLoss

plt.rcParams["font.family"] = "serif"


# EARLY STOPPING

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


# TRAINER

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
        # unfreeze_bert_at = 3,
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
        # self.unfreeze_bert_at = unfreeze_bert_at
        self.clip_grad_norm   = clip_grad_norm
        self.early_stopping   = early_stopping
        self.checkpoint_dir   = checkpoint_dir

        os.makedirs(checkpoint_dir, exist_ok=True)

        # state
        self.epoch         = 0
        self.best_val_loss = float('inf')
        self.state_best    = None

        self.train_losses = []
        self.val_losses   = []

    # INTERNAL HELPERS

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

        return total, breakdown

    # CLASS-WEIGHT HELPER

    def compute_pos_weights(self):
        pos_bert = neg_bert = 0   # for bert_pos_weight (bert_labels)
        pos_fcn  = neg_fcn  = 0   # for fcn_pos_weight  (fcn_signal)
        pos_seq  = neg_seq  = 0   # for seq_pos_weight  (seq_label)

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

    # TRAIN ONE EPOCH

    def _train_epoch(self):
        self.model.train()
        agg      = {'token': 0.0, 'seq': 0.0, 'total': 0.0}
        progress = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")

        for batch in progress:
            batch = self._to_device(batch)
            self.optimizer.zero_grad()

            total, breakdown = self._step(batch)

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

    # VALIDATE ONE EPOCH

    @torch.no_grad()
    def _val_epoch(self):
        self.model.eval()
        agg      = {'token': 0.0, 'seq': 0.0, 'total': 0.0}
        progress = tqdm(self.val_loader, desc="Validating")

        for batch in progress:
            batch = self._to_device(batch)
            _, breakdown = self._step(batch)

            for k in agg:
                agg[k] += breakdown[k]

        n   = len(self.val_loader)
        avg = {k: v / n for k, v in agg.items()}
        self.val_losses.append(avg['total'])

        print(f"  [val] loss_total={avg['total']:.4f}  "
              f"token={avg['token']:.4f}  seq={avg['seq']:.4f}")

        # save best state by val_loss
        if avg['total'] < self.best_val_loss:
            self.best_val_loss = avg['total']
            self.state_best    = deepcopy(self.model.state_dict())
            self._save_checkpoint(self.epoch, avg['total'])

        return avg

    # CHECKPOINT

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

    # FIT

    def fit(self):
        early_stopped = False

        for epoch in range(1, self.max_epoch + 1):
            self.epoch = epoch

            # unfreeze BERT after warm-up epochs
            # if epoch == self.unfreeze_bert_at + 1:
            #     print(f"\n[Epoch {epoch}] Unfreezing BERT with reduced LR...")
            #     for p in self.model.bert.parameters():
            #         p.requires_grad = True
            #     for pg in self.optimizer.param_groups[:2]:   # first two groups are BERT
            #         pg['lr'] = pg['lr'] * 0.1
            #         print(f"  BERT lr reduced to {pg['lr']:.2e}")

            print(f"\n{'═' * 60}")
            print(f"Epoch {epoch} / {self.max_epoch}")
            print(f"{'═' * 60}")

            self._train_epoch()
            val_avg = self._val_epoch()

            if self.early_stopping is not None:
                if self.early_stopping(val_avg['total'], epoch):
                    early_stopped = True
                    break

        print("\nTraining finished. Plotting loss curve...")
        self.plot_results()

        status = f"stopped early at epoch {epoch}" if early_stopped \
                 else f"completed {self.max_epoch} epochs"
        print(f"Training {status}  |  best_val_loss={self.best_val_loss:.4f}")

        return self.best_val_loss, self.state_best

    # PLOTTING

    def plot_results(self):
        cp = self.checkpoint_dir

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