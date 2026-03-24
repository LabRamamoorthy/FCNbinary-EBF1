import os
import sys
import json
import random
import argparse
import os.path as osp

import numpy as np
import torch
import torch.optim as optim
import transformers.optimization as op
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from fused_model import FusedModel, FusedDNADataset
from loss import FusedLoss
from fusedTrainer import Trainer, EarlyStopping

torch.multiprocessing.set_sharing_strategy('file_system')


# HYPERPARAMETERS

HPARAMS = {
    # model
    "motiflen"          : 10,
    "n_features"        : 0,
    "dropout"           : 0.2,
    "max_len"           : 512,

    # loss
    "t_seq"             : 0.2,
    "token_alpha"       : 0.5,   # weight for bert_labels in token loss
    "token_beta"        : 0.5,   # weight for fcn_signal in token loss
    "bert_pos_weight"   : None,
    "fcn_pos_weight"    : None,
    "seq_pos_weight"    : None,

    # optimiser
    "lr_bert"           : 2e-6,
    "lr_fcn"            : 1e-3,
    "lr_heads"          : 1e-3,
    "weight_decay_bert" : 0.01,
    "weight_decay_other": 1e-4,
    "adam_eps"          : 1e-8,
    "adam_betas"        : (0.9, 0.999),
    "warmup_ratio"      : 0.1,

    # training
    "batch_size"        : 32,
    "max_epoch"         : 20,
    "patience"          : 5,
    # "unfreeze_bert_at"  : 10,
    "clip_grad_norm"    : 1.0,
    "val_split"         : 0.2,
    "threshold"         : 0.5,
    "num_workers"       : 0,
}


# ARGS

def get_args():
    parser = argparse.ArgumentParser(
        description="FusedModel: FCNsignalMod + BERTSNR for base-resolution TFBS prediction"
    )
    parser.add_argument(
        "-d", dest="data_dir", type=str, required=True,
        help=(
            "Root output directory from dataset preparation. "
            "Expected structure: "
            "<data_dir>/bertsnr/train/dataset.txt  and  "
            "<data_dir>/fcn/train.npz"
        )
    )
    parser.add_argument(
        "-g", dest="gpu", type=str, default='0',
        help="GPU device id. Use '-1' for CPU."
    )
    parser.add_argument(
        "-s", dest="seed", type=int, default=42,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "-c", dest="checkpoint", type=str, default='./checkpoints/',
        help="Directory to save checkpoints and outputs."
    )
    parser.add_argument(
        "--bert_path", type=str,
        default="/home/ubuntu/data/dnabert2",
        help="Path to local DNABERT-2 snapshot used as backbone for BERTSNR."
    )
    return parser.parse_args()


# MAIN

def main():
    args = get_args()

    # reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    # checkpoint dir
    os.makedirs(args.checkpoint, exist_ok=True)
    print(f"Checkpoint directory: {args.checkpoint}")

    # device
    if args.gpu == '-1':
        device = torch.device("cpu")
        print("Using CPU.")
    else:
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
            device = torch.device("cuda:0")
            print(f"Using GPU: {args.gpu}")
        else:
            print("CUDA not available — falling back to CPU.")
            device = torch.device("cpu")

    # tokeniser
    print(f"Loading tokenizer from: {args.bert_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_path,
                                              trust_remote_code=True)

    bertsnr_path = osp.join(args.data_dir, "bertsnr", "train", "dataset.txt")
    fcn_path     = osp.join(args.data_dir, "fcn", "train.npz")

    if not osp.exists(bertsnr_path):
        raise FileNotFoundError(f"BERTsnr dataset not found: {bertsnr_path}")
    if not osp.exists(fcn_path):
        raise FileNotFoundError(f"FCN dataset not found: {fcn_path}")

    print(f"Loading BERTsnr data from : {bertsnr_path}")
    print(f"Loading FCN data from     : {fcn_path}")

    # dataset
    full_dataset = FusedDNADataset(
        bertsnr_txt = bertsnr_path,
        fcn_npz     = fcn_path,
        tokenizer   = tokenizer,
        max_len     = HPARAMS["max_len"],
    )

    n_total     = len(full_dataset)
    n_val       = max(1, int(n_total * HPARAMS["val_split"]))
    indices     = list(range(n_total))
    index_val   = random.sample(indices, n_val)
    index_train = list(set(indices) - set(index_val))

    print(f"Dataset: {n_total} total  |  "
          f"{len(index_train)} train  |  {n_val} val")

    train_dataset = Subset(full_dataset, index_train)
    val_dataset   = Subset(full_dataset, index_val)

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size  = HPARAMS["batch_size"],
        shuffle     = True,
        num_workers = HPARAMS["num_workers"],
        generator   = g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = 32,
        shuffle     = False,
        num_workers = HPARAMS["num_workers"],
    )

    # model
    print("Building FusedModel...")
    model = FusedModel(
        motiflen    = HPARAMS["motiflen"],
        n_features  = HPARAMS["n_features"],
        dropout     = HPARAMS["dropout"],
        # freeze_bert = True,
    ).to(device)

    # optimiser
    no_decay = ["bias", "LayerNorm.weight"]

    bert_decay    = [p for n, p in model.bert.named_parameters()
                     if not any(nd in n for nd in no_decay)]
    bert_no_decay = [p for n, p in model.bert.named_parameters()
                     if     any(nd in n for nd in no_decay)]

    optimizer = optim.AdamW(
        [
            {"params": bert_decay,
             "lr": HPARAMS["lr_bert"],
             "weight_decay": HPARAMS["weight_decay_bert"]},
            {"params": bert_no_decay,
             "lr": HPARAMS["lr_bert"],
             "weight_decay": 0.0},
            {"params": model.fcn.parameters(),
             "lr": HPARAMS["lr_fcn"],
             "weight_decay": HPARAMS["weight_decay_other"]},
            {"params": (list(model.bert_proj.parameters())        +
                        list(model.token_classifier.parameters()) +
                        list(model.seq_classifier.parameters())),
             "lr": HPARAMS["lr_heads"],
             "weight_decay": HPARAMS["weight_decay_other"]},
        ],
        eps   = HPARAMS["adam_eps"],
        betas = HPARAMS["adam_betas"],
    )

    # linear warmup + decay
    t_total   = len(train_loader) * HPARAMS["max_epoch"]
    scheduler = op.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = int(t_total * HPARAMS["warmup_ratio"]),
        num_training_steps = t_total,
    )

    # loss
    loss_fn = FusedLoss(
        t_seq           = HPARAMS["t_seq"],
        bert_pos_weight = HPARAMS["bert_pos_weight"],
        fcn_pos_weight  = HPARAMS["fcn_pos_weight"],
        seq_pos_weight  = HPARAMS["seq_pos_weight"],
        alpha           = HPARAMS["token_alpha"],
        beta            = HPARAMS["token_beta"],
    )

    # early stopping
    early_stopping = EarlyStopping(
        patience = HPARAMS["patience"],
        verbose  = True,
    )

    # trainer
    trainer = Trainer(
        model            = model,
        train_loader     = train_loader,
        val_loader       = val_loader,
        optimizer        = optimizer,
        loss_fn          = loss_fn,
        device           = device,
        checkpoint_dir   = args.checkpoint,
        scheduler        = scheduler,
        max_epoch        = HPARAMS["max_epoch"],
        threshold        = HPARAMS["threshold"],
        # unfreeze_bert_at = HPARAMS["unfreeze_bert_at"],
        clip_grad_norm   = HPARAMS["clip_grad_norm"],
        early_stopping   = early_stopping,
    )

    # auto-compute pos_weights from training data
    if (HPARAMS["bert_pos_weight"] is None or
        HPARAMS["fcn_pos_weight"]   is None or
        HPARAMS["seq_pos_weight"]   is None):

        print("pos_weights not set — computing from training data...")
        bert_pw, fcn_pw, seq_pw = trainer.compute_pos_weights()

        trainer.loss_fn = FusedLoss(
            t_seq           = HPARAMS["t_seq"],
            bert_pos_weight = bert_pw.item(),
            fcn_pos_weight  = fcn_pw.item(),
            seq_pos_weight  = seq_pw.item(),
            alpha           = HPARAMS["token_alpha"],
            beta            = HPARAMS["token_beta"],
        )

    # train
    best_val_loss, best_state = trainer.fit()

    # save best model
    if best_state is not None:
        model_path = osp.join(args.checkpoint, "model_best.pth")
        torch.save({'model_state_dict': best_state}, model_path)
        print(f"Best model saved → {model_path}")

    # save run config
    config = {**vars(args), **HPARAMS, "best_val_loss": best_val_loss}
    config["adam_betas"] = list(config["adam_betas"])
    config_path = osp.join(args.checkpoint, "run_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Run config saved → {config_path}")


if __name__ == "__main__":
    main()