import os
import sys
import argparse
import random
import json
import numpy as np
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from hyperopt import hp
from hyperopt.pyll.stochastic import sample

# custom modules
from VanillaFCNmotif import FCNA, FCNsignal, BPNet
from FCNmotif import FCNsignalMod
from datasets import EPIDataSet
from trainer import Trainer
from loss import BinaryLoss

torch.multiprocessing.set_sharing_strategy('file_system')


def randomparams():
    """Sample random hyperparameters."""
    space = {
        "beta1": 0.9,
        "beta2": 0.99,
        "lr": 0.001,
        "weight": 0.001,
    }
    params = sample(space)
    return params


def get_args():
    """Parse all the arguments."""
    parser = argparse.ArgumentParser(
        description="FCNsignal for predicting base-resolution binding signals"
    )

    parser.add_argument(
        "-d", dest="data_dir", type=str, required=True,
        help="Directory containing training data (.npz files)."
    )
    parser.add_argument(
        "-n", dest="name", type=str, required=True,
        help="Dataset name (used to locate '<name>_train.npz')."
    )
    parser.add_argument(
        "-g", dest="gpu", type=str, default='0',
        help="Choose GPU device. Use '-1' for CPU. E.g., '0,1'."
    )
    parser.add_argument(
        "-s", dest="seed", type=int, default=5,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "-b", dest="batch_size", type=int, default=500,
        help="Batch size for training."
    )
    parser.add_argument(
        "-e", dest="max_epoch", type=int, default=50,
        help="Number of training epochs."
    )
    parser.add_argument(
        "-c", dest="checkpoint", type=str, default='./models/',
        help="Directory to save model checkpoints and outputs."
    )
    return parser.parse_args()


def main():
    """Create the model and start the training."""
    args = get_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Checkpoint directory
    if not osp.exists(args.checkpoint):
        os.makedirs(args.checkpoint)
        print(f"Created checkpoint directory: {args.checkpoint}")

    # Device setup
    if args.gpu == '-1':
        device = torch.device("cpu")
        print("Using CPU for training.")
    else:
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
            device = torch.device("cuda:0")
            torch.cuda.manual_seed(args.seed)
            print(f"Using GPU(s): {args.gpu}")
        else:
            print("CUDA not available. Falling back to CPU.")
            device = torch.device("cpu")

    motifLen = 12

    # Load data
    data_path = osp.join(args.data_dir, f'{args.name}_train.npz')
    if not osp.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    Data = np.load(data_path)
    seqs, signals = Data['data'], Data['signal']

    # Split into train/validation
    ratio = 0.1
    n_val = int(len(seqs) * ratio)
    indices = list(range(len(seqs)))
    index_val = random.sample(indices, n_val)
    index_train = list(set(indices) - set(index_val))

    # Training data
    data_tr, signal_tr = seqs[index_train], signals[index_train]
    train_data = EPIDataSet(data_tr, signal_tr)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    # Validation data
    data_va, signal_va = seqs[index_val], signals[index_val]
    val_data = EPIDataSet(data_va, signal_va)
    val_loader = DataLoader(
        val_data, batch_size=1, shuffle=False, num_workers=0
    )

    best_f1_score = 0
    best_params = None

    for trial in range(1):
        params = randomparams()
        print(f"Trial {trial + 1} params: {params}")

        # Model
        model = FCNsignalMod(motiflen=motifLen, n_features=0)

        # Optimizer & Scheduler
        optimizer = optim.Adam(
            model.parameters(),
            lr=params['lr'],
            betas=(params['beta1'], params['beta2']),
            weight_decay=params['weight']
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

        # Criterion with class imbalance handling
        executor = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=nn.BCEWithLogitsLoss(),
            device=device,
            checkpoint=args.checkpoint,
            max_epoch=args.max_epoch,
            train_loader=train_loader,
            test_loader=val_loader,
            scheduler=scheduler,
        )

        pos_weight = executor.compute_pos_weight()
        executor.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Train
        best_f1, state_dict = executor.train()

        # Save best model
        if best_f1 > best_f1_score:
            best_f1_score = best_f1
            best_params = params
            checkpoint_file = osp.join(args.checkpoint, 'model_best_ebf.pth')
            torch.save({'model_state_dict': state_dict}, checkpoint_file)
            print(f"Best model saved to {checkpoint_file}")

    # Save best params
    if best_params is not None:
        param_path = osp.join(args.checkpoint, 'best_params.json')
        with open(param_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"Best params saved to {param_path}")


if __name__ == "__main__":
    main()
