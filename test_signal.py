#!/usr/bin/python

import os
import sys
import argparse
import numpy as np
import os.path as osp

import torch
from torch.utils.data import DataLoader

from FCNmotif import BPNet, FCNA, FCNsignal
from datasets import EPIDataSet
from trainer import Trainer
from loss import NormalLoss
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

torch.multiprocessing.set_sharing_strategy('file_system')


def get_args():
    parser = argparse.ArgumentParser(description="FCNsignal for testing data")
    parser.add_argument("-d", dest="data_dir", type=str, default=None, help="A directory containing the test data.")
    parser.add_argument("-n", dest="name", type=str, default=None, help="The name of a specified data.")
    parser.add_argument("-g", dest="gpu", type=str, default='cpu', help="choose gpu device.")
    parser.add_argument("-c", dest="checkpoint", type=str, default='./models/', help="Where to save snapshots of the model.")
    return parser.parse_args()

def psl_correlation(true_list, pred_list):
    psl_list = []
    for t, p in zip(true_list, pred_list):
        t = np.array(t)
        p = np.array(p)
        if np.sum(t) == 0 or np.sum(p) == 0:
            psl_list.append(0)
            continue
        t_norm = t / np.linalg.norm(t)
        p_norm = p / np.linalg.norm(p)
        psl_list.append(np.sum(t_norm * p_norm))
    return np.mean(psl_list)

def main():
    args = get_args()
    if torch.cuda.is_available():
        if len(args.gpu.split(',')) == 1:
            device = torch.device("cuda:" + args.gpu)
        else:
            device = torch.device("cuda:" + args.gpu.split(',')[0])
    else:
        device = torch.device("cpu")

    f = open(osp.join(args.checkpoint, 'record.txt'), 'w')
    f.write('Set\tRMSE\tPearson\tPSL\n')

    motifLen = 12
    model = FCNsignal(motiflen=motifLen)
    checkpoint_file = osp.join(args.checkpoint, 'model_best_ebf.pth')
    chk = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(chk['model_state_dict'])
    model.to(device)
    model.eval()

    for set_type in ['test']:   #, 'neg'
        Data = np.load(osp.join(args.data_dir, f'{args.name}_{set_type}.npz'))
        seqs, signals = Data['data'], Data['signal']
        dataset = EPIDataSet(seqs, signals)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        p_all = []
        t_all = []
        for sample_batch in loader:
            X_data = sample_batch["data"].float().to(device)
            signal = sample_batch["signal"].float()
            with torch.no_grad():
                pred = model(X_data)
            signal_p = pred.view(-1).data.cpu().numpy()
            signal_t = signal.view(-1).data.cpu().numpy()
            p_all.append(signal_p)
            t_all.append(signal_t)

        rmse = np.mean([mean_squared_error(t, p) for t, p in zip(t_all, p_all)])
        pearson = np.mean([pearsonr(t, p)[0] for t, p in zip(t_all, p_all)])
        psl = psl_correlation(t_all, p_all)

        print(f"{set_type}: {rmse:.3f}\t{pearson:.3f}\t{psl:.3f}")
        f.write(f"{set_type}\t{rmse:.3f}\t{pearson:.3f}\t{psl:.3f}\n")

    f.close()

if __name__ == "__main__":
    main()
