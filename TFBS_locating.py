#!/usr/bin/python

import argparse
import os
import sys
import numpy as np
import os.path as osp
from Bio import SeqIO
import torch
from FCNmotif import FCNsignalMod
torch.multiprocessing.set_sharing_strategy('file_system')
import pandas as pd
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pyBigWig
plt.rcParams["font.family"] = "serif"


#  Helper functions 

def Smoothavg(x, window):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode='same')


def GetbigWig(bw_file, chrom, start, end):
    with pyBigWig.open(bw_file) as bw:
        return np.array(bw.values(chrom, start, end))


def extract_additional_features(bed_lines, feature_files, location_func,
                                window, seq_len, smooth, chrom_sizes):
    all_feats = []
    for feat_file in feature_files:
        feat_signals = []
        for line in bed_lines:
            chrom, start_s, end_s = line.strip().split()[:3]
            start, end = int(start_s), int(end_s)
            start_f, end_f = location_func(chrom, start, end)

            chrom_len = chrom_sizes.get(chrom, end_f + window)
            start_c = max(0, start_f - window // 2)
            end_c = min(end_f + window // 2, chrom_len)

            if smooth:
                vals = GetbigWig(feat_file, chrom, start_c, end_c)
                vals[np.isnan(vals)] = 0.
                vals = Smoothavg(vals, window)
            else:
                vals = GetbigWig(feat_file, chrom, start_f, end_f)
                vals[np.isnan(vals)] = 0.

            if len(vals) < seq_len:
                vals = np.pad(vals, (0, seq_len - len(vals)), mode='constant')
            elif len(vals) > seq_len:
                vals = vals[:seq_len]

            feat_signals.append(vals)

        feat_signals = np.array(feat_signals, dtype=np.float32)
        feat_signals = np.log10(1 + feat_signals)
        all_feats.append(feat_signals)

    all_feats = np.array(all_feats)
    all_feats = np.transpose(all_feats, (1, 0, 2))  # [N, F, SEQ_LEN]
    return all_feats


def one_hot(sequence_dict, chrom, start, end):
    seq_dict = {'A':[1, 0, 0, 0], 'G':[0, 0, 1, 0],
                'C':[0, 1, 0, 0], 'T':[0, 0, 0, 1],
                'a':[1, 0, 0, 0], 'g':[0, 0, 1, 0],
                'c':[0, 1, 0, 0], 't':[0, 0, 0, 1]}
    temp = []
    seq = str(sequence_dict[chrom].seq[start:end])
    for c in seq:
        temp.append(seq_dict.get(c, [0, 0, 0, 0]))
    temp = np.array([temp], dtype=np.float32)
    return temp.transpose((0, 2, 1))


#  Binary peak extraction 

def locate_binary(parameters, sequence_dict, model, beds, outdir, bw, feature_files=None, smooth=True, chrom_sizes=None):
    model.eval()
    beds_pos = []
    signals_prob = {}
    signals_binary = {}
    true_signals = {}
    num_pos = 0

    f_prob = open(osp.join(outdir, 'predictions_prob.txt'), 'w')
    f_bin = open(osp.join(outdir, 'predictions_binary.txt'), 'w')

    for i, line in enumerate(beds):
        bed = line.strip().split()
        chrom = bed[0]
        start = int(bed[1])
        end = int(bed[2])
        seq_len = end - start

        X_data = torch.from_numpy(one_hot(sequence_dict, chrom, start, end)).float().to(parameters['Device'])

        use_features = feature_files is not None and len(feature_files) > 0
        if use_features:
            feat_input = extract_additional_features(
                [line], feature_files, lambda c,s,e: (s,e),
                window=parameters['Window'],
                seq_len=seq_len,
                smooth=smooth,
                chrom_sizes=chrom_sizes
            )
            feat_input_tensor = torch.from_numpy(feat_input).float().to(parameters['Device'])
            X_input = torch.cat([X_data, feat_input_tensor], dim=1)
        else:
            X_input = X_data

        with torch.no_grad():
            logits = model(X_input)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            binary = (probs > parameters['Threshold']).astype(np.float32)

        f_prob.write('\t'.join([str(round(x, 3)) for x in probs]) + '\n')
        f_bin.write('\t'.join([str(int(x)) for x in binary]) + '\n')

        signals_prob[f"{chrom}:{start}-{end}"] = probs
        signals_binary[f"{chrom}:{start}-{end}"] = binary

        signal_true = np.nan_to_num(bw.values(chrom, start, end))
        true_signals[f"{chrom}:{start}-{end}"] = signal_true

        for pos in np.where(binary == 1)[0]:
            bed_start = start + pos
            bed_end = bed_start + 1
            beds_pos.append((chrom, bed_start, bed_end))
            num_pos += 1

    f_prob.close()
    f_bin.close()

    print(f"A total of {num_pos} positive positions (binary=1) found.")
    return beds_pos, signals_prob, signals_binary, true_signals

#  Saving values 

def save_per_base_true_values(signals_prob, signals_binary, true_signals, outdir,
                                       prob_threshold=0.5, true_threshold=0.1):
    
    os.makedirs(outdir, exist_ok=True)
    all_probs = []
    all_pred_bin = []
    all_true_val = []
    all_true_bin = []
    keys_list = []

    per_base_bed = osp.join(outdir, "per_base_values.bed")
    f = open(per_base_bed, 'w')

    for key in signals_prob.keys():
        chrom, start_end = key.split(":")
        start0, end0 = map(int, start_end.split("-"))

        probs = np.array(signals_prob[key], dtype=np.float32)
        pred_bin = (probs >= prob_threshold).astype(np.int32)
        true_raw = np.array(true_signals[key], dtype=np.float32)
        true_val = np.log10(1 + true_raw)
        true_bin = (true_raw >= true_threshold).astype(np.int32)

        # append to lists
        all_probs.append(probs)
        all_pred_bin.append(pred_bin)
        all_true_val.append(true_val)
        all_true_bin.append(true_bin)
        keys_list.append(key)

        # write BED
        for i in range(len(probs)):
            bed_start = start0 + i
            bed_end = bed_start + 1
            f.write(
                f"{chrom}\t{bed_start}\t{bed_end}\t"
                f"{probs[i]:.6f}\t{pred_bin[i]}\t"
                f"{true_val[i]:.6f}\t{true_bin[i]}\n"
            )

    f.close()
    print(f"Saved per-base BED → {per_base_bed}")

    # save all sequences as object arrays (variable length)
    combined_npz = osp.join(outdir, "all_sequences.npz")
    np.savez(
        combined_npz,
        pred_prob=np.array(all_probs, dtype=object),
        pred_bin=np.array(all_pred_bin, dtype=object),
        true_val=np.array(all_true_val, dtype=object),
        true_bin=np.array(all_true_bin, dtype=object),
        keys=np.array(keys_list)
    )

    print(f"Saved combined NPZ → {combined_npz}")

#  Plotting 

def lineplot_binary(signals_prob, signals_binary, true_signals, name, outdir, thres):
    for key in signals_prob.keys():
        prob = np.array(signals_prob[key])
        binary = np.array(signals_binary[key])
        true_signal = np.log10(1 + np.array(true_signals.get(key, np.zeros_like(prob))))

        x = np.arange(len(prob))
        fig, ax = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

        ax[0].plot(x, prob, color='orange', label="Predicted Probability")
        ax[0].axhline(thres, color='red', linestyle='--', label=f"Threshold = {thres}")
        ax[0].legend()
        ax[0].set_ylabel("Probability")

        ax[1].step(x, binary, where='mid', color='green', label="Binary (0/1)")
        ax[1].legend()
        ax[1].set_ylabel("Binary Output")

        ax[2].plot(x, true_signal, color='black', label="True Signal (log10)")
        ax[2].legend()
        ax[2].set_ylabel("True Signal")
        ax[2].set_xlabel("Position")

        plt.tight_layout()
        plt.savefig(osp.join(outdir, f"{key}_binary_plot.pdf"), format='pdf')
        plt.savefig(osp.join(outdir, f"{key}_binary_plot.png"), format='png')

        plt.close()


#  Utilities 

def load_chrom_sizes(path):
    chrom_sizes = {}
    with open(path) as f:
        for line in f:
            chrom, size = line.strip().split()
            chrom_sizes[chrom] = int(size)
    return chrom_sizes


def get_args():
    parser = argparse.ArgumentParser(description="FCNsignal binary classification (base resolution)")
    parser.add_argument("-i", dest="infile", type=str, required=True, help="Input BED file")
    parser.add_argument("-n", dest="name", type=str, default="ebf", help="Name of data/model")
    parser.add_argument("-g", dest="gpu", type=str, default='0', help="GPU device (or -1 for CPU)")
    parser.add_argument("-t", dest="threshold", type=float, default=0.5, help="Threshold for binary classification")
    parser.add_argument("-w", dest="window", type=int, default=60, help="Length of binding region")
    parser.add_argument("-c", dest="checkpoint", type=str, default='./models/', help="Model checkpoint folder")
    parser.add_argument("--features", nargs='*', default=None, help="Paths to bigWig files for additional features")
    parser.add_argument("--smooth", action='store_true', help="Apply smoothing to features")
    parser.add_argument("--chrom_sizes", type=str, default="Genome/mm9.chrom.sizes", help="Chromosome sizes file")
    return parser.parse_args()


#  Main 

def main():
    args = get_args()
    root = osp.dirname(__file__)
    infile = args.infile
    name = args.name
    threshold = args.threshold
    window = args.window

    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() and args.gpu != '-1' else "cpu")

    bw_truth = pyBigWig.open("EBF/p-value.bw")
    model_file = osp.join(args.checkpoint, "model_best_ebf.pth")
    outdir = osp.join(root, "Results/test_0.25")
    os.makedirs(outdir, exist_ok=True)

    sequence_dict = SeqIO.to_dict(SeqIO.parse(open(osp.join(root, "Genome/mm9.fa")), 'fasta'))
    with open(infile) as f:
        beds = f.readlines()

    chk = torch.load(model_file, map_location=device)
    state_dict = chk['model_state_dict']

    n_features = len(args.features) if args.features else 0
    model = FCNsignalMod(motiflen=12, n_features=n_features)
    model.load_state_dict(state_dict)
    model.to(device)

    parameters = {'Window': window, 'Name': name, 'Threshold': threshold, 'Device': device, 'Feature_Files': args.features}

    beds_pos, signals_prob, signals_binary, signals_true = locate_binary(
        parameters, sequence_dict, model, beds, outdir, bw_truth,
        feature_files=args.features, smooth=args.smooth,
        chrom_sizes=load_chrom_sizes(args.chrom_sizes)
    )

    save_per_base_true_values(signals_prob, signals_binary, signals_true, outdir, prob_threshold=0.5, true_threshold=0.1)
    with open(osp.join(outdir, 'binding_binary.bed'), 'w') as f:
        for chrom, start, end in beds_pos:
            f.write(f"{chrom}\t{start}\t{end}\n")

    lineplot_binary(signals_prob, signals_binary, signals_true, name, outdir, threshold)
    print("Binary classification complete")


if __name__ == "__main__":
    main()
