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
import pyBigWig

#  Helper functions 

def Smoothavg(x, window):
    pad = window // 2
    x_padded = np.pad(x, (pad, pad), mode='edge')
    avg = []
    for j in range(len(x_padded) - window + 1):
        avg.append(np.mean(x_padded[j:(j + window)]))
    return np.array(avg)[:len(x)]  


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

def locate_binary(parameters, sequence_dict, model, beds, outdir, bw,
                  feature_files=None, smooth=True, chrom_sizes=None,
                  feature_files_rep=None):
    model.eval()
    beds_pos = []
    signals_prob = {}
    signals_binary = {}
    true_signals = {}
    atac_signals = {}
    atac_signals_rep = {}
    num_pos = 0

    f_prob = open(osp.join(outdir, 'predictions_prob.txt'), 'w')
    f_bin  = open(osp.join(outdir, 'predictions_binary.txt'), 'w')

    for i, line in enumerate(beds):
        bed     = line.strip().split()
        chrom   = bed[0]
        start   = int(bed[1])
        end     = int(bed[2])
        seq_len = end - start

        X_data = torch.from_numpy(
            one_hot(sequence_dict, chrom, start, end)
        ).float().to(parameters['Device'])

        use_features = feature_files is not None and len(feature_files) > 0
        if use_features:
            feat_input = extract_additional_features(
                [line], feature_files, lambda c, s, e: (s, e),
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
            probs  = torch.sigmoid(logits).cpu().numpy().flatten()
            binary = (probs > parameters['Threshold']).astype(np.float32)

        f_prob.write('\t'.join([str(round(x, 3)) for x in probs]) + '\n')
        f_bin.write('\t'.join([str(int(x)) for x in binary]) + '\n')

        key = f"{chrom}:{start}-{end}"
        signals_prob[key]   = probs
        signals_binary[key] = binary

        # Truth signal 
        signal_true = np.nan_to_num(bw.values(chrom, start, end))
        true_signals[key] = signal_true  

        # ATAC signal 
        if use_features and len(feature_files) >= 1:
            atac_raw = GetbigWig(feature_files[0], chrom, start, end)
            atac_raw = np.nan_to_num(atac_raw)
            atac_signals[key] = atac_raw  
        else:
            atac_signals[key] = None

        # ATAC replicate signal 
        use_features_rep = feature_files_rep is not None and len(feature_files_rep) >= 1
        if use_features_rep:
            atac_rep_raw = GetbigWig(feature_files_rep[0], chrom, start, end)
            atac_rep_raw = np.nan_to_num(atac_rep_raw)
            atac_signals_rep[key] = atac_rep_raw  
        else:
            atac_signals_rep[key] = None

        for pos in np.where(binary == 1)[0]:
            beds_pos.append((chrom, start + pos, start + pos + 1))
            num_pos += 1

    f_prob.close()
    f_bin.close()

    print(f"A total of {num_pos} positive positions (binary=1) found.")
    return beds_pos, signals_prob, signals_binary, true_signals, atac_signals, atac_signals_rep


#  Saving values 

def save_per_base_true_values(signals_prob, signals_binary, true_signals, outdir,
                              prob_threshold=0.5, true_threshold=0.1):

    os.makedirs(outdir, exist_ok=True)
    all_probs    = []
    all_pred_bin = []
    all_true_val = []
    all_true_bin = []
    keys_list    = []

    per_base_bed = osp.join(outdir, "per_base_values.bed")
    f = open(per_base_bed, 'w')

    for key in signals_prob.keys():
        chrom, start_end = key.split(":")
        start0, end0 = map(int, start_end.split("-"))

        probs    = np.array(signals_prob[key], dtype=np.float32)
        pred_bin = (probs >= prob_threshold).astype(np.int32)
        true_raw = np.array(true_signals[key], dtype=np.float32)
        true_val = np.log10(1 + true_raw)
        true_bin = (true_raw >= true_threshold).astype(np.int32)

        all_probs.append(probs)
        all_pred_bin.append(pred_bin)
        all_true_val.append(true_val)
        all_true_bin.append(true_bin)
        keys_list.append(key)

        for i in range(len(probs)):
            bed_start = start0 + i
            bed_end   = bed_start + 1
            f.write(
                f"{chrom}\t{bed_start}\t{bed_end}\t"
                f"{probs[i]:.6f}\t{pred_bin[i]}\t"
                f"{true_val[i]:.6f}\t{true_bin[i]}\n"
            )

    f.close()
    print(f"Saved per-base BED → {per_base_bed}")

    combined_npz = osp.join(outdir, "all_sequences.npz")
    np.savez(
        combined_npz,
        pred_prob=np.array(all_probs,    dtype=object),
        pred_bin =np.array(all_pred_bin, dtype=object),
        true_val =np.array(all_true_val, dtype=object),
        true_bin =np.array(all_true_bin, dtype=object),
        keys     =np.array(keys_list)
    )
    print(f"Saved combined NPZ → {combined_npz}")


#  Plotting 

def lineplot_binary(signals_prob, signals_binary, true_signals, atac_signals,
                    name, outdir, thres, true_threshold, smooth_window=15,
                    true_signals_rep=None, atac_signals_rep=None):
    for key in signals_prob.keys():
        prob   = np.array(signals_prob[key])
        binary = np.array(signals_binary[key])
        x      = np.arange(len(prob))

        true_raw    = np.nan_to_num(np.array(true_signals.get(key, np.zeros_like(prob))))
        true_log    = np.log10(1 + true_raw)
        true_smooth = Smoothavg(true_log, smooth_window)

        # --- True signal replicate ---
        has_rep = true_signals_rep is not None and key in true_signals_rep
        if has_rep:
            rep_raw    = np.nan_to_num(np.array(true_signals_rep[key], dtype=float))
            rep_log    = np.log10(1 + rep_raw)
            rep_smooth = Smoothavg(rep_log, smooth_window)

        # --- ATAC signal ---
        atac_raw_arr = atac_signals.get(key, None)
        has_atac     = atac_raw_arr is not None and len(atac_raw_arr) > 0
        if has_atac:
            atac_raw_arr = np.nan_to_num(np.array(atac_raw_arr, dtype=float))
            atac_log     = np.log10(1 + atac_raw_arr)
            atac_smooth  = Smoothavg(atac_log, smooth_window)

        # --- ATAC replicate signal ---
        atac_rep_raw_arr = atac_signals_rep.get(key, None) if atac_signals_rep is not None else None
        has_atac_rep     = atac_rep_raw_arr is not None and len(atac_rep_raw_arr) > 0
        if has_atac_rep:
            atac_rep_raw_arr = np.nan_to_num(np.array(atac_rep_raw_arr, dtype=float))
            atac_rep_log     = np.log10(1 + atac_rep_raw_arr)
            atac_rep_smooth  = Smoothavg(atac_rep_log, smooth_window)

        # Row count: true + rep? + atac? + atac_rep? + prob + binary
        n_rows = 3 + int(has_rep) + int(has_atac) + int(has_atac_rep)
        fig, ax = plt.subplots(n_rows, 1, figsize=(12, 2.8 * n_rows), sharex=True)
        fig.suptitle(key, fontsize=14, fontweight='bold')

        row = 0  # dynamic row index

        # Row 0: True signal
        ax[row].fill_between(x, true_log, alpha=0.25, color='black', label='')
        ax[row].plot(x, true_smooth, color='black', linewidth=1.3,
                     label=f'Smoothed (w={smooth_window})')
        ax[row].axhline(true_threshold, color='red', linestyle='--',
                        label=f'Threshold = {true_threshold}')
        ax[row].set_ylabel('pro-B EBF1 ChIP signal', fontweight='bold')
        ax[row].legend(prop={'weight': 'bold'})
        row += 1

        #  True signal replicate — directly below true signal
        if has_rep:
            ax[row].fill_between(x, rep_log, alpha=0.25, color='pink',
                                 label='Rep ')
            ax[row].plot(x, rep_smooth, color='navy', linewidth=1.3,
                         label=f'Smoothed (w={smooth_window})')
            ax[row].axhline(true_threshold, color='red', linestyle='--',
                            label=f'Threshold = {true_threshold}')
            ax[row].set_ylabel('3T3L1 EBF1 ChIP signal', fontweight='bold')
            ax[row].legend(prop={'weight': 'bold'})
            row += 1

        #  ATAC signal
        if has_atac:
            ax[row].fill_between(x, atac_log, alpha=0.25, color='forestgreen',
                                 label='ATAC ')
            ax[row].plot(x, atac_smooth, color='darkgreen', linewidth=1.3,
                         label=f'Smoothed (w={smooth_window})')
            ax[row].set_ylabel('pro-B ATAC', fontweight='bold')
            ax[row].legend(prop={'weight': 'bold'})
            row += 1

        #  ATAC replicate signal — directly below ATAC
        if has_atac_rep:
            ax[row].fill_between(x, atac_rep_log, alpha=0.25, color='yellowgreen',
                                 label='ATAC rep ')
            ax[row].plot(x, atac_rep_smooth, color='seagreen', linewidth=1.3,
                         label=f'Smoothed (w={smooth_window})')
            ax[row].set_ylabel('3T3L1 ATAC', fontweight='bold')
            ax[row].legend(prop={'weight': 'bold'})
            row += 1

        # Predicted probability
        ax[row].plot(x, prob, color='orange', label='Predicted Probability')
        ax[row].axhline(thres, color='red', linestyle='--', label=f'Threshold = {thres}')
        ax[row].set_ylabel('Predicted\nProbability', fontweight='bold')
        ax[row].set_ylim(-0.05, 1.15)
        ax[row].legend(prop={'weight': 'bold'})
        row += 1

        # Predicted binary
        ax[row].step(x, binary, where='mid', color='blue', label='Binary (0/1)')
        ax[row].set_ylabel('Predicted\nBinary', fontweight='bold')
        ax[row].legend(prop={'weight': 'bold'})

        ax[-1].set_xlabel('Position', fontweight='bold')
        for axis in ax:
            axis.title.set_fontweight('bold')
            for label in axis.get_xticklabels() + axis.get_yticklabels():
                label.set_fontweight('bold')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
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
    parser = argparse.ArgumentParser(
        description="FCNsignal binary classification (base resolution)")
    parser.add_argument("-i",  dest="infile",     type=str, required=True,
                        help="Input BED file")
    parser.add_argument("-n",  dest="name",        type=str, default="EBF",
                        help="Name of data/model")
    parser.add_argument("-g",  dest="gpu",         type=str, default='0',
                        help="GPU device (or -1 for CPU)")
    parser.add_argument("-t",  dest="threshold",   type=float, default=0.5,
                        help="Threshold for binary classification")
    parser.add_argument("-w",  dest="window",      type=int, default=15,
                        help="Length of binding region")
    parser.add_argument("-c",  dest="checkpoint",  type=str, default='./models/',
                        help="Model checkpoint folder")
    parser.add_argument("--features",     nargs='*', default=None,
                        help="Paths to bigWig files for additional features")
    parser.add_argument("--features_rep", nargs='*', default=None,
                        help="Paths to bigWig files for additional features (replicate)")
    parser.add_argument("--smooth",       action='store_true',
                        help="Apply smoothing to features")
    parser.add_argument("--chrom_sizes",  type=str,
                        default="Genome/mm9.chrom.sizes",
                        help="Chromosome sizes file")
    return parser.parse_args()


#  Main 

def main():
    args   = get_args()
    root   = osp.dirname(__file__)
    name   = args.name
    threshold = args.threshold
    window    = args.window

    device = torch.device(
        "cuda:" + args.gpu
        if torch.cuda.is_available() and args.gpu != '-1'
        else "cpu"
    )

    bw_truth_rep   = pyBigWig.open("EBF/adipose.bw")
    bw_truth = pyBigWig.open("EBF/p-value.bw")
    model_file = osp.join(args.checkpoint, "model_best_ebf.pth")
    outdir     = osp.join(root, "Results/test4_bcell")
    os.makedirs(outdir, exist_ok=True)

    sequence_dict = SeqIO.to_dict(
        SeqIO.parse(open(osp.join(root, "Genome/mm9.fa")), 'fasta'))
    with open(args.infile) as f:
        beds = f.readlines()

    chk        = torch.load(model_file, map_location=device)
    state_dict = chk['model_state_dict']

    n_features = len(args.features) if args.features else 0
    model = FCNsignalMod(motiflen=12, n_features=n_features)
    model.load_state_dict(state_dict)
    model.to(device)

    parameters = {
        'Window': window, 'Name': name,
        'Threshold': threshold, 'Device': device,
        'Feature_Files': args.features
    }

    beds_pos, signals_prob, signals_binary, signals_true, atac_signals, atac_signals_rep = locate_binary(
        parameters, sequence_dict, model, beds, outdir, bw_truth,
        feature_files=args.features, smooth=args.smooth,
        chrom_sizes=load_chrom_sizes(args.chrom_sizes),
        feature_files_rep=args.features_rep
    )

    true_signals_rep = {}
    for line in beds:
        bed   = line.strip().split()
        chrom, start, end = bed[0], int(bed[1]), int(bed[2])
        key   = f"{chrom}:{start}-{end}"
        true_signals_rep[key] = np.nan_to_num(bw_truth_rep.values(chrom, start, end))

    bw_truth_rep.close()

    save_per_base_true_values(
        signals_prob, signals_binary, signals_true, outdir,
        prob_threshold=0.5, true_threshold=0.1
    )

    with open(osp.join(outdir, 'binding_binary.bed'), 'w') as f:
        for chrom, start, end in beds_pos:
            f.write(f"{chrom}\t{start}\t{end}\n")

    lineplot_binary(
        signals_prob, signals_binary, signals_true, atac_signals,
        name, outdir, threshold, true_threshold=0.1, smooth_window=15,
        true_signals_rep=true_signals_rep,
        atac_signals_rep=atac_signals_rep
    )

    print("Binary classification complete")


if __name__ == "__main__":
    main()