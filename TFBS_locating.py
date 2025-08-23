#!/usr/bin/python

import argparse
import os
import sys
import numpy as np
import os.path as osp
from Bio import SeqIO
from scipy.stats import pearsonr
import torch
# custom functions defined by user
from FCNmotif import FCNsignalMod
torch.multiprocessing.set_sharing_strategy('file_system')
import pandas as pd
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pyBigWig
    
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

            # Clamp coordinates
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

def lineplot(signals, true_signals, name, outdir, thres, smooth_window=5):
    for key in signals.keys():
        signal = np.array(signals[key])
        true_signal = np.array(true_signals.get(key, np.zeros_like(signal)))
        # if smooth_window > 1:
        #     signal = Smoothavg(signal, smooth_window)
        true_signal = np.log10(1 + true_signal)

        segment_len = 1000
        length = len(signal)
        n = int(np.floor(length / segment_len))
        positions_x = []
        positions_y = []

        for i in range(n):
            segment = signal[i * segment_len:(i + 1) * segment_len]
            position = np.argmax(segment) + i * segment_len
            Max = np.max(segment)
            if Max > thres:
                positions_x.append(position)
                positions_y.append(np.min(segment) + 0.2)

        x = np.arange(length)

        # Pearson correlation
        if len(signal) == len(true_signal) and len(signal) > 1:
            r, _ = pearsonr(signal, true_signal)
            r_text = f"Pearson r = {r:.3f}"
        else:
            r_text = "Pearson r = N/A"

        # Create vertically stacked subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        # Prediction plot
        ax1.plot(x, signal, label="Prediction", color='orange')
        for position_x, position_y in zip(positions_x, positions_y):
            ax1.axvline(x=position_x, ls="--", c="green")
            ax1.text(position_x, position_y, str(position_x), fontsize=7)
        ax1.set_ylabel("Prediction", fontsize=12)
        ax1.set_title(f"{key} | {r_text}", fontsize=13)
        ax1.tick_params(labelsize=10)

        # True signal plot (log-normalized)
        ax2.plot(x, true_signal, label="True Signal (log10)", color='black')
        ax2.set_ylabel("True", fontsize=12)
        ax2.set_xlabel("Position", fontsize=12)
        ax2.tick_params(labelsize=10)

        plt.tight_layout()
        plt.savefig(osp.join(outdir, f"{key}.png"), format='png')
        plt.close(fig)

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


def extract(signal, thres, Window):
    segment_len = 1000
    length = len(signal)
    n = int(np.ceil(length / segment_len))
    segments = []
    for i in range(n):
        segments.append(signal[i*segment_len:(i+1)*segment_len])
    starts = []
    ends = []
    for i in range(len(segments)):
        segment = segments[i]
        position = np.argmax(segment) + i*segment_len
        Max = np.max(segment)
        if Max > thres:
            start = int(position - Window // 2)
            end = int(position + Window // 2)
        else:
            start = 0
            end = 0
        starts.append(start)
        ends.append(end)
    return starts, ends


def locating(parameters, sequence_dict, model, beds, outdir, bw, feature_files=None, smooth=True, chrom_sizes=None):
    model.eval()
    beds_pos = []
    signals = {}
    true_signals = {}
    num_pos = 0
    f = open(osp.join(outdir, 'predictions.txt'), 'w')
    for i, line in enumerate(beds):
        bed = line.strip().split()
        chrom = bed[0]
        start = int(bed[1])
        end = int(bed[2])
        seq_len = end - start
        X_data = torch.from_numpy(one_hot(sequence_dict, chrom, start, end))
        X_data = X_data.float().to(parameters['Device'])
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
            signal_p = model(X_input)
        else:
            X_input = X_data
        with torch.no_grad():
            signal_p = model(X_input)
        signal_p = signal_p.view(-1).data.cpu().numpy()
        signal_true = np.nan_to_num(bw.values(chrom, start, end))
        f.write('\t'.join([str(round(x, 3)) for x in signal_p]) + '\n')
        signals["{}:{}-{}".format(chrom, start, end)] = signal_p
        true_signals["{}:{}-{}".format(chrom, start, end)] = signal_true
        starts, ends = extract(signal_p, parameters['Threshold'], parameters['Window'])
        for start_p, end_p in zip(starts, ends):
            if start_p == 0 and end_p == 0:
                continue
            else:
                start_p += start
                end_p += start
                num_pos += 1
                beds_pos.append((chrom, start_p, end_p))
    print("A total of {} potential binding regions are found.".format(num_pos))
    f.close()
    return beds_pos, signals , true_signals

def load_chrom_sizes(path):
    chrom_sizes = {}
    with open(path) as f:
        for line in f:
            chrom, size = line.strip().split()
            chrom_sizes[chrom] = int(size)
    return chrom_sizes

def get_args():
    parser = argparse.ArgumentParser(description="FCNsignal with optional features")
    parser.add_argument("-i", dest="infile", type=str, required=True,
                        help="Input BED file")
    parser.add_argument("-n", dest="name", type=str, default="ebf",
                        help="Name of the data/model")
    parser.add_argument("-g", dest="gpu", type=str, default='0',
                        help="GPU device (or -1 for CPU)")
    parser.add_argument("-t", dest="threshold", type=float, default=1.5,
                        help="Threshold value")
    parser.add_argument("-w", dest="window", type=int, default=60,
                        help="Length of binding regions")
    parser.add_argument("-c", dest="checkpoint", type=str, default='./models/',
                        help="Model checkpoint folder")
    parser.add_argument("--features", nargs='*', default=None,
                        help="Paths to bigWig files for additional features")
    parser.add_argument("--smooth", action='store_true',
                        help="Apply smoothing to additional features")
    parser.add_argument("--chrom_sizes", type=str, default="Genome/mm9.chrom.sizes",
                        help="Chromosome sizes file")
    
    return parser.parse_args()


def main():
    args = get_args()
    root = osp.dirname(__file__)
    # infile = osp.join(root, "input.bed")
    infile = args.infile
    name = args.name
    threshold = args.threshold
    window = args.window
    if torch.cuda.is_available() and args.gpu != '-1':
        device = torch.device("cuda:" + args.gpu)
    else:
        device = torch.device("cpu")
    bw_truth = pyBigWig.open("EBF/p-value.bw")

    # model_file = osp.join(root, "models/{}/model_best.pth".format(name))
    model_file = osp.join(args.checkpoint, "model_best_ebf.pth")
    outdir = osp.join(root, "output_Mod_withfeatures")
    if not osp.exists(outdir):
        os.mkdir(outdir)
    sequence_dict = SeqIO.to_dict(SeqIO.parse(open(osp.join(root, "Genome/mm9.fa")), 'fasta'))
    with open(infile) as f:
        beds = f.readlines()
    # Load weights
    chk = torch.load(model_file, map_location=device)
    state_dict = chk['model_state_dict']
    if args.features is not None and len(args.features) > 0:
        n_features = len(args.features)
        bw_list = [pyBigWig.open(f) for f in args.features]
    else:
        n_features = 0
        bw_list = []
    model = FCNsignalMod(motiflen=12, n_features=n_features)
    model.load_state_dict(state_dict)
    model.to(device)
    parameters = {'Window': window, 'Name': name, 'Threshold': threshold, 'Device': device, 'Feature_Files': args.features}
    # beds_pos, signal_p = locating(parameters, sequence_dict, model, beds, outdir)
    beds_pos, signal_p, signal_t = locating(parameters, sequence_dict, model, beds, outdir, bw_truth, feature_files=args.features, smooth=args.smooth, chrom_sizes=load_chrom_sizes(args.chrom_sizes))
    f = open(osp.join(outdir, 'binding.bed'), 'w')
    for bed in beds_pos:
        chrom = bed[0]
        start = bed[1]
        end = bed[2]
        f.write("{}\t{}\t{}\n".format(chrom, start, end))
    f.close()
    # lineplot(signal_p, name, outdir, parameters['Threshold'])
    lineplot(signal_p, signal_t, name, outdir, parameters['Threshold'])
    # Group Pearson correlation by sequence length bins and plot bar chart
    lengths, correlations = [], []
    for key in signal_p.keys():
        pred = np.array(signal_p[key])
        true = np.array(signal_t.get(key, np.zeros_like(pred)))
        length = int(key.split(":")[1].split("-")[1]) - int(key.split(":")[1].split("-")[0])
        if len(pred) == len(true) and len(pred) > 1:
            r, _ = pearsonr(pred, true)
            lengths.append(length)
            correlations.append(r)

    if lengths:
        df_corr = pd.DataFrame({'Length': lengths, 'Pearson_r': correlations})

        bin_size = 1000
        min_len, max_len = df_corr['Length'].min(), df_corr['Length'].max()
        bins = np.arange((min_len // bin_size) * bin_size,
                         ((max_len // bin_size) + 1) * bin_size + 1,
                         bin_size)
        labels = [f"{b}-{b+bin_size-1}" for b in bins[:-1]]
        df_corr['Length_bin'] = pd.cut(df_corr['Length'], bins=bins, labels=labels, include_lowest=True)

        grouped = df_corr.groupby('Length_bin', observed=True)['Pearson_r'].mean().reset_index()

        plt.figure(figsize=(10, 6))
        sns.barplot(data=grouped, x='Length_bin', y='Pearson_r', color='skyblue', edgecolor='black')
        plt.title("Mean Pearson Correlation by Sequence Length", fontsize=14)
        plt.xlabel("Sequence Length Bin (bp)", fontsize=12)
        plt.ylabel("Mean Pearson r", fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(osp.join(outdir, f"{name}_pearson_by_length_bins.png"), format='png')
        plt.close()
    else:
        print("No sequences found to calculate Pearson correlation for.")



if __name__ == "__main__":
    main()

