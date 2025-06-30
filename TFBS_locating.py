#!/usr/bin/python

import argparse
import os
import sys
import numpy as np
import os.path as osp
from Bio import SeqIO

import torch
# custom functions defined by user
from FCNmotif import FCNsignal
torch.multiprocessing.set_sharing_strategy('file_system')
import pandas as pd
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns

import pyBigWig

def lineplot(signals, true_signals, name, outdir, thres):
    for key in signals.keys():
        signal = signals[key]
        true_signal = true_signals.get(key, np.zeros_like(signal))
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

        # Create vertically stacked subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        sns.set_theme(style="whitegrid")

        # Prediction plot
        ax1.plot(x, signal, label="Prediction", color='blue')
        for position_x, position_y in zip(positions_x, positions_y):
            ax1.axvline(x=position_x, ls="--", c="green")
            ax1.text(position_x, position_y, str(position_x), fontsize=7)
        ax1.set_ylabel("Prediction", fontsize=12)
        ax1.tick_params(labelsize=10)

        # True signal plot
        ax2.plot(x, true_signal, label="True Signal", color='orange')
        ax2.set_ylabel("True", fontsize=12)
        ax2.set_xlabel(key, fontsize=12)
        ax2.tick_params(labelsize=10)

        # Save the plot
        plt.tight_layout()
        plt.savefig(osp.join(outdir, "{}.png".format(key)), format='png')
        plt.close(fig)

# def lineplot(signals, name, outdir, thres):
#     for key in signals.keys():
#         signal = signals[key]
#         segment_len = 1000
#         length = len(signal)
#         n = int(np.floor(length / segment_len))
#         positions_x = []
#         positions_y = []
#         for i in range(n):
#             segment = signal[i * segment_len:(i + 1) * segment_len]
#             position = np.argmax(segment) + i * segment_len
#             Max = np.max(segment)
#             if Max > thres:
#                 positions_x.append(position)
#                 positions_y.append(np.min(segment) + 0.2)
#         x = np.asarray(list(range(length))) + 1
#         df = pd.DataFrame({'x': x, 'Prediction': signal})
#         fig = plt.figure(figsize=(10, 6))
#         sns.set_theme(style="whitegrid")
#         sns.lineplot(x="x", y="Prediction", data=df)
#         for position_x, position_y in zip(positions_x, positions_y):
#             plt.axvline(x=position_x, ls="--", c="green")
#             plt.text(position_x, position_y, str(position_x))
#         plt.xlabel(key, fontsize=15)
#         plt.ylabel("Prediction", fontsize=15)
#         plt.tick_params(labelsize=12)
#         plt.title("{}".format(name), fontsize=18)

#         plt.savefig(osp.join(outdir, "{}.png".format(key)), format='png')
#         plt.close(fig)

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


def locating(parameters, sequence_dict, model, beds, outdir, bw):
# def locating(parameters, sequence_dict, model, beds, outdir):
    # loading model parameters
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
        X_data = torch.from_numpy(one_hot(sequence_dict, chrom, start, end))
        X_data = X_data.float().to(parameters['Device'])
        with torch.no_grad():
            signal_p = model(X_data)
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


def get_args():
    """Parse all the arguments.
        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="FCNsignal for locating binding regions")
    parser.add_argument("-i", dest="infile", type=str, default=None,
                        help="An input file in bed format.")
    parser.add_argument("-n", dest="name", type=str, default=None,
                        help="The name of a specified data.")
    parser.add_argument("-g", dest="gpu", type=str, default='0',
                        help="Choose gpu device.")
    parser.add_argument("-t", dest="threshold", type=float, default=1.5,
                        help="The threshold value.")
    parser.add_argument("-w", dest="window", type=float, default=60,
                        help="The length of binding regions.")
    parser.add_argument("-c", dest="checkpoint", type=str, default='./models/',
                        help="Where to save snapshots of the model.")

    return parser.parse_args()


args = get_args()


def main():
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
    bw = pyBigWig.open("EBF/p-value.bw")

    # model_file = osp.join(root, "models/{}/model_best.pth".format(name))
    model_file = osp.join(args.checkpoint, "model_best_ebf.pth")
    outdir = osp.join(root, "outputs_ebf")
    if not osp.exists(outdir):
        os.mkdir(outdir)
    sequence_dict = SeqIO.to_dict(SeqIO.parse(open(osp.join(root, "Genome/mm9.fa")), 'fasta'))
    with open(infile) as f:
        beds = f.readlines()
    # Load weights
    chk = torch.load(model_file, map_location=device)
    state_dict = chk['model_state_dict']
    model = FCNsignal(motiflen=12)
    model.load_state_dict(state_dict)
    model.to(device)
    parameters = {'Window': window, 'Name': name, 'Threshold': threshold, 'Device': device}
    # beds_pos, signal_p = locating(parameters, sequence_dict, model, beds, outdir)
    beds_pos, signal_p, signal_t = locating(parameters, sequence_dict, model, beds, outdir, bw)
    f = open(osp.join(outdir, 'binding.bed'), 'w')
    for bed in beds_pos:
        chrom = bed[0]
        start = bed[1]
        end = bed[2]
        f.write("{}\t{}\t{}\n".format(chrom, start, end))
    f.close()
    # lineplot(signal_p, name, outdir, parameters['Threshold'])
    lineplot(signal_p, signal_t, name, outdir, parameters['Threshold'])


if __name__ == "__main__":
    main()

