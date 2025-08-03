import numpy as np
from Bio import SeqIO
import pyBigWig
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from FCNmotif import FCNsignalMod
from tqdm import tqdm
import argparse
from scipy.stats import pearsonr

SEQ_LEN = 200
FLANK = 100
INDEX = ['chr' + str(i + 1) for i in range(23)] + ['chrX']

def one_hot(sequence_dict, chrom, start, end):
    seq_dict = {'A':[1,0,0,0], 'C':[0,1,0,0], 'G':[0,0,1,0], 'T':[0,0,0,1],
                'a':[1,0,0,0], 'c':[0,1,0,0], 'g':[0,0,1,0], 't':[0,0,0,1]}
    seq = str(sequence_dict[chrom].seq[start:end])
    return [seq_dict.get(base, [0,0,0,0]) for base in seq]

def extract_true_data(bed_file, genome_fa, bw_file, output_path):
    sequence_dict = SeqIO.to_dict(SeqIO.parse(open(genome_fa), 'fasta'))
    bw = pyBigWig.open(bw_file)

    seqs, signals = [], []
    with open(bed_file) as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Extracting true values"):
        chrom, start, end = line.strip().split()[:3]
        if chrom not in INDEX: continue
        start, end = int(start), int(end)
        center = (start + end) // 2
        start_pos = center - FLANK
        end_pos = center + FLANK

        if start_pos < 0 or end_pos > len(sequence_dict[chrom]):
            continue

        signal = bw.values(chrom, start_pos, end_pos)
        signal = np.array([0 if np.isnan(x) else x for x in signal])
        signal_avg = np.mean(signal)

        seq = one_hot(sequence_dict, chrom, start_pos, end_pos)
        seqs.append(seq)
        signals.append(signal)

    bw.close()
    seqs = np.array(seqs, dtype=np.float32).transpose(0, 2, 1)
    signals = np.array(signals, dtype=np.float32)  
    labels = np.ones((len(seqs), 1), dtype=np.float32)
    np.savez(output_path, data=seqs, signal=signals, label=labels)
    print(f"Saved {len(seqs)} sequences to {output_path}")

def get_fcn_predictions(model_path, npz_file, output_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    print(f"Loading data from: {npz_file}")
    data = np.load(npz_file)
    inputs = torch.tensor(data['data'], dtype=torch.float32).to(device)
    true_signals = data['signal'].squeeze()

    num_sequences = len(inputs)
    print(f"Number of sequences in input data: {num_sequences}")

    model = FCNsignalMod()
    state_dict = torch.load(model_path, map_location=device)
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    preds = []
    batch_size = 128
    with torch.no_grad():
        for i in tqdm(range(0, num_sequences, batch_size), desc="Predicting"):
            batch = inputs[i:i+batch_size]
            out = model(batch).cpu().numpy()
            preds.append(out)

    pred_signals = np.concatenate(preds, axis=0)
    print(f"Number of sequences predicted: {len(pred_signals)}")

    if output_path:
        print(f"Saving predictions to: {output_path}")
        np.savez(output_path, pred_signal=pred_signals, true_signal=true_signals)

    return true_signals, pred_signals

def plot_max_scatter(true_signals, pred_signals, out='scatter_max.png'):
    # Calculate max values for each sequence
    true_signals = np.log10(1 + true_signals)
    true_max = np.max(true_signals, axis=1)
    pred_max = np.max(pred_signals, axis=1)

    # Pearson correlation
    overall_pearson, _ = pearsonr(true_max, pred_max)

    # Create scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(pred_max, true_max, s=2, alpha=0.5)
    
    # Regression line 
    fit = np.polyfit(pred_max, true_max, 1)
    fit_fn = np.poly1d(fit)
    plt.plot(pred_max, fit_fn(pred_max), '--', color='black')
    plt.xlabel('Predicted Maximum Values')
    plt.ylabel('True Maximum Values')
    plt.xlim(0, np.max(pred_max)+0.5)
    plt.ylim(0, np.max(true_max)+0.5)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"Max scatter plot saved to {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--plot', action='store_true')
    
    parser.add_argument('--bed', type=str)
    parser.add_argument('--fasta', type=str)
    parser.add_argument('--bw', type=str)
    parser.add_argument('--output', type=str, default='true_value.npz')
    parser.add_argument('--model', type=str)
    parser.add_argument('--npz', type=str)
    parser.add_argument('--scatter', type=str, default='scatter_colored.png')

    args = parser.parse_args()

    if args.generate:
        extract_true_data(args.bed, args.fasta, args.bw, args.output)

    if args.predict:
        if not args.model or not args.npz:
            print("Need both --model and --npz for prediction")
        else:
            true_vals, pred_vals = get_fcn_predictions(args.model, args.npz)
            np.savez('predicted.npz', true=true_vals, pred=pred_vals)

    if args.plot:
        data = np.load(args.npz)
        true_vals, pred_vals = data['true'], data['pred']
        plot_max_scatter(true_vals, pred_vals, out=args.scatter)

