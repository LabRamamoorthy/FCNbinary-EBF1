#!/usr/bin/python


import argparse
import os
import os.path as osp
import numpy as np
import torch
import pyBigWig
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from Bio import SeqIO
from transformers import AutoTokenizer

from fused_model import FusedModel

torch.multiprocessing.set_sharing_strategy('file_system')
plt.rcParams["font.family"] = "serif"


#  helpers 

def smooth_avg(x, window):
    pad = window // 2
    x_padded = np.pad(x, (pad, pad), mode='edge')
    out = []
    for j in range(len(x_padded) - window + 1):
        out.append(np.mean(x_padded[j: j + window]))
    return np.array(out)[: len(x)]


def get_bigwig(bw_file, chrom, start, end):
    with pyBigWig.open(bw_file) as bw:
        vals = np.array(bw.values(chrom, start, end), dtype=np.float64)
    vals[np.isnan(vals)] = 0.0
    return vals


def load_chrom_sizes(path):
    sizes = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                sizes[parts[0]] = int(parts[1])
    return sizes


#  one-hot encoding 

def one_hot_encode(sequence):
    mapping = {
        'A': [1, 0, 0, 0], 'a': [1, 0, 0, 0],
        'C': [0, 1, 0, 0], 'c': [0, 1, 0, 0],
        'G': [0, 0, 1, 0], 'g': [0, 0, 1, 0],
        'T': [0, 0, 0, 1], 't': [0, 0, 0, 1],
    }
    enc = np.array([mapping.get(c, [0, 0, 0, 0]) for c in sequence],
                   dtype=np.float32)   # [L, 4]
    return enc.T                        # [4, L]


#  BigWig feature extraction 

def extract_bw_feature(bw_file, chrom, start, end, seq_len,
                        chrom_sizes, smooth, window):
   
    clen = chrom_sizes.get(chrom, end + window)
    if smooth:
        s = max(0, start - window // 2)
        e = min(end + window // 2, clen)
        raw = get_bigwig(bw_file, chrom, s, e)
        sig = smooth_avg(raw, window)
    else:
        s = max(0, start)
        e = min(end, clen)
        sig = get_bigwig(bw_file, chrom, s, e)

    if len(sig) < seq_len:
        sig = np.pad(sig, (0, seq_len - len(sig)))
    else:
        sig = sig[:seq_len]

    return np.log10(1.0 + sig).astype(np.float32)


def build_fcn_input(seq, chrom, start, end, feature_files,
                     chrom_sizes, smooth, window, device):
    
    seq_len = end - start
    oh = one_hot_encode(seq)       # [4, L]
    parts = [oh]

    for bw_file in feature_files:
        feat = extract_bw_feature(bw_file, chrom, start, end,
                                   seq_len, chrom_sizes, smooth, window)
        parts.append(feat[np.newaxis, :])   # [1, L]

    fcn_input = np.concatenate(parts, axis=0)   # [4+F, L]
    return torch.from_numpy(fcn_input[np.newaxis]).float().to(device)  # [1, 4+F, L]


#  BERT tokenisation 

def tokenise_sequence(seq, tokenizer, max_len, device):
    
    enc = tokenizer(
        seq,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
        return_offsets_mapping=True,
    )
    return {
        'input_ids':      enc['input_ids'].to(device),
        'attention_mask': enc['attention_mask'].to(device),
        'offset_mapping': enc['offset_mapping'].to(device),
    }


#  nucleotide alignment 

def align_to_nucleotides(fused_logits, offset_mapping_cpu, seq_len, key):
    
    nuc_probs = np.zeros(seq_len, dtype=np.float32)

    tok_idx = 0  # index into fused_logits (CLS token at offset pos 0 is skipped)
    for t in range(1, len(offset_mapping_cpu)):
        char_start = int(offset_mapping_cpu[t][0])
        char_end   = int(offset_mapping_cpu[t][1])

        if char_start == 0 and char_end == 0:
            break   # padding sentinel — no more real tokens

        if tok_idx >= len(fused_logits):
            # Sequence was longer than bert_max_len tokens allowed
            print(f"  WARNING [{key}]: BERT coverage ended at nucleotide "
                  f"{char_start}/{seq_len}. "
                  f"Remaining positions set to 0. "
                  f"Consider increasing --bert_max_len.")
            break

        prob = float(torch.sigmoid(torch.tensor(fused_logits[tok_idx])).item())

        clip_s = max(0, char_start)
        clip_e = min(seq_len, char_end)
        if clip_s < clip_e:
            nuc_probs[clip_s:clip_e] = prob

        tok_idx += 1

    return nuc_probs


#  inference loop 

def run_inference(beds, sequence_dict, model, tokenizer,
                  bw_truth, feature_files,
                  chrom_sizes, smooth, window,
                  threshold, bert_max_len, device, outdir):
    
    model.eval()

    nuc_probs    = {}
    nuc_binary   = {}
    true_signals = {}
    atac_signals = {}

    f_prob = open(osp.join(outdir, 'predictions_prob.txt'),   'w')
    f_bin  = open(osp.join(outdir, 'predictions_binary.txt'), 'w')

    total_pos = 0

    for line in beds:
        parts = line.strip().split()
        chrom, start, end = parts[0], int(parts[1]), int(parts[2])
        seq_len = end - start
        key = f"{chrom}:{start}-{end}"

        #  sequence 
        seq = str(sequence_dict[chrom].seq[start:end])

        #  FCN input  [1, 4+F, L] 
        fcn_input = build_fcn_input(seq, chrom, start, end,
                                     feature_files, chrom_sizes,
                                     smooth, window, device)

        #  BERT input 
        bert_enc = tokenise_sequence(seq, tokenizer, bert_max_len, device)

        #  forward pass 
        with torch.no_grad():
            _logits_seq, fused_logits_list = model(
                fcn_input,
                bert_enc['input_ids'],
                bert_enc['attention_mask'],
                bert_enc['offset_mapping'],
            )

        # fused_logits_list[0]: [L_fused, 1]
        # L_fused = nucleotides covered by non-truncated BPE tokens
        raw_logits     = fused_logits_list[0].squeeze(-1).cpu().numpy()  # [L_fused]
        offset_map_cpu = bert_enc['offset_mapping'][0].cpu().numpy()     # [T, 2]

        # expand BPE tokens → per-nucleotide probabilities
        probs  = align_to_nucleotides(raw_logits, offset_map_cpu, seq_len, key)
        binary = (probs >= threshold).astype(np.int8)

        total_pos += int(binary.sum())

        f_prob.write('\t'.join(f"{v:.4f}" for v in probs)   + '\n')
        f_bin.write( '\t'.join(str(int(v)) for v in binary) + '\n')

        nuc_probs[key]  = probs
        nuc_binary[key] = binary

        #  truth BigWig 
        true_raw = np.nan_to_num(
            np.array(bw_truth.values(chrom, start, end), dtype=np.float64)
        )
        true_signals[key] = true_raw

        #  ATAC (first feature file, raw values) 
        if feature_files:
            atac_raw = get_bigwig(feature_files[0], chrom, start, end)
            if len(atac_raw) < seq_len:
                atac_raw = np.pad(atac_raw, (0, seq_len - len(atac_raw)))
            else:
                atac_raw = atac_raw[:seq_len]
            atac_signals[key] = atac_raw
        else:
            atac_signals[key] = None

    f_prob.close()
    f_bin.close()

    print(f"Total positive nucleotides (binary=1): {total_pos}")
    return nuc_probs, nuc_binary, true_signals, atac_signals


#  save per-base BED + NPZ 

def save_per_base(nuc_probs, nuc_binary, true_signals,
                   outdir, prob_threshold=0.5, true_threshold=0.1):
    os.makedirs(outdir, exist_ok=True)
    bed_path = osp.join(outdir, 'per_base_values.bed')

    all_probs    = []
    all_pred_bin = []
    all_true_val = []
    all_true_bin = []
    keys_list    = []

    with open(bed_path, 'w') as f:
        for key, probs in nuc_probs.items():
            chrom, coords = key.split(':')
            s = int(coords.split('-')[0])

            probs    = np.array(probs,             dtype=np.float32)
            pred_bin = (probs >= prob_threshold).astype(np.int32)
            true_raw = np.array(true_signals[key], dtype=np.float32)
            true_val = np.log10(1.0 + true_raw)
            true_bin = (true_raw >= true_threshold).astype(np.int32)

            all_probs.append(probs)
            all_pred_bin.append(pred_bin)
            all_true_val.append(true_val)
            all_true_bin.append(true_bin)
            keys_list.append(key)

            for i in range(len(probs)):
                f.write(
                    f"{chrom}\t{s+i}\t{s+i+1}\t"
                    f"{probs[i]:.6f}\t{pred_bin[i]}\t"
                    f"{true_val[i]:.6f}\t{true_bin[i]}\n"
                )

    print(f"Saved per-base BED → {bed_path}")

    npz_path = osp.join(outdir, 'all_sequences.npz')
    np.savez(
        npz_path,
        pred_prob=np.array(all_probs,    dtype=object),
        pred_bin =np.array(all_pred_bin, dtype=object),
        true_val =np.array(all_true_val, dtype=object),
        true_bin =np.array(all_true_bin, dtype=object),
        keys     =np.array(keys_list),
    )
    print(f"Saved combined NPZ → {npz_path}")


#  plotting 

def lineplot_fused(nuc_probs, nuc_binary, true_signals, atac_signals,
                    outdir, threshold, true_threshold, smooth_window=15):
    for key, probs in nuc_probs.items():
        probs  = np.array(probs,           dtype=np.float64)
        binary = np.array(nuc_binary[key], dtype=np.float64)
        x      = np.arange(len(probs))

        #  true signal 
        true_raw    = np.nan_to_num(
            np.array(true_signals.get(key, np.zeros_like(probs))))
        true_log    = np.log10(1.0 + true_raw)
        true_smooth = smooth_avg(true_log, smooth_window)

        #  ATAC 
        atac_arr = atac_signals.get(key, None)
        has_atac = atac_arr is not None and len(atac_arr) > 0
        if has_atac:
            atac_arr    = np.nan_to_num(np.array(atac_arr, dtype=np.float64))
            atac_log    = np.log10(1.0 + atac_arr)
            atac_smooth = smooth_avg(atac_log, smooth_window)

        n_rows = 4 if has_atac else 3
        fig, ax = plt.subplots(n_rows, 1,
                                figsize=(12, 2.8 * n_rows),
                                sharex=True)
        fig.suptitle(key, fontsize=14, fontweight='bold')

        row = 0  
        #  True signal 
        ax[row].fill_between(x, true_log, alpha=0.25, color='black',
                             label='log₁₀(1+x)')
        ax[row].plot(x, true_smooth, color='black', linewidth=1.3,
                     label=f'Smoothed (w={smooth_window})')
        ax[row].axhline(true_threshold, color='red', linestyle='--',
                        label=f'Threshold = {true_threshold}')
        ax[row].set_ylabel('True signal\nlog₁₀(1+x)', fontweight='bold')
        ax[row].legend(prop={'weight': 'bold'})
        row += 1

        if has_atac:   #  ATAC 
            ax[row].fill_between(x, atac_log, alpha=0.25, color='forestgreen',
                                 label='ATAC log₁₀(1+x)')
            ax[row].plot(x, atac_smooth, color='darkgreen', linewidth=1.3,
                         label=f'Smoothed (w={smooth_window})')
            ax[row].set_ylabel('ATAC\nlog₁₀(1+x)', fontweight='bold')
            ax[row].legend(prop={'weight': 'bold'})
            row += 1

        #  Per-nucleotide probability 
        ax[row].plot(x, probs, color='orange', linewidth=1.0,
                     label='Predicted probability')
        ax[row].axhline(threshold, color='red', linestyle='--',
                        label=f'Threshold = {threshold}')
        ax[row].set_ylabel('Predicted\nProbability', fontweight='bold')
        ax[row].set_ylim(-0.05, 1.15)
        ax[row].legend(prop={'weight': 'bold'})
        row += 1

        #  Per-nucleotide binary 
        ax[row].step(x, binary, where='mid', color='blue',
                     label='Binary (0/1)')
        ax[row].set_ylabel('Predicted\nBinary', fontweight='bold')
        ax[row].legend(prop={'weight': 'bold'})

        ax[-1].set_xlabel('Position (bp)', fontweight='bold')

        for axis in ax:
            for lbl in axis.get_xticklabels() + axis.get_yticklabels():
                lbl.set_fontweight('bold')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        safe_key = key.replace(':', '_').replace('-', '_')
        for ext in ('pdf', 'png'):
            plt.savefig(osp.join(outdir, f"{safe_key}_fused_plot.{ext}"),
                        format=ext, dpi=150)
        plt.close()
        print(f"  Plot saved → {safe_key}_fused_plot.{{pdf,png}}")


#  CLI 

def get_args():
    p = argparse.ArgumentParser(
        description="FusedModel per-nucleotide inference + plotting (BED input)")

    p.add_argument('-i',  dest='infile',   required=True,
                   help='Input BED file')
    p.add_argument('-g',  dest='genome',   required=True,
                   help='FASTA genome file (e.g. Genome/mm9.fa)')
    p.add_argument('--truth_bw',    required=True,
                   help='Ground-truth BigWig file')
    p.add_argument('--chrom_sizes', required=True,
                   help='Chromosome sizes file')
    p.add_argument('--checkpoint',  required=True,
                   help='FusedModel checkpoint (.pth)')
    p.add_argument('--tokenizer',
                   default='/home/ubuntu/data/dnabert2',
                   help='DNABERT-2 tokenizer directory')
    p.add_argument('-o', dest='outdir',
                   default='Results/fused_output',
                   help='Output directory')
    p.add_argument('-t', dest='threshold', type=float, default=0.5,
                   help='Probability threshold for binary calls')
    p.add_argument('--true_threshold', type=float, default=0.1,
                   help='Log10-signal threshold for ground-truth binary')
    p.add_argument('--features', nargs='*', default=[],
                   help='BigWig feature files (e.g. ATAC)')
    p.add_argument('--smooth',    action='store_true', default=True)
    p.add_argument('--no_smooth', dest='smooth', action='store_false')
    p.add_argument('--smooth_window', type=int, default=15)
    p.add_argument('--bert_max_len',  type=int, default=512,
                   help='Max BPE token length for BERT')
    p.add_argument('--motiflen', type=int, default=10,
                   help='motiflen for FCNsignalMod inside FusedModel')
    p.add_argument('--gpu', default='0',
                   help='GPU index (or -1 for CPU)')
    return p.parse_args()


#  main 

def main():
    args = get_args()

    device = torch.device(
        f"cuda:{args.gpu}"
        if torch.cuda.is_available() and args.gpu != '-1'
        else "cpu"
    )
    print(f"Using device: {device}")
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading genome: {args.genome}")
    sequence_dict = SeqIO.to_dict(SeqIO.parse(open(args.genome), 'fasta'))

    with open(args.infile) as f:
        beds = [l for l in f if not l.startswith('#') and l.strip()]
    print(f"Loaded {len(beds)} BED regions")

    chrom_sizes = load_chrom_sizes(args.chrom_sizes)

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer,
                                              trust_remote_code=True)

    n_features = len(args.features) if args.features else 0
    model = FusedModel(motiflen=args.motiflen, n_features=n_features)
    chk = torch.load(args.checkpoint, map_location=device)
    state_dict = chk.get('model_state_dict', chk)
    model.load_state_dict(state_dict)
    model.to(device)
    print(f"Loaded FusedModel from {args.checkpoint}")

    bw_truth = pyBigWig.open(args.truth_bw)

    nuc_probs, nuc_binary, true_signals, atac_signals = run_inference(
        beds, sequence_dict, model, tokenizer,
        bw_truth, args.features,
        chrom_sizes, args.smooth, args.smooth_window,
        args.threshold, args.bert_max_len, device, args.outdir,
    )

    bw_truth.close()

    save_per_base(
        nuc_probs, nuc_binary, true_signals,
        args.outdir,
        prob_threshold=args.threshold,
        true_threshold=args.true_threshold,
    )

    binding_bed = osp.join(args.outdir, 'binding_binary.bed')
    with open(binding_bed, 'w') as f:
        for key, binary in nuc_binary.items():
            chrom, coords = key.split(':')
            start = int(coords.split('-')[0])
            for i, b in enumerate(binary):
                if b:
                    f.write(f"{chrom}\t{start+i}\t{start+i+1}\n")
    print(f"Binding BED → {binding_bed}")

    lineplot_fused(
        nuc_probs, nuc_binary, true_signals, atac_signals,
        args.outdir, args.threshold, args.true_threshold,
        smooth_window=args.smooth_window,
    )

    print("Done.")


if __name__ == "__main__":
    main()