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

def smooth_avg(x, window):
    x = np.asarray(x, dtype=np.float32)
    if window <= 1 or len(x) < window:
        return x
    pad = (window - 1) // 2
    # reflect-pad to mitigate edge effects
    x_padded = np.pad(x, (pad, window - 1 - pad), mode='reflect')
    avg = []
    for j in range(len(x)):
        window_slice = x_padded[j:(j + window)]
        avg.append(np.mean(window_slice))
    return np.array(avg, dtype=np.float32)


def get_bigwig_vals(bw_handle, chrom, start, end):
    vals = bw_handle.values(chrom, start, end)
    vals = np.array([0. if np.isnan(x) else x for x in vals], dtype=np.float32)
    return vals

def extract_additional_features(
    bed_lines,
    feature_bw_handles,
    location_func,
    chrom_lengths,
    seq_len=SEQ_LEN,
    window=21,
    smooth=True
):
    """
    bed_lines: list of lines (bed entries)
    feature_bw_handles: list of opened pyBigWig handles
    location_func: function(chrom, start, end) -> (center_pos) or (start_f, end_f)
    chrom_lengths: dict mapping chromosome name to length
    Returns: numpy array of shape [N, F, SEQ_LEN]
    """
    all_feats = []
    for bw in feature_bw_handles:
        feat_signals = []
        for line in bed_lines:
            chrom, start_s, end_s = line.strip().split()[:3]
            if chrom not in chrom_lengths:
                continue
            start, end = int(start_s), int(end_s)
            start_f, end_f = location_func(chrom, start, end)

            # Determine windowed region: center around start_f/end_f with seq_len
            center = (start_f + end_f) // 2 if isinstance(start_f, int) and isinstance(end_f, int) else start_f
            half = seq_len // 2
            start_c = max(0, center - half)
            end_c = min(chrom_lengths[chrom], start_c + seq_len)
            # adjust if truncated at end
            if end_c - start_c < seq_len:
                start_c = max(0, end_c - seq_len)

            vals = get_bigwig_vals(bw, chrom, start_c, end_c)
            if smooth:
                vals = smooth_avg(vals, window)
            # pad/truncate to exactly seq_len
            if len(vals) < seq_len:
                pad_len = seq_len - len(vals)
                vals = np.pad(vals, (0, pad_len), mode='constant', constant_values=0)
            elif len(vals) > seq_len:
                vals = vals[:seq_len]

            feat_signals.append(vals)
        feat_signals = np.array(feat_signals, dtype=np.float32)  # [N, SEQ_LEN]
        feat_signals = np.log10(1 + feat_signals)
        all_feats.append(feat_signals)

    if not all_feats:
        return np.zeros((0, 0, seq_len), dtype=np.float32)
    all_feats = np.stack(all_feats, axis=0)  # [F, N, SEQ_LEN]
    all_feats = np.transpose(all_feats, (1, 0, 2))  # [N, F, SEQ_LEN]
    return all_feats

def extract_true_data(bed_file, genome_fa, bw_file, output_path, feature_bw_files=None):
    sequence_dict = SeqIO.to_dict(SeqIO.parse(open(genome_fa), 'fasta'))
    bw = pyBigWig.open(bw_file)

    seqs, signals, groups, bed_lines = [], [], [], []
    with open(bed_file) as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Extracting true values"):
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        chrom, start, end = parts[0], parts[1], parts[2]
        if chrom not in INDEX:
            continue
        start, end = int(start), int(end)
        center = (start + end) // 2
        start_pos = center - FLANK
        end_pos = center + FLANK
        if start_pos < 0 or end_pos > len(sequence_dict[chrom]):
            continue

        # Keep original bed line for later feature extraction
        bed_lines.append(line.strip())

        # Group assignment
        source_field = parts[3] if len(parts) > 3 else ''
        has3R = 'R2KOWT3R' in source_field
        has2 = 'R2KOWT2' in source_field
        if has3R and has2:
            group = 'common'
        elif has3R:
            group = 'R2KOWT3R'
        elif has2:
            group = 'R2KOWT2'
        else:
            group = 'unknown'

        signal = bw.values(chrom, start_pos, end_pos)
        signal = np.array([0 if np.isnan(x) else x for x in signal], dtype=np.float32)
        seq = one_hot(sequence_dict, chrom, start_pos, end_pos)

        seqs.append(seq)
        signals.append(signal)
        groups.append(group)

    bw.close()
    seqs = np.array(seqs, dtype=np.float32).transpose(0, 2, 1)
    signals = np.array(signals, dtype=np.float32)
    groups = np.array(groups, dtype=object)
    labels = np.ones((len(seqs), 1), dtype=np.float32)

    # Extract additional features if files provided
    if feature_bw_files:
        print(f"Extracting additional features from {len(feature_bw_files)} bigWig files...")
        feature_bw_handles = [pyBigWig.open(f) for f in feature_bw_files]
        chrom_lengths = {chrom: len(seq) for chrom, seq in sequence_dict.items()}

        def location_func(chrom, start, end):
            return start, end  # Here we just use the given start/end

        features = extract_additional_features(
            bed_lines, feature_bw_handles, location_func, chrom_lengths, seq_len=SEQ_LEN
        )
        for h in feature_bw_handles:
            h.close()
    else:
        features = np.zeros((len(seqs), 0, SEQ_LEN), dtype=np.float32)

    # Save everything
    np.savez(output_path, data=seqs, signal=signals, label=labels, group=groups, features=features)
    print(f"Saved {len(seqs)} sequences to {output_path} (features shape: {features.shape})")



def get_fcn_predictions(model_path, npz_file, output_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    print(f"Loading data from: {npz_file}")
    data = np.load(npz_file, allow_pickle=True)

    inputs = torch.tensor(data['data'], dtype=torch.float32).to(device)  # [N, 4, L]
    true_signals = data['signal'].squeeze()
    groups = data.get('group', None)  # could be None if not present

    # Load features if present
    if 'features' in data and data['features'].shape[1] > 0:
        features = torch.tensor(data['features'], dtype=torch.float32).to(device)  # [N, n_features, L]
        use_features = True
        print(f"Loaded additional features with shape: {features.shape}")
    else:
        features = None
        use_features = False
        print("No additional features found or empty.")

    num_sequences = len(inputs)
    print(f"Number of sequences in input data: {num_sequences}")

    # Make sure to pass correct n_features to your model
    n_features = features.shape[1] if use_features else 0
    model = FCNsignalMod(motiflen=12, n_features=n_features)

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
            batch_seq = inputs[i:i+batch_size]
            if use_features:
                batch_feat = features[i:i+batch_size]
                # Concatenate along channel dim (1)
                batch_combined = torch.cat([batch_seq, batch_feat], dim=1)  # shape: [batch, 6, length]
                out = model(batch_combined)
            else:
                out = model(batch_seq)
            preds.append(out.cpu().numpy())


    pred_signals = np.concatenate(preds, axis=0)
    print(f"Number of sequences predicted: {len(pred_signals)}")

    if output_path:
        print(f"Saving predictions to: {output_path}")
        save_dict = {'true': true_signals, 'pred': pred_signals}
        if groups is not None:
            save_dict['group'] = groups
        np.savez(output_path, **save_dict)

    return true_signals, pred_signals, groups


def plot_max_scatter(true_signals, pred_signals, groups=None, include_groups=None, out='scatter_filtered.png'):
    # Log-transform and get maxima
    true_signals = np.log10(1 + true_signals)
    true_max = np.max(true_signals, axis=1)
    pred_max = np.max(pred_signals, axis=1)

    # If include_groups is specified, filter both arrays (and groups) to only those
    if include_groups is not None:
        if groups is None:
            raise ValueError("include_groups specified but no groups array provided.")
        mask = np.isin(groups, include_groups)
        if np.sum(mask) == 0:
            raise ValueError(f"No examples found for groups {include_groups}")
        true_max = true_max[mask]
        pred_max = pred_max[mask]
        filtered_groups = groups[mask]
    else:
        filtered_groups = groups

    # Pearson correlation
    pearson_corr, _ = pearsonr(pred_max, true_max)
    print(f"Pearson correlation: {pearson_corr:.4f}")

    # Minimum predicted max value
    min_pred_max = np.min(pred_max)
    print(f"Minimum predicted max value: {min_pred_max:.4f}")

    plt.figure(figsize=(6, 6))
    color_map = {'R2KOWT3R': 'tab:blue', 'R2KOWT2': 'tab:orange', 'common': 'tab:green', 'unknown': 'tab:gray'}

    if filtered_groups is None:
        plt.scatter(pred_max, true_max, s=2, alpha=0.5, label='all')
    else:
        unique_groups = np.unique(filtered_groups)
        for grp in unique_groups:
            grp_mask = filtered_groups == grp
            pm = pred_max[grp_mask]
            tm = true_max[grp_mask]
            label = f"{grp} (n={len(pm)})"
            plt.scatter(pm, tm, s=2, alpha=0.5, label=label, color=color_map.get(grp, 'black'))

    # Regression line
    fit = np.polyfit(pred_max, true_max, 1)
    fit_fn = np.poly1d(fit)
    sorted_x = np.sort(pred_max)
    plt.plot(sorted_x, fit_fn(sorted_x), '--', color='black', label='regression')

    # Add Pearson in title
    plt.title(f"Max Scatter Plot (Pearson r = {pearson_corr:.4f})")

    plt.xlabel('Predicted Maximum Values')
    plt.ylabel('True Maximum Values')
    plt.xlim(0, np.max(pred_max) + 0.5)
    plt.ylim(0, np.max(true_max) + 0.5)
    plt.legend(markerscale=4, fontsize='small', frameon=True)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"Max scatter plot (filtered) saved to {out}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', action='store_true',help='To generate dataset')
    parser.add_argument('--predict', action='store_true',help='To predict for the generated dataset')
    parser.add_argument('--plot', action='store_true',help='To plot the scatter plot')

    parser.add_argument('--bed', type=str,help='Path to the bed file of the regions')
    parser.add_argument('--fasta', type=str,help='Path to the genome file')
    parser.add_argument('--bw', type=str,help='Path to the ture signal file')
    parser.add_argument('--output', type=str, default='true_value.npz',help='Path to the output file')
    parser.add_argument('--model', type=str,help='Path to the model checkpoint/path file')
    parser.add_argument('--npz', type=str,help='path to the output file')
    parser.add_argument('--scatter', type=str, default='scatter_colored.png',help='path to the output file')
    parser.add_argument('--feature_bw', nargs='*', default=None,
                        help='Optional additional bigWig feature files')

    args = parser.parse_args()

    if args.generate:
        if not (args.bed and args.fasta and args.bw):
            print("Need --bed, --fasta, and --bw for generation.")
        else:
            extract_true_data(
                args.bed,
                args.fasta,
                args.bw,
                args.output,
                feature_bw_files=args.feature_bw  # <-- new
            )

    if args.predict:
        if not args.model or not args.npz:
            print("Need both --model and --npz for prediction")
        else:
            print(f"Running prediction with model: {args.model} and npz: {args.npz}")
            true_vals, pred_vals, groups = get_fcn_predictions(
                args.model, args.npz, output_path='predicted.npz'
            )
            if groups is not None:
                np.savez('predicted.npz', true=true_vals, pred=pred_vals, group=groups)
            else:
                np.savez('predicted.npz', true=true_vals, pred=pred_vals)
            print("Prediction complete. Saved to predicted.npz")

    if args.plot:
        if not args.npz:
            print("Need --npz for plotting")
        else:
            data = np.load(args.npz, allow_pickle=True)
            true_vals = data.get('true') if 'true' in data else data.get('signal')
            pred_vals = data.get('pred') if 'pred' in data else data.get('pred_signal')
            groups = data.get('group', None)
            if true_vals is None or pred_vals is None:
                print("Could not find expected keys in npz for plotting.")
            else:
                plot_max_scatter(true_vals, pred_vals, groups=groups, out=args.scatter)



# import numpy as np
# from Bio import SeqIO
# import pyBigWig
# import os
# from tqdm import tqdm
# import torch
# import argparse
# from FCNmotif import FCNsignalMod
# import os.path as osp
# from scipy.stats import pearsonr
# SEQ_LEN = 200
# FLANK = 100
# INDEX = ['chr' + str(i + 1) for i in range(23)] + ['chrX']

# def one_hot(sequence_dict, chrom, start, end):
#     seq_map = {
#         'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1],
#         'a': [1,0,0,0], 'c': [0,1,0,0], 'g': [0,0,1,0], 't': [0,0,0,1]
#     }
#     seq = str(sequence_dict[chrom].seq[start:end])
#     return [seq_map.get(base, [0,0,0,0]) for base in seq]

# def smooth_avg(x, window):
#     x = np.asarray(x, dtype=np.float32)
#     if window <= 1 or len(x) < window:
#         return x
#     pad = (window - 1) // 2
#     # reflect-pad to mitigate edge effects
#     x_padded = np.pad(x, (pad, window - 1 - pad), mode='reflect')
#     avg = []
#     for j in range(len(x)):
#         window_slice = x_padded[j:(j + window)]
#         avg.append(np.mean(window_slice))
#     return np.array(avg, dtype=np.float32)


# def get_bigwig_vals(bw_handle, chrom, start, end):
#     vals = bw_handle.values(chrom, start, end)
#     vals = np.array([0. if np.isnan(x) else x for x in vals], dtype=np.float32)
#     return vals

# def extract_additional_features(
#     bed_lines,
#     feature_bw_handles,
#     location_func,
#     chrom_lengths,
#     seq_len=SEQ_LEN,
#     window=21,
#     smooth=True
# ):
#     """
#     bed_lines: list of lines (bed entries)
#     feature_bw_handles: list of opened pyBigWig handles
#     location_func: function(chrom, start, end) -> (center_pos) or (start_f, end_f)
#     chrom_lengths: dict mapping chromosome name to length
#     Returns: numpy array of shape [N, F, SEQ_LEN]
#     """
#     all_feats = []
#     for bw in feature_bw_handles:
#         feat_signals = []
#         for line in bed_lines:
#             chrom, start_s, end_s = line.strip().split()[:3]
#             if chrom not in chrom_lengths:
#                 continue
#             start, end = int(start_s), int(end_s)
#             start_f, end_f = location_func(chrom, start, end)

#             # Determine windowed region: center around start_f/end_f with seq_len
#             center = (start_f + end_f) // 2 if isinstance(start_f, int) and isinstance(end_f, int) else start_f
#             half = seq_len // 2
#             start_c = max(0, center - half)
#             end_c = min(chrom_lengths[chrom], start_c + seq_len)
#             # adjust if truncated at end
#             if end_c - start_c < seq_len:
#                 start_c = max(0, end_c - seq_len)

#             vals = get_bigwig_vals(bw, chrom, start_c, end_c)
#             if smooth:
#                 vals = smooth_avg(vals, window)
#             # pad/truncate to exactly seq_len
#             if len(vals) < seq_len:
#                 pad_len = seq_len - len(vals)
#                 vals = np.pad(vals, (0, pad_len), mode='constant', constant_values=0)
#             elif len(vals) > seq_len:
#                 vals = vals[:seq_len]

#             feat_signals.append(vals)
#         feat_signals = np.array(feat_signals, dtype=np.float32)  # [N, SEQ_LEN]
#         feat_signals = np.log10(1 + feat_signals)
#         all_feats.append(feat_signals)

#     if not all_feats:
#         return np.zeros((0, 0, seq_len), dtype=np.float32)
#     all_feats = np.stack(all_feats, axis=0)  # [F, N, SEQ_LEN]
#     all_feats = np.transpose(all_feats, (1, 0, 2))  # [N, F, SEQ_LEN]
#     return all_feats

# def extract_true_data(
#     bed_file,
#     genome_fa,
#     bw_file,
#     feature_bw_files,
#     output_path,
#     location_func,
#     seq_len=SEQ_LEN,
#     flank=FLANK,
#     window=21,
#     smooth=True
# ):
#     # Load reference genome
#     sequence_dict = SeqIO.to_dict(SeqIO.parse(open(genome_fa), 'fasta'))
#     chrom_lengths = {chrom: len(sequence_dict[chrom]) for chrom in sequence_dict}

#     # Open primary bigWig
#     bw = pyBigWig.open(bw_file)
#     # Open additional feature bigWigs
#     feature_bws = [pyBigWig.open(f) for f in feature_bw_files]

#     seqs, signals, additional = [], [], []
#     with open(bed_file) as f:
#         lines = f.readlines()

#     # Extract additional features once for all lines
#     feats = extract_additional_features(
#         lines,
#         feature_bws,
#         location_func,
#         chrom_lengths,
#         seq_len=seq_len,
#         window=window,
#         smooth=smooth
#     )  # shape [N, F, SEQ_LEN]

#     feat_idx = 0  # to keep in sync with successful sequence extraction
#     for i, line in enumerate(tqdm(lines, desc="Extracting true values and features")):
#         chrom, start_s, end_s = line.strip().split()[:3]
#         if chrom not in INDEX:
#             continue
#         start, end = int(start_s), int(end_s)
#         center = (start + end) // 2
#         start_pos = center - flank
#         end_pos = center + flank

#         if start_pos < 0 or end_pos > len(sequence_dict[chrom]):
#             continue

#         signal = bw.values(chrom, start_pos, end_pos)
#         signal = np.array([0 if np.isnan(x) else x for x in signal], dtype=np.float32)
#         # optionally could normalize or keep raw
#         seq = one_hot(sequence_dict, chrom, start_pos, end_pos)
#         seqs.append(seq)
#         signals.append(signal)
#         # Pull corresponding additional feature row; assume extract_additional_features kept same ordering
#         if feat_idx < feats.shape[0]:
#             additional.append(feats[feat_idx])
#         else:
#             # fallback if mismatch
#             additional.append(np.zeros((feats.shape[1], seq_len), dtype=np.float32))
#         feat_idx += 1

#     bw.close()
#     for fbw in feature_bws:
#         fbw.close()

#     seqs = np.array(seqs, dtype=np.float32).transpose(0, 2, 1)  # [N, 4, SEQ_LEN]
#     signals = np.array(signals, dtype=np.float32)  # [N, SEQ_LEN]
#     additional = np.array(additional, dtype=np.float32)  # [N, F, SEQ_LEN]
#     labels = np.ones((len(seqs), 1), dtype=np.float32)

#     np.savez(output_path, data=seqs, signal=signals, label=labels, extra=additional)
#     print(f"Saved {len(seqs)} sequences with {additional.shape[1] if additional.size else 0} features to {output_path}")

# def get_fcn_predictions(model_path, npz_file, output_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
#     print(f"Loading data from: {npz_file}")
#     data = np.load(npz_file)
#     seq_inputs = torch.tensor(data['data'], dtype=torch.float32).to(device)  # shape: [N, 4, SEQ_LEN]
#     true_signals = data['signal'].squeeze()
    
#     # Load extra features (shape: [N, F, SEQ_LEN], here F=2)
#     extra = data['extra'] if 'extra' in data else None
#     if extra is not None:
#         extra = torch.tensor(extra, dtype=torch.float32).to(device)  # [N, 2, SEQ_LEN]
#         # Concatenate along channel dimension (dim=1)
#         inputs = torch.cat([seq_inputs, extra], dim=1)  # [N, 6, SEQ_LEN]
#     else:
#         inputs = seq_inputs  # fallback
    
#     num_sequences = inputs.shape[0]
#     print(f"Number of sequences in input data: {num_sequences}")
    
#     model = FCNsignalMod(n_features=2)  # specify total channels: 4+2=6
#     state_dict = torch.load(model_path, map_location=device)
#     if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
#         state_dict = state_dict['model_state_dict']
#     model.load_state_dict(state_dict)
#     model.to(device)
#     model.eval()
    
#     preds = []
#     batch_size = 128
#     with torch.no_grad():
#         for i in tqdm(range(0, num_sequences, batch_size), desc="Predicting"):
#             batch = inputs[i:i+batch_size]
#             out = model(batch).cpu().numpy()
#             preds.append(out)
    
#     pred_signals = np.concatenate(preds, axis=0)
#     print(f"Number of sequences predicted: {len(pred_signals)}")
    
#     if output_path:
#         print(f"Saving predictions to: {output_path}")
#         np.savez(output_path, pred_signal=pred_signals, true_signal=true_signals)
    
#     return true_signals, pred_signals


# def plot_max_scatter(true_signals, pred_signals, out='scatter_max.png'):
#     # Calculate max values for each sequence
#     true_signals = np.log10(1 + true_signals)
#     true_max = np.max(true_signals, axis=1)
#     pred_max = np.max(pred_signals, axis=1)

#     # Pearson correlation
#     overall_pearson, _ = pearsonr(true_max, pred_max)

#     # Create scatter plot
#     plt.figure(figsize=(6, 6))
#     plt.scatter(pred_max, true_max, s=2, alpha=0.5)
    
#     # Regression line 
#     fit = np.polyfit(pred_max, true_max, 1)
#     fit_fn = np.poly1d(fit)
#     plt.plot(pred_max, fit_fn(pred_max), '--', color='black')
#     plt.xlabel('Predicted Maximum Values')
#     plt.ylabel('True Maximum Values')
#     plt.xlim(0, np.max(pred_max)+0.5)
#     plt.ylim(0, np.max(true_max)+0.5)
#     plt.tight_layout()
#     plt.savefig(out)
#     plt.close()
#     print(f"Max scatter plot saved to {out}")


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--generate', action='store_true')
#     parser.add_argument('--predict', action='store_true')
#     parser.add_argument('--plot', action='store_true')
    
#     parser.add_argument('--bed', type=str)
#     parser.add_argument('--fasta', type=str)
#     parser.add_argument("--primary_bw",  help="Path to primary bigWig (signal)")
#     parser.add_argument("--feature_bws", nargs="+",  help="Paths to additional feature bigWigs")
#     parser.add_argument('--output', type=str, default='true_value.npz')
#     parser.add_argument('--model', type=str)
#     parser.add_argument('--npz', type=str)
#     parser.add_argument('--scatter', type=str, default='scatter_colored.png')

#     args = parser.parse_args()

#     if args.generate:
#         extract_true_data(
#             bed_file=args.bed,
#             genome_fa=args.fasta,
#             bw_file=args.primary_bw,       
#             feature_bw_files=args.feature_bws,  
#             output_path=args.output,
#             location_func=lambda c, s, e: ((s + e) // 2, (s + e) // 2),  
#             window=21,
#             smooth=True
#         )

#     if args.predict:
#         if not args.model or not args.npz:
#             print("Need both --model and --npz for prediction")
#         else:
#             true_vals, pred_vals = get_fcn_predictions(args.model, args.npz)
#             np.savez('predicted.npz', true=true_vals, pred=pred_vals)

#     if args.plot:
#         data = np.load(args.npz)
#         true_vals, pred_vals = data['true'], data['pred']
#         plot_max_scatter(true_vals, pred_vals, out=args.scatter)