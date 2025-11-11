# coding:utf-8
import os.path as osp
import os
import sys
import argparse
import itertools
import random
import numpy as np
from Bio import SeqIO
import pyBigWig

SEQ_LEN = 1001
OFFSET = 3000
INDEX = ['chr' + str(i + 1) for i in range(23)]
INDEX[22] = 'chrX'
SMOOTH = True
WINDOW = 15
THRESHOLD = 0.05
ROOT = osp.dirname(__file__)
CHROM = {}
with open('Genome/mm9.chrom.sizes') as f:
    for i in f:
        line_split = i.strip().split()
        CHROM[line_split[0]] = int(line_split[1])


def one_hot(sequence_dict, chrom, start, end):
    seq_dict = {'A':[1, 0, 0, 0], 'G':[0, 0, 1, 0],
                'C':[0, 1, 0, 0], 'T':[0, 0, 0, 1],
                'a':[1, 0, 0, 0], 'g':[0, 0, 1, 0],
                'c':[0, 1, 0, 0], 't':[0, 0, 0, 1]}
    temp = []
    seq = str(sequence_dict[chrom].seq[start:end])
    for c in seq:
        temp.append(seq_dict.get(c, [0, 0, 0, 0]))
    return temp


def GetbigWig(file, chrom, start, end):
    start = max(0, start)
    end = min(end, CHROM.get(chrom, end))  # Clamp to chrom length
    bw = pyBigWig.open(file)
    sample = np.array(bw.values(chrom, start, end))
    bw.close()
    return sample


def Smoothavg(x, window):
    avg =[]
    for j in range(len(x)-window+1):
        avg.append(np.mean(x[j:(j+window)]))

    return np.array(avg)


# random position for the max signal
def pos_location(chrom, start, end):
    original_len = end - start
    if original_len < SEQ_LEN:
        start_update = start - np.ceil((SEQ_LEN - original_len) / 2)
    elif original_len > SEQ_LEN:
        start_update = start + np.ceil((original_len - SEQ_LEN) / 2)
    else:
        start_update = start

    start_update += np.random.randint(-300, 300)
    end_update = start_update + SEQ_LEN
    if end_update > CHROM[chrom]:
        end_update = CHROM[chrom]
        start_update = end_update - SEQ_LEN
    return int(start_update), int(end_update)


# upstream of positive peaks
def neg_location(chrom, start, end):
    start_update = start - OFFSET
    if start_update < 0:
        start_update = 0
    end_update = start_update + SEQ_LEN

    return int(start_update), int(end_update)


def get_data_local_neg(seqs_bed, sequence_dict, signal_file, max_neg, additional_feature_files=[]):
    seqs_neg = []
    signals_neg = []
    filtered_bed_lines = []
    lines = open(seqs_bed).readlines()
    index = list(range(len(lines)))
    for i in index:
        line_split = lines[i].strip().split()
        chrom = line_split[0]
        if chrom not in INDEX:
            continue
        start, end = int(line_split[1]), int(line_split[2])
        # for negative peaks
        start_n, end_n = neg_location(chrom, start, end)
        # for positive peaks
        signal = GetbigWig(signal_file, chrom, start_n, end_n)
        signal[np.isnan(signal)] = 0.
        if max_neg <= max(signal):
            continue
        seqs_neg.append(one_hot(sequence_dict, chrom, start_n, end_n))
        signals_neg.append(signal)
        filtered_bed_lines.append(lines[i])
    seqs_neg = np.array(seqs_neg, dtype=np.float32)
    seqs_neg = seqs_neg.transpose((0, 2, 1))
    signals_neg = [[x] for x in signals_neg]
    signals_neg = np.array(signals_neg, dtype=np.float32)

    if additional_feature_files:
        additional_feats = extract_additional_features(filtered_bed_lines, additional_feature_files, neg_location)
    else:
        additional_feats = None
    return seqs_neg, signals_neg, additional_feats



def get_data_local(seqs_bed, sequence_dict, signal_file, additional_feature_files=[]):
    seqs = []
    signals = []
    max_all = []
    filtered_bed_lines = []   
    lines = open(seqs_bed).readlines()
    index = list(range(len(lines)))
    if SMOOTH:
        print("Using the smooth way.")
    else:
        print("Using the normal way.")
    for i in index:
        line_split = lines[i].strip().split()
        chrom = line_split[0]
        if chrom not in INDEX:
            continue
        start, end = int(line_split[1]), int(line_split[2])
        start_p, end_p = pos_location(chrom, start, end)
        if SMOOTH:
            signal = GetbigWig(signal_file, chrom, start_p - WINDOW // 2, end_p + WINDOW // 2)
            signal[np.isnan(signal)] = 0.
            signal = Smoothavg(signal, WINDOW)
            Max = np.max(signal)
        else:
            signal = GetbigWig(signal_file, chrom, start_p, end_p)
            signal[np.isnan(signal)] = 0.
            Max = np.max(signal)
        if Max == 0: continue
        max_all.append(Max)
        signals.append(signal)
        seqs.append(one_hot(sequence_dict, chrom, start_p, end_p))
        filtered_bed_lines.append(lines[i])

    print("The minimum and maximum are {:.3f} and {:.3f}".format(np.min(max_all), np.max(max_all)))
    sort_i = np.argsort(max_all)
    start_i = int(THRESHOLD*len(sort_i))
    seqs_filter = []
    signals_filter = []
    filtered_bed_lines_filter = []
    for i in sort_i[start_i:]:
        seqs_filter.append(seqs[i])
        signals_filter.append(signals[i])
        filtered_bed_lines_filter.append(filtered_bed_lines[i])
    seqs_filter = np.array(seqs_filter, dtype=np.float32)
    seqs_filter = seqs_filter.transpose((0, 2, 1))
    signals_filter = [[x] for x in signals_filter]
    signals_filter = np.array(signals_filter, dtype=np.float32)
    if additional_feature_files:
        additional_feats = extract_additional_features(filtered_bed_lines_filter, additional_feature_files, pos_location)
    else:
        additional_feats = None

    return seqs_filter, signals_filter, max_all[sort_i[start_i]], additional_feats

def extract_additional_features(bed_lines, feature_files, location_func):
    all_feats = []
    for feat_file in feature_files:
        feat_signals = []
        for line in bed_lines:
            chrom, start_s, end_s = line.strip().split()[:3]
            start, end = int(start_s), int(end_s)
            start_f, end_f = location_func(chrom, start, end)

            # Clamp coordinates
            start_c = max(0, start_f - WINDOW // 2)
            end_c = min(end_f + WINDOW // 2, CHROM.get(chrom, end_f + WINDOW // 2))

            if SMOOTH:
                vals = GetbigWig(feat_file, chrom, start_c, end_c)
                vals[np.isnan(vals)] = 0.
                vals = Smoothavg(vals, WINDOW)
            else:
                vals = GetbigWig(feat_file, chrom, start_f, end_f)
                vals[np.isnan(vals)] = 0.

            if len(vals) < SEQ_LEN:
                pad_len = SEQ_LEN - len(vals)
                vals = np.pad(vals, (0, pad_len), mode='constant', constant_values=0)
            elif len(vals) > SEQ_LEN:
                vals = vals[:SEQ_LEN]

            feat_signals.append(vals)

        feat_signals = np.array(feat_signals, dtype=np.float32)  # [N, SEQ_LEN]
        feat_signals = np.log10(1 + feat_signals)
        all_feats.append(feat_signals)

    all_feats = np.array(all_feats)  # shape [F, N, SEQ_LEN]
    all_feats = np.transpose(all_feats, (1, 0, 2))  # [N, F, SEQ_LEN]
    return all_feats

def save_peak_signals_separate(seqs_bed, signal_file, out_file, smooth=SMOOTH, window=WINDOW, offset=OFFSET):
    """
    Save per-peak signals (positive and negative) to a separate TSV file.
    Ensures the coordinates and signal lengths match what is used in .npz files.
    """
    with open(seqs_bed) as f_in, open(out_file, 'w') as f_out:
        f_out.write('peak_type\tchrom\tstart\tend\traw_signal\tsmoothed_norm_signal\n')

        for line in f_in:
            chrom, start_s, end_s = line.strip().split()[:3]
            start, end = int(start_s), int(end_s)

            if chrom not in CHROM:
                continue

            # ---------------- Positive peak ----------------
            start_p, end_p = pos_location(chrom, start, end)
            start_p = max(0, start_p)
            end_p = min(end_p, CHROM[chrom])
            if end_p <= start_p:
                continue

            raw_signal = GetbigWig(signal_file, chrom, start_p, end_p)
            raw_signal[np.isnan(raw_signal)] = 0.
            if smooth:
                raw_signal_smooth = Smoothavg(raw_signal, window)
            else:
                raw_signal_smooth = raw_signal

            # pad/crop to SEQ_LEN
            if len(raw_signal_smooth) < SEQ_LEN:
                raw_signal_smooth = np.pad(raw_signal_smooth, (0, SEQ_LEN - len(raw_signal_smooth)), mode='constant')
            elif len(raw_signal_smooth) > SEQ_LEN:
                raw_signal_smooth = raw_signal_smooth[:SEQ_LEN]

            smoothed_norm = np.log10(1 + raw_signal_smooth)
            raw_str = ','.join([f'{x:.4f}' for x in raw_signal])
            smooth_str = ','.join([f'{x:.4f}' for x in smoothed_norm])
            f_out.write(f'pos\t{chrom}\t{start_p}\t{end_p}\t{raw_str}\t{smooth_str}\n')

            # ---------------- Negative peak ----------------
            start_n, end_n = neg_location(chrom, start, end)
            start_n = max(0, start_n)
            end_n = min(end_n, CHROM[chrom])
            if end_n <= start_n:
                continue

            raw_signal_n = GetbigWig(signal_file, chrom, start_n, end_n)
            raw_signal_n[np.isnan(raw_signal_n)] = 0.
            if smooth:
                raw_signal_smooth_n = Smoothavg(raw_signal_n, window)
            else:
                raw_signal_smooth_n = raw_signal_n

            # pad/crop to SEQ_LEN
            if len(raw_signal_smooth_n) < SEQ_LEN:
                raw_signal_smooth_n = np.pad(raw_signal_smooth_n, (0, SEQ_LEN - len(raw_signal_smooth_n)), mode='constant')
            elif len(raw_signal_smooth_n) > SEQ_LEN:
                raw_signal_smooth_n = raw_signal_smooth_n[:SEQ_LEN]

            smoothed_norm_n = np.log10(1 + raw_signal_smooth_n)
            raw_str_n = ','.join([f'{x:.4f}' for x in raw_signal_n])
            smooth_str_n = ','.join([f'{x:.4f}' for x in smoothed_norm_n])
            f_out.write(f'neg\t{chrom}\t{start_n}\t{end_n}\t{raw_str_n}\t{smooth_str_n}\n')




def get_args():
    parser = argparse.ArgumentParser(description="pre-process data.")
    parser.add_argument("-d", dest="dir", type=str, default='')
    parser.add_argument("-n", dest="name", type=str, default='')
    parser.add_argument("-s", dest="seed", type=int, default=666, help="Random seed to have reproducible results.")
    parser.add_argument("--threshold", dest="thresh", type=float, default=0.2,
                        help="Threshold for binary classification of signal after log10 normalization.")
    return parser.parse_args()



def main():
    params = get_args()
    random.seed(params.seed)
    name = params.name
    data_dir = params.dir
    out_dir = osp.join(data_dir, 'data_binary_WT2_features')
    if not osp.exists(out_dir):
        os.mkdir(out_dir)
        
    sequence_dict = SeqIO.to_dict(SeqIO.parse(open('Genome/mm9.fa'), 'fasta'))
    print('Experiment on %s dataset' % name)

    seqs_bed = osp.join(data_dir, 'WT2_0based.bed')
    pvalue_file = osp.join(data_dir, 'p-value.bw')
    out_file = osp.join(data_dir, 'peak_signals.tsv')
    save_peak_signals_separate(seqs_bed, pvalue_file, out_file)
    
    # Uncomment or leave empty if no extra features
    additional_feature_files = [
        osp.join(data_dir, 'histone_marks.bw'),
        osp.join(data_dir, 'ATAC2.bw')
    ]
    # additional_feature_files = []  

    # ----- Positive samples -----
    seqs, signals, max_neg, additional_feats = get_data_local(
        seqs_bed, sequence_dict, pvalue_file, additional_feature_files=additional_feature_files
    )

    # Log + threshold
    signals = np.log10(1 + signals)
    signals = (signals >= params.thresh).astype(np.int8)

    if additional_feats is not None and len(additional_feats) > 0:
        print("seqs shape:", seqs.shape)
        print("additional_feats shape:", additional_feats.shape)
        seqs_full = np.concatenate((seqs, additional_feats), axis=1)
    else:
        seqs_full = seqs

    # Train/test split
    ratio = 0.2
    number_t = int(len(seqs_full) * ratio)
    index = list(range(len(seqs_full)))
    index_test = random.sample(index, number_t)
    seqs_te = seqs_full[index_test]
    signals_te = signals[index_test]
    index_train = list(set(index) - set(index_test))
    seqs_tr = seqs_full[index_train]
    signals_tr = signals[index_train]
    labels_tr = np.ones((len(seqs_tr), 1))
    labels_te = np.ones((len(seqs_te), 1))

    np.savez(osp.join(out_dir, f'{name}_test.npz'), data=seqs_te, signal=signals_te, label=labels_te)

    # ----- Negative samples -----
    seqs_neg, signals_neg, additional_feats_neg = get_data_local_neg(
        seqs_bed, sequence_dict, pvalue_file, max_neg, additional_feature_files=additional_feature_files
    )

    signals_neg = np.log10(1 + signals_neg)
    signals_neg = (signals_neg >= params.thresh).astype(np.int8)

    if additional_feats_neg is not None and len(additional_feats_neg) > 0:
        print("seqs_neg shape:", seqs_neg.shape)
        print("additional_feats_neg shape:", additional_feats_neg.shape)
        seqs_neg_full = np.concatenate((seqs_neg, additional_feats_neg), axis=1)
    else:
        seqs_neg_full = seqs_neg

    # Train/test split for negatives
    number_t = int(len(seqs_neg_full) * ratio)
    index_neg = list(range(len(seqs_neg_full)))
    index_neg_test = random.sample(index_neg, number_t)
    seqs_neg_te = seqs_neg_full[index_neg_test]
    signals_neg_te = signals_neg[index_neg_test]
    index_neg_train = list(set(index_neg) - set(index_neg_test))
    seqs_neg_tr = seqs_neg_full[index_neg_train]
    signals_neg_tr = signals_neg[index_neg_train]

    labels_neg_tr = np.zeros((len(seqs_neg_tr), 1))
    labels_neg_te = np.zeros((len(seqs_neg_te), 1))

    np.savez(osp.join(out_dir, f'{name}_neg.npz'), data=seqs_neg_te, signal=signals_neg_te, label=labels_neg_te)

    # ----- Merge train -----
    seqs_total = np.concatenate((seqs_tr, seqs_neg_tr), axis=0)
    signals_total = np.concatenate((signals_tr, signals_neg_tr))
    labels_total = np.concatenate((labels_tr, labels_neg_tr), axis=0)

    np.savez(osp.join(out_dir, f'{name}_train.npz'),
             data=seqs_total, signal=signals_total, label=labels_total)

    print(f"{name}: The train data are: {len(seqs_total)}")
    print(f"{name}: The positive test data are: {len(seqs_te)}")
    print(f"{name}: The negative test data are: {len(seqs_neg_te)}\n")
    

if __name__ == '__main__':  main()