#!/usr/bin/env python3
import random
import argparse
import logging
import os
import numpy as np
from Bio import SeqIO
try:
    import pyBigWig
    HAS_PYBIGWIG = True
except ImportError:
    HAS_PYBIGWIG = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Genome / BED helpers

def load_genome(genome_file):
    logger.info(f"Loading genome from {genome_file}")
    genome = {}
    for record in SeqIO.parse(genome_file, "fasta"):
        genome[record.id] = str(record.seq).upper()
    logger.info(f"Loaded {len(genome)} chromosomes")
    return genome

def read_bed(bed_file):
    logger.info(f"Reading BED file: {bed_file}")
    peaks = []
    with open(bed_file) as f:
        for line in f:
            if line.startswith("#") or line.startswith("track"):
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            chrom, start, end = parts[0], int(parts[1]), int(parts[2])
            peaks.append((chrom, start, end))
    logger.info(f"Found {len(peaks)} peaks")
    return peaks


# Exclusion index (for negative sampling)
def build_exclusion_index(peaks, expand_bp=0):
    index = {}
    for chrom, start, end in peaks:
        exc_start = max(0, start - expand_bp)
        exc_end   = end + expand_bp
        index.setdefault(chrom, []).append((exc_start, exc_end))
    for chrom in index:
        index[chrom].sort()
    return index


def overlaps_excluded(chrom, start, end, exclusion_index):
    if chrom not in exclusion_index:
        return False
    intervals = exclusion_index[chrom]
    lo, hi = 0, len(intervals)
    while lo < hi:
        mid = (lo + hi) // 2
        if intervals[mid][1] <= start:
            lo = mid + 1
        else:
            hi = mid
    for i in range(lo, len(intervals)):
        if intervals[i][0] >= end:
            break
        return True
    return False



# Window extraction


def extract_positive_window(genome, chrom, start, end, seq_length):
    """
    Extract a window of exactly seq_length that contains [start, end).
    The peak is placed at a random offset within the window.
    Returns (sequence, bertsnr_labels, coords) or (None, None, None).
    """
    if chrom not in genome:
        return None, None, None

    center     = (start + end) // 2
    peak_width = end - start

    min_offset = peak_width // 2
    max_offset = seq_length - (peak_width - peak_width // 2)

    if min_offset > max_offset:
        offset = random.randint(0, seq_length - 1)
    else:
        offset = random.randint(min_offset, max_offset - 1)

    seq_start = center - offset
    seq_end   = seq_start + seq_length

    chrom_len = len(genome[chrom])
    if seq_start < 0:
        seq_start = 0
        seq_end   = seq_length
    if seq_end > chrom_len:
        seq_end   = chrom_len
        seq_start = max(0, chrom_len - seq_length)

    sequence = genome[chrom][seq_start:seq_end]
    if len(sequence) < seq_length:
        sequence += "N" * (seq_length - len(sequence))

    # BERTsnr per-nucleotide label: 1 where the peak overlaps this window
    labels    = np.zeros(seq_length, dtype=np.int8)
    rel_start = start - seq_start
    rel_end   = end   - seq_start
    clip_s    = max(0, rel_start)
    clip_e    = min(seq_length, rel_end)
    if clip_s < clip_e:
        labels[clip_s:clip_e] = 1

    coords = (chrom, seq_start, seq_end, "+")
    return sequence, labels, coords


def sample_negative_window(genome, exclusion_index, seq_length,
                            max_n_fraction=0.1, max_attempts=1000,
                            expand_bp=0):
    chroms       = list(genome.keys())
    chrom_lengths = {c: len(genome[c]) for c in chroms}
    total_len    = sum(chrom_lengths.values())
    weights      = [chrom_lengths[c] / total_len for c in chroms]

    for _ in range(max_attempts):
        chrom     = random.choices(chroms, weights=weights, k=1)[0]
        chrom_len = chrom_lengths[chrom]
        if chrom_len < seq_length:
            continue
        start = random.randint(0, chrom_len - seq_length)
        end   = start + seq_length

        exc_start = max(0, start - expand_bp)
        exc_end   = end + expand_bp

        if overlaps_excluded(chrom, exc_start, exc_end, exclusion_index):
            continue

        sequence = genome[chrom][start:end]
        n_count  = sequence.count("N") + sequence.count("n")
        if n_count / seq_length > max_n_fraction:
            continue

        coords = (chrom, start, end, "+")
        return sequence, coords

    logger.warning(f"Could not find a valid negative window after {max_attempts} attempts")
    return None, None



# Sequence encoders
def one_hot(sequence):
    seq_dict = {
        'A': [1, 0, 0, 0], 'a': [1, 0, 0, 0],
        'C': [0, 1, 0, 0], 'c': [0, 1, 0, 0],
        'G': [0, 0, 1, 0], 'g': [0, 0, 1, 0],
        'T': [0, 0, 0, 1], 't': [0, 0, 0, 1],
    }
    return [seq_dict.get(c, [0, 0, 0, 0]) for c in sequence]


def one_hot_encode(sequence):
    """Return array of shape [4, L] (channel-first, matches FCN convention)."""
    enc = np.array(one_hot(sequence), dtype=np.float32)  # [L, 4]
    return enc.T                               # [4, L]



# BigWig / FCN signal helpers
def smooth_avg(x, window):
    avg = []
    for j in range(len(x) - window + 1):
        avg.append(np.mean(x[j:(j + window)]))
    return np.array(avg)


def get_bigwig_signal(bw_file, chrom, start, end, chrom_sizes):
    start = max(0, start)
    end   = min(end, chrom_sizes.get(chrom, end))
    bw    = pyBigWig.open(bw_file)
    vals  = np.array(bw.values(chrom, start, end), dtype=np.float64)
    bw.close()
    vals[np.isnan(vals)] = 0.0
    return vals


def extract_fcn_label(bw_file, chrom, seq_start, seq_end, seq_length,
                       chrom_sizes, smooth=True, window=15, threshold=0.05):
    if smooth:
        raw  = get_bigwig_signal(bw_file, chrom,
                                  seq_start - window // 2,
                                  seq_end   + window // 2,
                                  chrom_sizes)
        sig  = smooth_avg(raw, window)
    else:
        sig  = get_bigwig_signal(bw_file, chrom, seq_start, seq_end, chrom_sizes)

    # Pad / crop to seq_length
    if len(sig) < seq_length:
        sig = np.pad(sig, (0, seq_length - len(sig)))
    else:
        sig = sig[:seq_length]

    log_sig = np.log10(1.0 + sig)
    return (log_sig >= threshold).astype(np.int8)


def write_bertsnr(out_dir, sequences, bertsnr_labels, seq_labels, coords):
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "dataset.txt"), "w") as f:
        f.write("sequence\tlabel\tseq_label\n")
        for seq, lab, sl in zip(sequences, bertsnr_labels, seq_labels):
            f.write(f"{seq}\t{''.join(map(str, lab))}\t{sl}\n")

    with open(os.path.join(out_dir, "coord.txt"), "w") as f:
        for c in coords:
            f.write(f"{c[0]}\t{c[1]}\t{c[2]}\t{c[3]}\n")

    logger.info(f"BERTsnr files written to {out_dir}  ({len(sequences)} samples)")


def write_fcn(out_path, one_hots, fcn_labels, seq_labels):
    data_arr   = np.stack(one_hots,   axis=0)              # [N, 4, L]
    signal_arr = np.stack(fcn_labels, axis=0)[:, np.newaxis, :]  # [N, 1, L]
    label_arr  = np.array(seq_labels, dtype=np.float32).reshape(-1, 1)
    np.savez(out_path, data=data_arr, signal=signal_arr, label=label_arr)
    logger.info(f"FCN npz written to {out_path}  ({len(one_hots)} samples)")

# Main

def main():
    parser = argparse.ArgumentParser(
        description="Unified BERTsnr + FCN dataset preparation."
    )
    # Required
    parser.add_argument("--bed_file",    required=True,  help="BED file of peaks")
    parser.add_argument("--genome_file", required=True,  help="FASTA genome file")
    parser.add_argument("--output_path", required=True,  help="Root output directory")

    # Sequence / sampling
    parser.add_argument("--seq_length",      type=int,   default=1001,
                        help="Window length in bp (default: 1001)")
    parser.add_argument("--negative_ratio",  type=int,   default=1,
                        help="Negative windows per positive (default: 1)")
    parser.add_argument("--expand_bp",       type=int,   default=0,
                        help="Expand peaks by this many bp when excluding for negatives (default: 0)")
    parser.add_argument("--max_n_fraction",  type=float, default=0.1,
                        help="Max fraction of N bases allowed in a window (default: 0.1)")
    parser.add_argument("--test_ratio",      type=float, default=0.2,
                        help="Fraction of data held out as test set (default: 0.2)")

    # FCN-specific (optional)
    parser.add_argument("--bigwig_file",  default=None,
                        help="BigWig signal file for FCN per-nucleotide labels. "
                             "If omitted, FCN signal labels default to BERTsnr labels.")
    parser.add_argument("--chrom_sizes",  default=None,
                        help="Chromosome sizes file (required when --bigwig_file is set)")
    parser.add_argument("--smooth",       action="store_true", default=True,
                        help="Apply moving-average smoothing to BigWig signal (default: True)")
    parser.add_argument("--no_smooth",    dest="smooth", action="store_false")
    parser.add_argument("--smooth_window",type=int,   default=15,
                        help="Smoothing window size (default: 15)")
    parser.add_argument("--fcn_threshold",type=float, default=0.05,
                        help="Log10-signal threshold for FCN binary labels")

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Validate 
    if args.bigwig_file and not HAS_PYBIGWIG:
        raise ImportError("pyBigWig is required when --bigwig_file is supplied. "
                          "Install with:  pip install pyBigWig")
    if args.bigwig_file and not args.chrom_sizes:
        raise ValueError("--chrom_sizes is required when --bigwig_file is set.")

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load resources 
    genome = load_genome(args.genome_file)
    peaks  = read_bed(args.bed_file)

    chrom_sizes = {}
    if args.chrom_sizes:
        with open(args.chrom_sizes) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    chrom_sizes[parts[0]] = int(parts[1])

    logger.info(f"Building exclusion index (expand_bp={args.expand_bp})")
    exclusion_index = build_exclusion_index(peaks, expand_bp=args.expand_bp)

    # Collect all samples 
    # Each sample: (sequence, bertsnr_label, fcn_label, seq_label, coords)
    all_samples = []

    logger.info("Extracting positive windows …")
    for chrom, start, end in peaks:
        seq, bert_lab, coords = extract_positive_window(
            genome, chrom, start, end, args.seq_length
        )
        if seq is None:
            continue

        # FCN label
        if args.bigwig_file:
            fcn_lab = extract_fcn_label(
                args.bigwig_file, coords[0], coords[1], coords[2],
                args.seq_length, chrom_sizes,
                smooth=args.smooth,
                window=args.smooth_window,
                threshold=args.fcn_threshold,
            )
        else:
            # Fall back: use BERTsnr peak-overlap labels
            fcn_lab = bert_lab.copy()

        all_samples.append((seq, bert_lab, fcn_lab, 1, coords))

    n_pos = len(all_samples)
    logger.info(f"Positive samples: {n_pos}")

    logger.info("Sampling negative windows …")
    neg_found = neg_failed = 0
    for i in range(n_pos):
        for _ in range(args.negative_ratio):
            neg_seq, neg_coords = sample_negative_window(
                genome, exclusion_index, args.seq_length,
                max_n_fraction=args.max_n_fraction,
                expand_bp=args.expand_bp,
            )
            if neg_seq is None:
                neg_failed += 1
                continue

            bert_lab = np.zeros(args.seq_length, dtype=np.int8)

            if args.bigwig_file:
                fcn_lab = extract_fcn_label(
                    args.bigwig_file,
                    neg_coords[0], neg_coords[1], neg_coords[2],
                    args.seq_length, chrom_sizes,
                    smooth=args.smooth,
                    window=args.smooth_window,
                    threshold=args.fcn_threshold,
                )
            else:
                fcn_lab = bert_lab.copy()

            all_samples.append((neg_seq, bert_lab, fcn_lab, 0, neg_coords))
            neg_found += 1

    logger.info(f"Negative samples found: {neg_found}")
    if neg_failed:
        logger.warning(f"Negative samples failed: {neg_failed}")

    # Train / test split
    n_total     = len(all_samples) 
    pos_indices = [i for i, s in enumerate(all_samples) if s[3] == 1]
    neg_indices = [i for i, s in enumerate(all_samples) if s[3] == 0]

    random.shuffle(pos_indices)
    random.shuffle(neg_indices)

    n_pos_test = max(1, round(len(pos_indices) * args.test_ratio))
    n_neg_test = max(1, round(len(neg_indices) * args.test_ratio))

    # Balance: use the smaller of the two test counts for both classes
    n_per_class_test = min(n_pos_test, n_neg_test)

    test_idx  = set(pos_indices[:n_per_class_test] + neg_indices[:n_per_class_test])
    train_idx = set(pos_indices[n_per_class_test:] + neg_indices[n_per_class_test:])

    logger.info(f"Stratified split — test: {n_per_class_test} pos + {n_per_class_test} neg = {n_per_class_test*2}")
    logger.info(f"Stratified split — train: {len(pos_indices)-n_per_class_test} pos + {len(neg_indices)-n_per_class_test} neg = {len(train_idx)}")


    def partition(idx_set):
        seqs, bert_labs, fcn_labs, seq_labs, coords = [], [], [], [], []
        for i in idx_set:
            s, b, f, sl, c = all_samples[i]
            seqs.append(s)
            bert_labs.append(b)
            fcn_labs.append(f)
            seq_labs.append(sl)
            coords.append(c)
        return seqs, bert_labs, fcn_labs, seq_labs, coords

    train = partition(sorted(train_idx))
    test  = partition(sorted(test_idx))

    #  Write outputs 
    bertsnr_root = os.path.join(args.output_path, "bertsnr")
    fcn_root     = os.path.join(args.output_path, "fcn")
    os.makedirs(fcn_root, exist_ok=True)

    for split_name, (seqs, bert_labs, fcn_labs, seq_labs, coords) in [
        ("train", train), ("test", test)
    ]:
        # BERTsnr text files
        write_bertsnr(
            os.path.join(bertsnr_root, split_name),
            seqs, bert_labs, seq_labs, coords
        )

        # FCN npz
        one_hots = [one_hot_encode(s) for s in seqs]
        write_fcn(
            os.path.join(fcn_root, f"{split_name}.npz"),
            one_hots, fcn_labs, seq_labs
        )

    #  Summary 
    logger.info("=" * 60)
    logger.info("DONE")
    logger.info(f"Total samples : {n_total}")
    logger.info(f"  Train       : {len(train[0])}  "
                f"(pos={sum(train[3])}  neg={len(train[3])-sum(train[3])})")
    logger.info(f"  Test        : {len(test[0])}   "
                f"(pos={sum(test[3])}   neg={len(test[3])-sum(test[3])})")
    logger.info(f"BigWig used   : {args.bigwig_file or 'No (BERTsnr labels used for FCN)'}")
    logger.info(f"FCN smooth    : {args.smooth}  window={args.smooth_window}  "
                f"threshold={args.fcn_threshold}")
    logger.info("Output layout:")
    logger.info(f"  {args.output_path}/bertsnr/train/  {{sequence, label, seq_label, coord}}.txt")
    logger.info(f"  {args.output_path}/bertsnr/test/   {{sequence, label, seq_label, coord}}.txt")
    logger.info(f"  {args.output_path}/fcn/train.npz   data=[N,4,L]  signal=[N,1,L]  label=[N,1]")
    logger.info(f"  {args.output_path}/fcn/test.npz    data=[N,4,L]  signal=[N,1,L]  label=[N,1]")
    logger.info("=" * 60)

    # Save summary text
    summary_path = os.path.join(args.output_path, "dataset_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Dataset summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Seq length      : {args.seq_length}\n")
        f.write(f"Negative ratio  : {args.negative_ratio}\n")
        f.write(f"Expand bp       : {args.expand_bp}\n")
        f.write(f"Max N fraction  : {args.max_n_fraction}\n")
        f.write(f"Test ratio      : {args.test_ratio}\n")
        f.write(f"BigWig file     : {args.bigwig_file or 'None'}\n")
        f.write(f"FCN smooth      : {args.smooth}  window={args.smooth_window}\n")
        f.write(f"FCN threshold   : {args.fcn_threshold}\n\n")
        f.write(f"Total samples   : {n_total}\n")
        f.write(f"  Positive      : {n_pos}\n")
        f.write(f"  Negative      : {neg_found}\n\n")
        f.write(f"Train split     : {len(train[0])}\n")
        f.write(f"  pos={sum(train[3])}  neg={len(train[3])-sum(train[3])}\n\n")
        f.write(f"Test split      : {len(test[0])}\n")
        f.write(f"  pos={sum(test[3])}   neg={len(test[3])-sum(test[3])}\n")
    logger.info(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()