#!/usr/bin/env python3

import random
import argparse
import logging
import os
import numpy as np
from Bio import SeqIO

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


# Load genome

def load_genome(genome_file):
    logger.info(f"Loading genome from {genome_file}")
    genome = {}
    for record in SeqIO.parse(genome_file, "fasta"):
        genome[record.id] = str(record.seq).upper()
    logger.info(f"Loaded {len(genome)} chromosomes")
    return genome


# Read BED

def read_bed(bed_file):
    logger.info(f"Reading BED file: {bed_file}")
    peaks = []
    with open(bed_file) as f:
        for line in f:
            if line.startswith("#") or line.startswith("track"):
                continue
            chrom, start, end = line.strip().split()[:3]
            peaks.append((chrom, int(start), int(end)))
    logger.info(f"Found {len(peaks)} regions")
    return peaks


# Build exclusion index
# For each chrom, store a sorted list of (expanded_start, expanded_end) intervals
# to quickly check overlaps when sampling negatives.

def build_exclusion_index(peaks, expand_bp=0):
    """
    Build a per-chromosome dict of excluded intervals.
    expand_bp: how many bp to expand each peak on both sides before excluding.
    """
    index = {}
    for chrom, start, end in peaks:
        exc_start = max(0, start - expand_bp)
        exc_end = end + expand_bp
        index.setdefault(chrom, []).append((exc_start, exc_end))
    # Sort intervals per chrom for efficient overlap checking
    for chrom in index:
        index[chrom].sort()
    return index


def overlaps_excluded(chrom, start, end, exclusion_index):
    """
    Return True if [start, end) overlaps any excluded interval on chrom.
    Uses binary search for efficiency.
    """
    if chrom not in exclusion_index:
        return False
    intervals = exclusion_index[chrom]
    # Binary search: find first interval whose end > start
    lo, hi = 0, len(intervals)
    while lo < hi:
        mid = (lo + hi) // 2
        if intervals[mid][1] <= start:
            lo = mid + 1
        else:
            hi = mid
    # Check from lo onward while interval start < end
    for i in range(lo, len(intervals)):
        if intervals[i][0] >= end:
            break
        return True  # overlap found
    return False


# Extract sequence + labels + coords

def extract_sequence_and_motif_labels(genome, chrom, start, end, seq_length):
    if chrom not in genome:
        return None, None, None

    center = (start + end) // 2
    peak_width = end - start

    min_offset = peak_width // 2
    max_offset = seq_length - (peak_width - peak_width // 2)

    if min_offset > max_offset:
        offset = random.randint(0, seq_length - 1)
    else:
        offset = random.randint(min_offset, max_offset - 1)

    seq_start = center - offset
    seq_end = seq_start + seq_length

    chrom_len = len(genome[chrom])
    if seq_start < 0:
        seq_start = 0
        seq_end = seq_length
    if seq_end > chrom_len:
        seq_end = chrom_len
        seq_start = max(0, chrom_len - seq_length)

    sequence = genome[chrom][seq_start:seq_end]
    if len(sequence) < seq_length:
        sequence += "N" * (seq_length - len(sequence))

    labels = np.zeros(seq_length, dtype=int)
    rel_start = start - seq_start
    rel_end = end - seq_start
    if 0 <= rel_start < seq_length and 0 <= rel_end <= seq_length:
        labels[rel_start:rel_end] = 1

    coords = (chrom, seq_start, seq_end, "+")
    return sequence, labels.tolist(), coords


# Sample a random negative region

def sample_negative_region(genome, exclusion_index, seq_length,
                            max_n_fraction=0.1, max_attempts=1000,
                            expand_bp=0):
    """
    Sample a random genomic window of `seq_length` that:
      - Does not overlap any excluded (peak-derived) interval expanded by expand_bp.
      - Does not contain more than max_n_fraction * seq_length N bases.

    Returns (sequence, coords) or (None, None) if no valid region found.
    """
    chroms = list(genome.keys())
    chrom_lengths = {c: len(genome[c]) for c in chroms}
    # Weight chromosomes by length for uniform positional sampling
    total_len = sum(chrom_lengths.values())
    weights = [chrom_lengths[c] / total_len for c in chroms]

    for _ in range(max_attempts):
        chrom = random.choices(chroms, weights=weights, k=1)[0]
        chrom_len = chrom_lengths[chrom]
        if chrom_len < seq_length:
            continue
        start = random.randint(0, chrom_len - seq_length)
        end = start + seq_length

        # Expand coordinates for overlap check
        exc_start = max(0, start - expand_bp)
        exc_end = end + expand_bp

        if overlaps_excluded(chrom, exc_start, exc_end, exclusion_index):
            continue

        sequence = genome[chrom][start:end]
        n_count = sequence.count("N") + sequence.count("n")
        if n_count / seq_length > max_n_fraction:
            continue

        coords = (chrom, start, end, "+")
        return sequence, coords

    logger.warning(f"Could not find a valid negative region after {max_attempts} attempts")
    return None, None


# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bed_file", required=True)
    parser.add_argument("--genome_file", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--seq_length", type=int, default=100)
    parser.add_argument("--negative_ratio", type=int, default=1)
    parser.add_argument("--expand_bp", type=int, default=3000,
                        help="Expand peak coordinates by this many bp on each side "
                             "when excluding regions for negative sampling")
    parser.add_argument("--max_n_fraction", type=float, default=0,
                        help="Maximum fraction of N bases allowed in a negative window (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_path, exist_ok=True)

    genome = load_genome(args.genome_file)
    peaks = read_bed(args.bed_file)

    # Build exclusion index (peaks expanded by --expand_bp)
    logger.info(f"Building exclusion index (expand_bp={args.expand_bp})")
    exclusion_index = build_exclusion_index(peaks, expand_bp=args.expand_bp)

    pos_seqs, pos_labels, pos_coords = [], [], []

    logger.info("Extracting positive samples")
    for chrom, start, end in peaks:
        seq, lab, coord = extract_sequence_and_motif_labels(
            genome, chrom, start, end, args.seq_length
        )
        if seq is not None:
            pos_seqs.append(seq)
            pos_labels.append(lab)
            pos_coords.append(coord)

    logger.info(f"Positive samples: {len(pos_seqs)}")

    # Interleaved POS → NEG → POS → NEG
    all_data = []
    neg_found = 0
    neg_failed = 0

    logger.info("Sampling negative regions")
    for i in range(len(pos_seqs)):
        # positive
        all_data.append((pos_seqs[i], pos_labels[i], pos_coords[i]))

        # negatives derived from random genomic regions
        for _ in range(args.negative_ratio):
            neg_seq, neg_coord = sample_negative_region(
                genome,
                exclusion_index,
                args.seq_length,
                max_n_fraction=args.max_n_fraction,
                expand_bp=args.expand_bp,
            )
            if neg_seq is not None:
                neg_lab = [0] * args.seq_length
                all_data.append((neg_seq, neg_lab, neg_coord))
                neg_found += 1
            else:
                neg_failed += 1

    sequences, labels, coords = zip(*all_data)

    # Write outputs
    seq_file = os.path.join(args.output_path, "sequence.txt")
    lab_file = os.path.join(args.output_path, "label.txt")
    coord_file = os.path.join(args.output_path, "coord.txt")

    with open(seq_file, "w") as f:
        for s in sequences:
            f.write(s + "\n")

    with open(lab_file, "w") as f:
        for l in labels:
            f.write("".join(map(str, l)) + "\n")

    with open(coord_file, "w") as f:
        for c in coords:
            f.write(f"{c[0]}\t{c[1]}\t{c[2]}\t{c[3]}\n")

    logger.info("=" * 60)
    logger.info("DONE")
    logger.info(f"Total samples written: {len(sequences)}")
    logger.info(f"Positive samples: {len(pos_seqs)}")
    logger.info(f"Negative samples found: {neg_found}")
    if neg_failed:
        logger.warning(f"Negative samples failed (no valid region): {neg_failed}")
    logger.info("Order: POS → NEG → POS → NEG")
    logger.info("Negative source: random genomic regions (non-overlapping with peaks)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()


