import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pyfaidx
import pyBigWig
from FCNmotif import FCNsignalMod
import argparse

# CONFIG

parser = argparse.ArgumentParser(description="Run FCN motif signal inference")

parser.add_argument("--chrom", required=True,
                    help="Chromosome sizes file, e.g. Genome/mm9.chrom.sizes")

parser.add_argument("--fasta", required=True,
                    help="FASTA genome file, e.g. Genome/mm9.fa")

parser.add_argument("--bed", required=True,
                    help="Input BED file")

parser.add_argument("--model", required=True,
                    help="Model checkpoint (.pth)")

parser.add_argument("--output", required=True,
                    help="Output directory")

parser.add_argument("--features", nargs="*", default=[],
                    help="Optional additional BigWig feature files")

parser.add_argument("--threshold", type=float, default=0.5,
                    help="Prediction threshold (default: 0.5)")

parser.add_argument("--peak-bps", type=int, default=12,
                    help="Minimum continuous peak length (default: 12)")

parser.add_argument("--window", type=int, default=15,
                    help="Smoothing window (default: 15)")

parser.add_argument("--smooth", action="store_true",
                    help="Apply smoothing to additional features")

args = parser.parse_args()

CHROM_FILE = args.chrom
FASTA_FILE = args.fasta
BED_FILE = args.bed
MODEL_FILE = args.model

FEATURE_FILES = args.features
THRESHOLD = args.threshold
PEAK_BPS = args.peak_bps

WINDOW = args.window
SMOOTH = args.smooth


# HELPER FUNCTIONS

# Load chromosome sizes
with open(CHROM_FILE) as f:
    for i in f:
        chrom, size = i.strip().split()
        CHROM[chrom] = int(size)

# One-hot encoding
def one_hot(sequence_dict, chrom, start, end):
    seq_dict = {'A':[1,0,0,0], 'C':[0,1,0,0], 'G':[0,0,1,0], 'T':[0,0,0,1],
                'a':[1,0,0,0], 'c':[0,1,0,0], 'g':[0,0,1,0], 't':[0,0,0,1]}
    
    seq_len = end - start  # expected length (5001)
    
    # Clamp to valid chromosome coordinates
    valid_start = max(0, start)
    valid_end   = min(end, CHROM.get(chrom, end))
    
    seq  = str(sequence_dict[chrom][valid_start:valid_end])
    arr  = [seq_dict.get(c, [0,0,0,0]) for c in seq]
    
    left_pad  = valid_start - start      
    right_pad = seq_len - left_pad - len(arr)
    
    arr = [[0,0,0,0]] * left_pad + arr + [[0,0,0,0]] * right_pad
    
    result = np.array(arr, dtype=np.float32).T  
    assert result.shape == (4, seq_len), f"Shape mismatch: {result.shape} for {chrom}:{start}-{end}"
    return result

# Read bigWig
def GetbigWig(file, chrom, start, end):
    seq_len = end - start

    valid_start = max(0, start)
    valid_end   = min(end, CHROM.get(chrom, end))

    bw   = pyBigWig.open(file)
    vals = np.array(bw.values(chrom, valid_start, valid_end))
    bw.close()
    vals[np.isnan(vals)] = 0.

    left_pad  = valid_start - start
    right_pad = seq_len - left_pad - len(vals)
    vals = np.concatenate([
        np.zeros(left_pad),
        vals,
        np.zeros(right_pad)
    ])

    assert len(vals) == seq_len, f"Length mismatch: {len(vals)} for {chrom}:{start}-{end}"
    return vals

# Smoothing
def Smoothavg(x, window):
    avg = []
    for j in range(len(x) - window + 1):
        avg.append(np.mean(x[j:(j+window)]))
    return np.array(avg)

# Extract optional extra features
def extract_additional_features(bed_lines, feature_files, location_func):
    if not feature_files:
        return [None] * len(bed_lines)
    all_feats = []
    for line in bed_lines:
        chrom, start_s, end_s = line.strip().split()[:3]
        start, end = int(start_s), int(end_s)
        start_f, end_f = location_func(chrom, start, end)
        feature_list = []
        for feat_file in feature_files:
            start_c = max(0, start_f - WINDOW//2)
            end_c = min(end_f + WINDOW//2, CHROM.get(chrom, end_f + WINDOW//2))
            vals = GetbigWig(feat_file, chrom, start_c, end_c)
            vals[np.isnan(vals)] = 0.
            if SMOOTH and len(vals) >= WINDOW:
                vals = Smoothavg(vals, WINDOW)
            vals = np.log10(1 + vals)
            feature_list.append(vals.astype(np.float32))
        feats = np.stack(feature_list, axis=0) if feature_list else None
        all_feats.append(feats)
    return all_feats


# DATASET

def has_continuous_peak(signal, min_len=6):
    count = 0
    for v in signal:
        if v == 1:
            count += 1
            if count >= min_len:
                return True
        else:
            count = 0
    return False

class TestSeqDataset(Dataset):
    def __init__(self, bed_file, fasta_file, extra_features=None):
        self.bed_lines = [l.strip() for l in open(bed_file) if l.strip() and not l.startswith("#")]
        self.fasta = pyfaidx.Fasta(fasta_file)
        self.extra = extra_features

    def __len__(self):
        return len(self.bed_lines)

    def __getitem__(self, idx):
        chrom, start_s, end_s = self.bed_lines[idx].split()[:3]
        start, end = int(start_s), int(end_s)
        seq = one_hot(self.fasta, chrom, start, end)
        if self.extra is not None and self.extra[idx] is not None:
            feats = self.extra[idx]
            min_len = min(seq.shape[1], feats.shape[1])
            seq = np.concatenate([seq[:, :min_len], feats[:, :min_len]], axis=0)
        return torch.tensor(seq, dtype=torch.float32), f"{chrom}:{start}-{end}"


# MODEL LOADING

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
location_func = lambda chrom, s, e: (s, e)

bed_lines = [l for l in open(BED_FILE) if l.strip() and not l.startswith("#")]
extra_feats = extract_additional_features(bed_lines, FEATURE_FILES, location_func)
n_features = 0 if extra_feats is None or extra_feats[0] is None else extra_feats[0].shape[0]

model = FCNsignalMod(motiflen=12, n_features=n_features)
checkpoint = torch.load(MODEL_FILE, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()


# INFERENCE

test_dataset = TestSeqDataset(BED_FILE, FASTA_FILE, extra_feats)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

OUTPUT_DIR = args.output

os.makedirs(OUTPUT_DIR, exist_ok=True)

summary_with = open(os.path.join(OUTPUT_DIR, "test_with_peak.tsv"), "w")
summary_without = open(os.path.join(OUTPUT_DIR, "test_without_peak.tsv"), "w")
fullpred_with = open(os.path.join(OUTPUT_DIR, "test_with_peak_full.tsv"), "w")
fullpred_without = open(os.path.join(OUTPUT_DIR, "test_without_peak_full.tsv"), "w")

for f in [summary_with, summary_without]:
    f.write("chr\tstart\tend\tlabel\n")
for f in [fullpred_with, fullpred_without]:
    f.write("chr\tstart\tend\tpredictions\n")

count_with, count_without = 0, 0

with torch.no_grad():
    for x, seq_id in test_loader:
        x = x.to(device)
        logits = model(x)
        pred_signal = (torch.sigmoid(logits) >= THRESHOLD).int().cpu().numpy().flatten()

        has_peak = has_continuous_peak(pred_signal, PEAK_BPS)
        if has_peak:
            count_with += 1
            label = f"with_peak_{count_with:05d}"
        else:
            count_without += 1
            label = f"without_peak_{count_without:05d}"
        chrom, positions = seq_id[0].split(":")
        start, end = map(int, positions.split("-"))

        pred_str = ",".join(map(str, pred_signal))

        if has_peak:
            summary_with.write(f"{chrom}\t{start}\t{end}\t{label}\n")
            fullpred_with.write(f"{chrom}\t{start}\t{end}\t{pred_str}\n")
        else:
            summary_without.write(f"{chrom}\t{start}\t{end}\t{label}\n")
            fullpred_without.write(f"{chrom}\t{start}\t{end}\t{pred_str}\n")

summary_with.close()
summary_without.close()
fullpred_with.close()
fullpred_without.close()

print("Done! Inference completed.")
print(f"Sequences with peak: {count_with}")
print(f"Sequences without peak: {count_without}")
print(f"Total sequences: {count_with + count_without}")
