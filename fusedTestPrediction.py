import os
import argparse
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pyfaidx
import pyBigWig
from fused_model import FusedModel


# CONFIG

THRESHOLD        = 0.5
PEAK_BPS         = 10
MAX_LEN          = 512

# additional BigWig files used during training — leave empty if none
SMOOTH           = True
SMOOTH_WINDOW    = 15


# HELPER FUNCTIONS

def one_hot(fasta, chrom, start, end):
    seq_dict = {
        'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1],
        'a': [1,0,0,0], 'c': [0,1,0,0], 'g': [0,0,1,0], 't': [0,0,0,1],
    }
    seq = str(fasta[chrom][start:end])
    arr = [seq_dict.get(c, [0,0,0,0]) for c in seq]
    return np.array(arr).T   # [4, seq_len]


def smooth_avg(x, window):
    return np.array([
        np.mean(x[j:j + window])
        for j in range(len(x) - window + 1)
    ])


def get_bigwig_signal(bw, chrom, start, end, seq_length):
    """Extract signal from an already-open pyBigWig handle, pad/crop to seq_length."""
    start = max(0, start)
    vals  = np.array(bw.values(chrom, start, end), dtype=np.float64)
    vals[np.isnan(vals)] = 0.0
    if len(vals) < seq_length:
        vals = np.pad(vals, (0, seq_length - len(vals)))
    else:
        vals = vals[:seq_length]
    return vals


def extract_extra_features(bw_handles, chrom, start, end, seq_length,
                            smooth=True, window=15):
    """
    Extract log10-normalised signal from each BigWig for one window.
    Returns array of shape [F, seq_length], matching training-time extraction.
    """
    feats = []
    for bw in bw_handles:
        if smooth:
            raw = get_bigwig_signal(bw, chrom,
                                    start - window // 2,
                                    end   + window // 2,
                                    seq_length + window - 1)
            sig = smooth_avg(raw, window)
        else:
            sig = get_bigwig_signal(bw, chrom, start, end, seq_length)

        if len(sig) < seq_length:
            sig = np.pad(sig, (0, seq_length - len(sig)))
        else:
            sig = sig[:seq_length]

        feats.append(np.log10(1.0 + sig).astype(np.float32))

    return np.stack(feats, axis=0)   # [F, seq_length]


def has_continuous_peak(signal, min_len=PEAK_BPS):
    count = 0
    for v in signal:
        if v == 1:
            count += 1
            if count >= min_len:
                return True
        else:
            count = 0
    return False


# DATASET

class FusedTestDataset(Dataset):
    def __init__(self, bed_file, fasta_file, tokenizer,
                 extra_bigwigs=None, smooth=True, smooth_window=15,
                 max_len=MAX_LEN):
        self.bed_lines     = [l.strip() for l in open(bed_file)
                              if l.strip() and not l.startswith("#")]
        self.fasta         = pyfaidx.Fasta(fasta_file)
        self.tokenizer     = tokenizer
        self.max_len       = max_len
        self.smooth        = smooth
        self.smooth_window = smooth_window

        # open BigWig handles once — same files and order as training
        if extra_bigwigs:
            self.bw_handles = [pyBigWig.open(f) for f in extra_bigwigs]
            self.n_features = len(self.bw_handles)
        else:
            self.bw_handles = []
            self.n_features = 0

    def __len__(self):
        return len(self.bed_lines)

    def __getitem__(self, idx):
        parts = self.bed_lines[idx].split()
        chrom, start, end = parts[0], int(parts[1]), int(parts[2])
        seq_length = end - start

        # one-hot [4, L]
        one_hot_arr = one_hot(self.fasta, chrom, start, end)
        fcn_input   = torch.tensor(one_hot_arr, dtype=torch.float32)

        # concatenate extra features if present — mirrors FusedDNADataset
        if self.bw_handles:
            extra = extract_extra_features(
                self.bw_handles, chrom, start, end, seq_length,
                smooth=self.smooth, window=self.smooth_window,
            )                                                       # [F, L]
            extra_t   = torch.tensor(extra, dtype=torch.float32)
            fcn_input = torch.cat([fcn_input, extra_t], dim=0)     # [4+F, L]

        # BERT input
        raw_seq  = str(self.fasta[chrom][start:end]).upper()
        encoding = self.tokenizer(
            raw_seq,
            max_length             = self.max_len,
            padding                = 'max_length',
            truncation             = True,
            return_tensors         = 'pt',
            return_offsets_mapping = True,
        )

        return {
            'fcn_input'      : fcn_input,
            'input_ids'      : encoding['input_ids'].squeeze(0),
            'attention_mask' : encoding['attention_mask'].squeeze(0),
            'offset_mapping' : encoding['offset_mapping'].squeeze(0),
            'seq_id'         : f"{chrom}:{start}-{end}",
        }

    def close(self):
        for bw in self.bw_handles:
            bw.close()


# COMMAND LINE

parser = argparse.ArgumentParser()
parser.add_argument("--fasta", required=True)
parser.add_argument("--bed", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--bert", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--extra-bigwig", nargs="*", default=[])
parser.add_argument("--threshold", type=float, default=THRESHOLD)
parser.add_argument("--peak-bps", type=int, default=PEAK_BPS)
parser.add_argument("--max-len", type=int, default=MAX_LEN)
parser.add_argument("--smooth", action=argparse.BooleanOptionalAction, default=SMOOTH)
parser.add_argument("--smooth-window", type=int, default=SMOOTH_WINDOW)
args = parser.parse_args()

FASTA_FILE       = args.fasta
BED_FILE         = args.bed
MODEL_FILE       = args.model
BERT_PATH        = args.bert
OUT_DIR          = args.output
EXTRA_BIGWIGS    = args.extra_bigwig
THRESHOLD        = args.threshold
PEAK_BPS         = args.peak_bps
MAX_LEN          = args.max_len
SMOOTH           = args.smooth
SMOOTH_WINDOW    = args.smooth_window


# INFERENCE

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(BERT_PATH, trust_remote_code=True)

# dataloader — batch_size=1 since sequences have variable lengths
test_dataset = FusedTestDataset(
    BED_FILE, FASTA_FILE, tokenizer,
    extra_bigwigs = EXTRA_BIGWIGS,
    smooth        = SMOOTH,
    smooth_window = SMOOTH_WINDOW,
    max_len       = MAX_LEN,
)
test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

# load model
model = FusedModel(motiflen=10, n_features=test_dataset.n_features)
checkpoint = torch.load(MODEL_FILE, map_location=device)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

# output files
os.makedirs(OUT_DIR, exist_ok=True)

summary_with     = open(os.path.join(OUT_DIR, "test_with_peak.tsv"),         "w")
summary_without  = open(os.path.join(OUT_DIR, "test_without_peak.tsv"),      "w")
fullpred_with    = open(os.path.join(OUT_DIR, "test_with_peak_full.tsv"),    "w")
fullpred_without = open(os.path.join(OUT_DIR, "test_without_peak_full.tsv"), "w")

for f in [summary_with, summary_without]:
    f.write("chr\tstart\tend\tlabel\n")
for f in [fullpred_with, fullpred_without]:
    f.write("chr\tstart\tend\tfused_pred\n")

count_with = count_without = 0

with torch.no_grad():
    for batch in test_loader:
        fcn_input      = batch['fcn_input'].to(device)
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        offset_mapping = batch['offset_mapping']
        seq_id         = batch['seq_id'][0]

        # forward pass
        logits_seq, fused_logits = model(
            fcn_input      = fcn_input,
            input_ids      = input_ids,
            attention_mask = attention_mask,
            offset_mapping = offset_mapping,
        )

        # per-nucleotide fused prediction
        nuc_logits = fused_logits[0]                               # [L, 1]
        nuc_probs  = torch.sigmoid(nuc_logits).squeeze(-1)        # [L]
        nuc_pred   = (nuc_probs >= THRESHOLD).int().cpu().numpy() # [L]

        # peak calling — PEAK_BPS consecutive 1s
        has_peak = has_continuous_peak(nuc_pred, PEAK_BPS)
        label    = "with_peak" if has_peak else "without_peak"

        # parse location
        chrom, positions = seq_id.split(":")
        start, end = map(int, positions.split("-"))

        pred_str = ",".join(map(str, nuc_pred))

        if has_peak:
            count_with += 1
            summary_with.write(f"{chrom}\t{start}\t{end}\t{label}\n")
            fullpred_with.write(f"{chrom}\t{start}\t{end}\t{pred_str}\n")
        else:
            count_without += 1
            summary_without.write(f"{chrom}\t{start}\t{end}\t{label}\n")
            fullpred_without.write(f"{chrom}\t{start}\t{end}\t{pred_str}\n")

summary_with.close()
summary_without.close()
fullpred_with.close()
fullpred_without.close()
test_dataset.close()

print("Done! Inference completed.")
print(f"Sequences with peak    : {count_with}")
print(f"Sequences without peak : {count_without}")
print(f"Total sequences        : {count_with + count_without}")