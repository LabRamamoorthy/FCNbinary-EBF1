import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pyfaidx
from fused_model import FusedModel


# CONFIG

FASTA_FILE = "Genome/mm9.fa"
BED_FILE   = "test_predict/input/ATAC_Extended.bed"
MODEL_FILE = "fusedModels/EBF_500_0.7_0.3_run1/best_model.pt"
BERT_PATH  = "/home/ubuntu/data/dnabert2"
THRESHOLD  = 0.5    # prediction threshold
PEAK_BPS   = 10     # minimum consecutive 1s to call a peak
MAX_LEN    = 512    # BERT max length


# HELPER FUNCTIONS

def one_hot(fasta, chrom, start, end):
    seq_dict = {
        'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1],
        'a': [1,0,0,0], 'c': [0,1,0,0], 'g': [0,0,1,0], 't': [0,0,0,1],
    }
    seq = str(fasta[chrom][start:end])
    arr = [seq_dict.get(c, [0,0,0,0]) for c in seq]
    return np.array(arr).T   # [4, seq_len]


def has_continuous_peak(signal, min_len=PEAK_BPS):
    """Returns True if signal has at least min_len consecutive 1s."""
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
    def __init__(self, bed_file, fasta_file, tokenizer, max_len=MAX_LEN):
        self.bed_lines = [l.strip() for l in open(bed_file)
                          if l.strip() and not l.startswith("#")]
        self.fasta     = pyfaidx.Fasta(fasta_file)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.bed_lines)

    def __getitem__(self, idx):
        parts = self.bed_lines[idx].split()
        chrom, start, end = parts[0], int(parts[1]), int(parts[2])

        # FCN input — 4-channel one-hot only
        seq       = one_hot(self.fasta, chrom, start, end)   # [4, L]
        fcn_input = torch.tensor(seq, dtype=torch.float32)   # [4, L]

        # BERT input — tokenize raw sequence
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


# INFERENCE

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(BERT_PATH, trust_remote_code=True)

# load model
model = FusedModel(motiflen=10, n_features=0)
checkpoint = torch.load(MODEL_FILE, map_location=device)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

# dataloader — batch_size=1 since sequences have variable lengths
test_dataset = FusedTestDataset(BED_FILE, FASTA_FILE, tokenizer)
test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

# output files
OUT_DIR = "test_predict/output/fused_0.7_0.3"
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
        offset_mapping = batch['offset_mapping']   # stays on CPU
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

print("Done! Inference completed.")
print(f"Sequences with peak    : {count_with}")
print(f"Sequences without peak : {count_without}")
print(f"Total sequences        : {count_with + count_without}")