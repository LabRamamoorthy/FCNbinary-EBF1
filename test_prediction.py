import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pyfaidx
import pyBigWig
from FCNmotif import FCNsignalMod


# CONFIG

SMOOTH = True
WINDOW = 15
CHROM = {}
CHROM_FILE = "Genome/mm9.chrom.sizes"
FASTA_FILE = "Genome/mm9.fa"
BED_FILE = "test_predict/Input/AWT3_peaks.bed"
MODEL_FILE = "models/EBF_binary_test/model_best_ebf.pth"
THRESHOLD = 0.5         # Prediction threshold
PEAK_BPS = 6           
FEATURE_FILES = []      # optional extra features
TRUE_BW_FILE = "EBF/p-value.bw"  
TRUE_THRESHOLD = 0.1    # binary threshold for true signal


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
    seq = str(sequence_dict[chrom][start:end])
    arr = [seq_dict.get(c, [0,0,0,0]) for c in seq]
    return np.array(arr).T  # shape [4, seq_len]

# Read bigWig
def GetbigWig(file, chrom, start, end):
    start = max(0, start)
    end = min(end, CHROM.get(chrom, end))
    bw = pyBigWig.open(file)
    vals = np.array(bw.values(chrom, start, end))
    bw.close()
    vals[np.isnan(vals)] = 0.
    return vals

# Smoothing
def Smoothavg(x, window):
    if len(x) < window:
        return x
    cumsum = np.cumsum(np.insert(x, 0, 0))
    smooth = (cumsum[window:] - cumsum[:-window]) / float(window)
    pad_left = window // 2
    pad_right = len(x) - len(smooth) - pad_left
    return np.pad(smooth, (pad_left, pad_right), mode='edge')

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
            if SMOOTH:
                vals = Smoothavg(vals, WINDOW)
            vals = np.log10(1 + vals)
            feature_list.append(vals.astype(np.float32))
        feats = np.stack(feature_list, axis=0) if feature_list else None
        all_feats.append(feats)
    return all_feats

# Extract true signal: smooth, normalize, binary
def get_true_signal(chrom, start, end):
    vals = GetbigWig(TRUE_BW_FILE, chrom, start, end)
    if SMOOTH and len(vals) >= WINDOW:
        vals = Smoothavg(vals, WINDOW)
    vals = np.log10(1 + vals)
    vals = (vals >= TRUE_THRESHOLD).astype(int)
    return vals


# DATASET

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

os.makedirs("test_predict/output/AWT3", exist_ok=True)

summary_with = open("test_predict/output/AWT3/test_with_peak.tsv", "w")
summary_without = open("test_predict/output/AWT3/test_without_peak.tsv", "w")
fullpred_with = open("test_predict/output/AWT3/test_with_peak_full.tsv", "w")
fullpred_without = open("test_predict/output/AWT3/test_without_peak_full.tsv", "w")

for f in [summary_with, summary_without]:
    f.write("chr\tstart\tend\tlabel\n")
for f in [fullpred_with, fullpred_without]:
    f.write("chr\tstart\tend\tpredictions\ttrue_signal\n")

count_with, count_without = 0, 0

with torch.no_grad():
    for x, seq_id in test_loader:
        x = x.to(device)
        logits = model(x)
        pred_signal = (torch.sigmoid(logits) >= THRESHOLD).int().cpu().numpy().flatten()

        has_peak = np.sum(pred_signal) >= PEAK_BPS
        label = "with_peak" if has_peak else "without_peak"

        chrom, positions = seq_id[0].split(":")
        start, end = map(int, positions.split("-"))

        pred_str = ",".join(map(str, pred_signal))
        true_signal = get_true_signal(chrom, start, end)
        true_str = ",".join(map(str, true_signal))

        if has_peak:
            count_with += 1
            summary_with.write(f"{chrom}\t{start}\t{end}\t{label}\n")
            fullpred_with.write(f"{chrom}\t{start}\t{end}\t{pred_str}\t{true_str}\n")
        else:
            count_without += 1
            summary_without.write(f"{chrom}\t{start}\t{end}\t{label}\n")
            fullpred_without.write(f"{chrom}\t{start}\t{end}\t{pred_str}\t{true_str}\n")

summary_with.close()
summary_without.close()
fullpred_with.close()
fullpred_without.close()

print(" Done! Inference completed.")
print(f"Sequences with peak: {count_with}")
print(f"Sequences without peak: {count_without}")
print(f"Total sequences: {count_with + count_without}")



# import os
# import argparse
# import numpy as np
# import torch
# from Bio import SeqIO
# from FCNmotif import FCNsignalMod
# torch.multiprocessing.set_sharing_strategy('file_system')


# def one_hot(sequence_dict, chrom, start, end):
#     seq_dict = {'A':[1, 0, 0, 0], 'G':[0, 0, 1, 0],
#                 'C':[0, 1, 0, 0], 'T':[0, 0, 0, 1],
#                 'a':[1, 0, 0, 0], 'g':[0, 0, 1, 0],
#                 'c':[0, 1, 0, 0], 't':[0, 0, 0, 1]}
#     temp = []
#     seq = str(sequence_dict[chrom].seq[start:end])
#     for c in seq:
#         temp.append(seq_dict.get(c, [0, 0, 0, 0]))
#     temp = np.array([temp], dtype=np.float32)
#     return temp.transpose((0, 2, 1))


# def predict_regions(model, sequence_dict, bed_lines, device):
#     with_peak = []
#     without_peak = []

#     for line in bed_lines:
#         chrom, start, end = line.strip().split()[:3]
#         start = int(start)
#         end = int(end)
#         seq_len = end - start
#         if seq_len == 0:
#             continue

#         X_data = torch.from_numpy(one_hot(sequence_dict, chrom, start, end)).float().to(device)
#         with torch.no_grad():
#             pred = model(X_data).view(-1).cpu().numpy()

    
#         has_peak = np.any(pred == 1)

#         line_data = [chrom, start, end] + [int(v) for v in pred] 

#         if has_peak:
#             with_peak.append(line_data)
#         else:
#             without_peak.append(line_data)

#     return with_peak, without_peak


# def write_output(file_path, data):
#     with open(file_path, 'w') as f:
#         for row in data:
#             f.write('\t'.join(map(str, row)) + '\n')


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-i', '--input_bed', type=str, required=True, help='Input BED file')
#     parser.add_argument('-m', '--model_path', type=str, required=True, help='Trained model .pth file')
#     parser.add_argument('-f', '--fasta', type=str, required=True, help='Reference genome FASTA')
#     parser.add_argument('-o', '--outdir', type=str, default='prediction_output', help='Output folder')
#     # parser.add_argument('-t', '--threshold', type=float, default=0.5, help='Peak signal threshold')
#     parser.add_argument('-g', '--gpu', type=str, default='0', help='GPU ID')
#     args = parser.parse_args()

#     os.makedirs(args.outdir, exist_ok=True)
#     device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")

#     # Load genome
#     sequence_dict = SeqIO.to_dict(SeqIO.parse(open(args.fasta), 'fasta'))

#     # Load model
#     checkpoint = torch.load(args.model_path, map_location=device)
#     state_dict = checkpoint['model_state_dict']

#     model = FCNsignalMod(motiflen=12, n_features=0)  # match your trained model
#     model.to(device)

#     # Only load matching layers
#     model_dict = model.state_dict()
#     pretrained_dict = {k: v for k, v in state_dict.items() 
#                     if k in model_dict and v.size() == model_dict[k].size()}
#     model_dict.update(pretrained_dict)
#     model.load_state_dict(model_dict)

#     model.eval()

#     # Optional: print loaded vs skipped layers
#     for k in model_dict:
#         if k in pretrained_dict:
#             print(f"Loaded: {k}")
#         else:
#             print(f"Skipped: {k}")

#     # Load BED file
#     with open(args.input_bed) as f:
#         bed_lines = f.readlines()

#     with_peak, without_peak = predict_regions(model, sequence_dict, bed_lines, device ) #args.threshold

#     # Output files
#     write_output(os.path.join(args.outdir, 'with_peak_WT2.bed'), with_peak)
#     write_output(os.path.join(args.outdir, 'without_peak_WT2.bed'), without_peak)

#     print(f"Predicted {len(with_peak)} regions with peaks and {len(without_peak)} without peaks.")


# if __name__ == '__main__':
#     main()


# import os
# import argparse
# import numpy as np
# import torch
# import pyBigWig
# from Bio import SeqIO
# from FCNmotif import FCNsignalMod

# torch.multiprocessing.set_sharing_strategy('file_system')

# def one_hot(sequence_dict, chrom, start, end):
#     seq_dict = {
#         'A':[1,0,0,0], 'G':[0,0,1,0], 'C':[0,1,0,0], 'T':[0,0,0,1],
#         'a':[1,0,0,0], 'g':[0,0,1,0], 'c':[0,1,0,0], 't':[0,0,0,1]
#     }
#     seq = str(sequence_dict[chrom].seq[start:end])
#     temp = [seq_dict.get(c, [0,0,0,0]) for c in seq]
#     temp = np.array([temp], dtype=np.float32)  # [1, L, 4]
#     return temp.transpose((0, 2, 1))           # [1, 4, L]

# def get_bigwig_vals(bw, chrom, start, end):
#     vals = np.array(bw.values(chrom, start, end))
#     vals[np.isnan(vals)] = 0
#     return vals

# def smooth_avg(x, window):
#     if len(x) < window:
#         return x
#     return np.convolve(x, np.ones(window) / window, mode='same')

# def extract_additional_features_variable(bed_lines, feature_bw_handles, chrom_lengths, smooth=True, window=21):
#     all_feats = []
#     n_features = len(feature_bw_handles)

#     for line in bed_lines:
#         chrom, start_s, end_s = line.strip().split()[:3]
#         if chrom not in chrom_lengths:
#             all_feats.append(np.zeros((n_features, 0), dtype=np.float32))
#             continue

#         start, end = int(start_s), int(end_s)
#         feats_per_region = []

#         for bw in feature_bw_handles:
#             vals = get_bigwig_vals(bw, chrom, start, end)
#             if smooth:
#                 vals = smooth_avg(vals, window)
#             feats_per_region.append(np.log10(1 + vals))

#         feats_per_region = np.array(feats_per_region, dtype=np.float32)  # [F, L]
#         all_feats.append(feats_per_region)

#     return all_feats  # list of arrays [F, L]

# def predict_regions_variable(model, sequence_dict, bed_lines, add_features_list, device):
#     with_peak, without_peak = [], []

#     for i, line in enumerate(bed_lines):
#         chrom, start, end = line.strip().split()[:3]
#         start, end = int(start), int(end)

#         X_seq = torch.from_numpy(one_hot(sequence_dict, chrom, start, end)).float().to(device)
#         X_feats = torch.from_numpy(add_features_list[i]).unsqueeze(0).float().to(device)

#         with torch.no_grad():
#             X_combined = torch.cat([X_seq, X_feats], dim=1)
#             pred = model(X_combined)

#         pred_np = pred.view(-1).cpu().numpy()
#         has_peak = np.any(pred_np == 1)   # <-- fix here
#         row = [chrom, start, end] + [round(v, 3) for v in pred_np]

#         if has_peak:
#             with_peak.append(row)
#         else:
#             without_peak.append(row)

#     return with_peak, without_peak



# def write_output(file_path, data):
#     with open(file_path, 'w') as f:
#         for row in data:
#             f.write('\t'.join(map(str, row)) + '\n')

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-i', '--input_bed', type=str, required=True, help='Input BED file')
#     parser.add_argument('-m', '--model_path', type=str, required=True, help='Trained model .pth file')
#     parser.add_argument('-f', '--fasta', type=str, required=True, help='Reference genome FASTA')
#     parser.add_argument('--features', type=str, nargs='*', default=[], help='List of .bw files for extra features')
#     parser.add_argument('-o', '--outdir', type=str, default='prediction_output', help='Output folder')
#     # parser.add_argument('-t', '--threshold', type=float, default=0.5, help='Peak signal threshold')
#     parser.add_argument('-g', '--gpu', type=str, default='0', help='GPU ID')
#     parser.add_argument('--smooth', action='store_true', help='Apply smoothing to bigwig signals')
#     parser.add_argument('--window', type=int, default=21, help='Smoothing window size')
#     args = parser.parse_args()

#     os.makedirs(args.outdir, exist_ok=True)
#     device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")

#     # Load genome
#     sequence_dict = SeqIO.to_dict(SeqIO.parse(open(args.fasta), 'fasta'))
#     chrom_lengths = {chrom: len(seq) for chrom, seq in sequence_dict.items()}

#     # Load features
#     feature_bw_handles = [pyBigWig.open(f) for f in args.features]
#     add_features_list = extract_additional_features_variable(
#         bed_lines=open(args.input_bed).readlines(),
#         feature_bw_handles=feature_bw_handles,
#         chrom_lengths=chrom_lengths,
#         smooth=args.smooth,
#         window=args.window
#     )

#     # Load model
#     checkpoint = torch.load(args.model_path, map_location=device)
#     n_features = add_features_list[0].shape[0] if len(add_features_list) > 0 else 0
#     model = FCNsignalMod(motiflen=12, n_features=n_features)
#     model.to(device)

#     # Safe partial load to avoid size mismatch
#     state_dict = checkpoint['model_state_dict']
#     model_dict = model.state_dict()
#     pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
#     model_dict.update(pretrained_dict)
#     model.load_state_dict(model_dict)

#     model.eval()

#     # Predict
#     bed_lines = open(args.input_bed).readlines()
#     with_peak, without_peak = predict_regions_variable(
#         model, sequence_dict, bed_lines, add_features_list, device, # args.threshold
#     )

#     # Save
#     write_output(os.path.join(args.outdir, 'with_peak_WT2.bed'), with_peak)
#     write_output(os.path.join(args.outdir, 'without_peak_WT2.bed'), without_peak)

#     print(f"Predicted {len(with_peak)} regions with peaks and {len(without_peak)} without peaks.")


# if __name__ == '__main__':
#     main()

