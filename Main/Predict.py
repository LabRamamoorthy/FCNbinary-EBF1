import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

#  Project imports 
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

import Model.BertSNR2 as BERTSNR2
import Utils.Metrics as Metrics
import pandas as pd

# CONFIG  

TF_NAME       = "EBF_2_250"
DATASET_ROOT  = "Dataset/ChIP-seq"
DNABERT2_PATH = "/home/ubuntu/data/dnabert2"
WEIGHTS_PATH  = f"Main/ModelWeight/multiModel/{TF_NAME}/pytorch_model.bin"
BATCH_SIZE    = 16
THRESHOLD     = 0.5
MAX_LENGTH    = 512

#  Dataset 
class DNABERT2Dataset(Dataset):
    """Reads the tab-separated test.txt produced by the dataset-creation script."""

    def __init__(self, filepath, tokenizer, max_length=MAX_LENGTH):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.sequences, self.dense_labels, self.labels = [], [], []

        with open(filepath) as f:
            next(f)                                     # skip header
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 3:
                    continue
                seq, dense, lab = parts
                self.sequences.append(seq)
                # dense label is a string of '0'/'1' chars, one per nucleotide
                self.dense_labels.append([int(c) for c in dense if c in "01"])
                self.labels.append(int(lab))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.dense_labels[idx], self.labels[idx]


def collate_fn(batch, tokenizer):
    seqs, dense_labels, labels = zip(*batch)

    encoded = tokenizer(
        list(seqs),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_offsets_mapping=True,
    )

    return (
        encoded["input_ids"],
        encoded["attention_mask"],
        encoded["offset_mapping"],
        dense_labels,          
        torch.tensor(labels, dtype=torch.float32),
    )


#  Main 
def main():
    test_file = os.path.join(DATASET_ROOT, TF_NAME, "dnabert2", "test.txt")
    if not os.path.isfile(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(DNABERT2_PATH, trust_remote_code=True)

    # dataset & loader
    dataset = DNABERT2Dataset(test_file, tokenizer)
    loader  = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )
    print(f"Test samples: {len(dataset)}")

    # model
    model = BertSNR2.BERTSNR2().to(device)
    state = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    #  accumulators 
    seq_pred_all,    seq_label_all,    seq_logit_all    = [], [], []
    token_pred_all,  token_label_all,  token_logit_all  = [], [], []

    with torch.no_grad():
        for input_ids, attention_mask, offset_mapping, dense_labels, seq_labels in tqdm(loader):
            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            offset_mapping = offset_mapping.to(device)

            logits_seq, all_nuc_logits = model(input_ids, attention_mask, offset_mapping)

            #  sequence-level 
            logits_seq_np = logits_seq.squeeze(1).cpu().numpy()   # (B,)
            preds_seq_np  = (logits_seq_np >= THRESHOLD).astype(int)
            seq_logit_all.extend(logits_seq_np.tolist())
            seq_pred_all.extend(preds_seq_np.tolist())
            seq_label_all.extend(seq_labels.numpy().astype(int).tolist())

            #  token / nucleotide-level 
            for b, nuc_logits in enumerate(all_nuc_logits):
                true_dense = dense_labels[b]           

                if nuc_logits.numel() == 0:
                    continue

                nuc_logits_np = nuc_logits.squeeze(1).cpu().numpy()  # (L,)
                nuc_preds_np  = (nuc_logits_np >= THRESHOLD).astype(int)

                # align lengths 
                L = min(len(true_dense), len(nuc_logits_np))
                token_logit_all.extend(nuc_logits_np[:L].tolist())
                token_pred_all.extend(nuc_preds_np[:L].tolist())
                token_label_all.extend(true_dense[:L])

    #  report 
    seq_true  = np.array(seq_label_all,   dtype=np.int32)
    seq_pred  = np.array(seq_pred_all,    dtype=np.int32)
    seq_logit = np.array(seq_logit_all,   dtype=np.float32)

    tok_true  = np.array(token_label_all, dtype=np.int32)
    tok_pred  = np.array(token_pred_all,  dtype=np.int32)
    tok_logit = np.array(token_logit_all, dtype=np.float32)

    seq_m = Metrics.EvaluationMetricsSequence(y_pred=seq_pred, y_true=seq_true, y_logits=seq_logit)
    tok_m = Metrics.EvaluationMetricsToken2(y_pred=tok_pred, y_true=tok_true, y_logits=tok_logit)

    print("\n=== Sequence-level ===")
    print("Acc=%.3f  Pre=%.3f  Rec=%.3f  F1=%.3f  AUC=%.3f  AUPR=%.3f" % tuple(seq_m))

    print("\n=== Nucleotide-level ===")
    print("Acc=%.3f  Pre=%.3f  Rec=%.3f  F1=%.3f  AUC=%.3f  AUPR=%.3f" % tuple(tok_m))

    print("\nTotal nucleotides evaluated:", len(tok_true))
    print("Positive nucleotides in ground truth:", tok_true.sum())
    #  save results
    os.makedirs("Result", exist_ok=True)
 
    header   = "TF\tAcc\tPre\tRec\tF1\tAUC\tAUPR\n"
    seq_line = "%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n" % (
        TF_NAME, seq_m[0], seq_m[1], seq_m[2], seq_m[3], seq_m[4], seq_m[5])
    tok_line = "%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n" % (
        TF_NAME, tok_m[0], tok_m[1], tok_m[2], tok_m[3], tok_m[4], tok_m[5])
 
    for out_path, line in [("Result/test_data_seq.txt",   seq_line),
                            ("Result/test_data_token.txt", tok_line)]:
        write_header = not os.path.isfile(out_path)
        with open(out_path, "a") as f:
            if write_header:
                f.write(header)
            f.write(line)
        print(f"Results appended to {out_path}")

if __name__ == "__main__":
    main()