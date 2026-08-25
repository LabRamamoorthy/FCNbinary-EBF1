#!/usr/bin/env python
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
)
from transformers import AutoTokenizer

#  path setup 
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from fused_model import FusedModel         
from fused_model import FusedDNADataset 


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_logit: np.ndarray, label: str = "") -> dict:
    unique = np.unique(y_true)
    if len(unique) < 2:
        print(f"  [warn] {label}: only class {unique} in ground truth — "
              "AUC / AUPR undefined, setting to NaN.")
        auc  = float("nan")
        aupr = float("nan")
    else:
        auc  = roc_auc_score(y_true, y_logit)
        aupr = average_precision_score(y_true, y_logit)

    return {
        "acc" : accuracy_score(y_true, y_pred),
        "pre" : precision_score(y_true, y_pred, zero_division=0),
        "rec" : recall_score(y_true, y_pred, zero_division=0),
        "f1"  : f1_score(y_true, y_pred, zero_division=0),
        "auc" : auc,
        "aupr": aupr,
    }


def avg_metrics(m1: dict, m2: dict) -> dict:
    out = {}
    for k in m1:
        v1, v2 = m1[k], m2[k]
        if np.isnan(v1) and np.isnan(v2):
            out[k] = float("nan")
        elif np.isnan(v1):
            out[k] = v2
        elif np.isnan(v2):
            out[k] = v1
        else:
            out[k] = (v1 + v2) / 2.0
    return out


def print_metrics(name: str, m: dict):
    print(
        f"  {name:<30s}  "
        f"Acc={m['acc']:.4f}  Pre={m['pre']:.4f}  Rec={m['rec']:.4f}  "
        f"F1={m['f1']:.4f}  AUC={m['auc']:.4f}  AUPR={m['aupr']:.4f}"
    )


def collate_fn(batch):
    keys = batch[0].keys()
    out  = {}
    for k in keys:
        tensors = [b[k] for b in batch]
        out[k]  = torch.stack(tensors, dim=0)
    return out

def threshold(logits: torch.Tensor, t: float = 0.5) -> np.ndarray:
    return (torch.sigmoid(logits).cpu().numpy() >= t).astype(np.int32)

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    #  Tokenizer & dataset 
    tokenizer = AutoTokenizer.from_pretrained(
        args.bert_path, trust_remote_code=True
    )
    dataset = FusedDNADataset(
        bertsnr_txt = args.bertsnr_txt,
        fcn_npz     = args.fcn_npz,
        tokenizer   = tokenizer,
        max_len     = args.max_len,
    )
    loader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        collate_fn  = collate_fn,
    )
    print(f"Dataset size : {len(dataset)} samples")
    print(f"n_features   : {dataset.n_features}")

    #  Model 
    model = FusedModel(
        motiflen   = args.motiflen,
        n_features = dataset.n_features,
        dropout    = args.dropout,
    ).to(device)
    state = torch.load(args.model_weights, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"Loaded weights from: {args.model_weights}\n")

    #  Accumulators 
    # Token-level — GT1 (bert_labels)
    tok_logit_gt1  = []   # sigmoid scores  
    tok_pred_gt1   = []   # binary preds    
    tok_true_gt1   = []   # ground truth   

    # Token-level — GT2 (fcn_signal)
    tok_logit_gt2  = []
    tok_pred_gt2   = []
    tok_true_gt2   = []

    # Sequence-level
    seq_logit      = []
    seq_pred       = []
    seq_true       = []

    threshold_val  = args.threshold

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):

            #  Inputs 
            fcn_input      = batch['one_hot'].to(device)       # [B, 4(+F), L]
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            offset_mapping = batch['offset_mapping'].to(device)

            #  Ground truths 
            bert_lbl   = batch['bert_labels']    # [B, max_len]  GT1 (CPU)
            fcn_signal = batch['fcn_signal']      # [B, L]        GT2 (CPU)
            seq_lbl    = batch['seq_label']       # [B]           seq GT (CPU)

            #  Forward 
            logits_seq, fused_logits = model(
                fcn_input, input_ids, attention_mask, offset_mapping
            )
            # logits_seq  : [B, 1]
            # fused_logits: list of B tensors, each [L_i, 1]

            #  Sequence-level 
            sig_seq = torch.sigmoid(logits_seq).squeeze(-1).cpu()  # [B]
            prd_seq = (sig_seq.numpy() >= threshold_val).astype(np.int32)

            seq_logit.append(sig_seq.numpy())
            seq_pred.append(prd_seq)
            seq_true.append(seq_lbl.numpy().astype(np.int32))

            #  Token-level 
            for b_idx, nuc_logits in enumerate(fused_logits):
                # nuc_logits : [L_i, 1]
                L_pred = nuc_logits.size(0)

                sig_nuc  = torch.sigmoid(nuc_logits).squeeze(-1).cpu().numpy()  # [L_i]
                pred_nuc = (sig_nuc >= threshold_val).astype(np.int32)

                # GT1: bert_labels — take the first L_pred positions
                gt1 = bert_lbl[b_idx, :L_pred].numpy().astype(np.int32)

                # GT2: fcn_signal — take the first L_pred positions
                #      fcn_signal shape is [B, L_fcn]; L_fcn may differ from L_pred
                L_fcn = fcn_signal.shape[1]
                L     = min(L_pred, L_fcn)

                sig_nuc_trimmed  = sig_nuc[:L]
                pred_nuc_trimmed = pred_nuc[:L]
                gt1_trimmed      = gt1[:L]
                gt2_trimmed      = fcn_signal[b_idx, :L].numpy().astype(np.int32)

                tok_logit_gt1.append(sig_nuc_trimmed)
                tok_pred_gt1.append(pred_nuc_trimmed)
                tok_true_gt1.append(gt1_trimmed)

                tok_logit_gt2.append(sig_nuc_trimmed)
                tok_pred_gt2.append(pred_nuc_trimmed)
                tok_true_gt2.append(gt2_trimmed)

    #  Concatenate 
    tok_logit_gt1 = np.concatenate(tok_logit_gt1)
    tok_pred_gt1  = np.concatenate(tok_pred_gt1)
    tok_true_gt1  = np.concatenate(tok_true_gt1)

    tok_logit_gt2 = np.concatenate(tok_logit_gt2)
    tok_pred_gt2  = np.concatenate(tok_pred_gt2)
    tok_true_gt2  = np.concatenate(tok_true_gt2)

    seq_logit = np.concatenate(seq_logit)
    seq_pred  = np.concatenate(seq_pred)
    seq_true  = np.concatenate(seq_true)

    #  Compute metrics 
    print("\n" + "=" * 72)
    print("TOKEN-LEVEL RESULTS")
    print("=" * 72)

    m_gt1 = compute_metrics(tok_true_gt1, tok_pred_gt1, tok_logit_gt1,
                             label="GT1 (bert_labels)")
    m_gt2 = compute_metrics(tok_true_gt2, tok_pred_gt2, tok_logit_gt2,
                             label="GT2 (fcn_signal)")
    m_avg = avg_metrics(m_gt1, m_gt2)

    print_metrics("vs GT1 (BERTsnr labels)", m_gt1)
    print_metrics("vs GT2 (FCN signal)",     m_gt2)
    print("-" * 72)
    print_metrics("FUSED (average of GT1+GT2)", m_avg)

    print("\n" + "=" * 72)
    print("SEQUENCE-LEVEL RESULTS")
    print("=" * 72)
    m_seq = compute_metrics(seq_true, seq_pred, seq_logit, label="seq")
    print_metrics("Sequence classifier (CLS)", m_seq)
    print("=" * 72 + "\n")

    #  Optional CSV save 
    if args.output_csv:
        rows = []
        for name, m in [
            ("token_vs_gt1_bert",    m_gt1),
            ("token_vs_gt2_fcn",     m_gt2),
            ("token_fused_avg",      m_avg),
            ("sequence_level",       m_seq),
        ]:
            rows.append({"level": name, **m})
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(args.output_csv, index=False)
        print(f"Results saved to: {args.output_csv}")


class Config:
    def __init__(self, args):
        # Paths
        self.bertsnr_txt   = args.bertsnr_txt
        self.fcn_npz       = args.fcn_npz
        self.model_weights = args.model_weights
        self.bert_path     = args.bert_path

        # Model
        self.motiflen = args.motiflen
        self.dropout  = args.dropout

        # Evaluation
        self.max_len     = args.max_len
        self.batch_size  = args.batch_size
        self.num_workers = args.num_workers
        self.threshold   = args.threshold

        # Output
        self.output_csv = args.output_csv


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate the fused DNA model."
    )

    # Paths
    parser.add_argument(
        "--bertsnr_txt",
        type=str,
        required=True,
        help="Path to BERTsnr test dataset."
    )
    parser.add_argument(
        "--fcn_npz",
        type=str,
        required=True,
        help="Path to FCN test NPZ file."
    )
    parser.add_argument(
        "--model_weights",
        type=str,
        required=True,
        help="Path to model weights."
    )
    parser.add_argument(
        "--bert_path",
        type=str,
        required=True,
        help="Path to DNABERT-2 model/tokenizer."
    )

    # Model
    parser.add_argument(
        "--motiflen",
        type=int,
        default=10,
        help="Motif length."
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate."
    )

    # Evaluation
    parser.add_argument(
        "--max_len",
        type=int,
        default=512,
        help="Maximum sequence length."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Evaluation batch size."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of DataLoader workers."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold."
    )

    # Output
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to output CSV file."
    )

    args = parser.parse_args()

    evaluate(Config(args))