#!/usr/bin/python

import os
import sys
import argparse
import numpy as np
import os.path as osp

import torch
from torch.utils.data import DataLoader

# custom functions defined by user
from FCNmotif import FCNsignalMod
from datasets import EPIDataSet
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt

torch.multiprocessing.set_sharing_strategy('file_system')
plt.rcParams["font.family"] = "serif"


def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="FCNsignal for testing data")

    parser.add_argument("-d", dest="data_dir", type=str, default=None,
                        help="A directory containing the test data.")
    parser.add_argument("-n", dest="name", type=str, default="EBF",
                        help="The name of a specified data.")

    parser.add_argument("-g", dest="gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("-c", dest="checkpoint", type=str, default='./models/',
                        help="Where to load the model from.")
    parser.add_argument("-o", dest="output_dir", type=str, default=None,
                        help="Where to save test results. If not specified, saves to checkpoint directory.")
    parser.add_argument("-t", dest="threshold", type=float, default=0.5,
                        help="Threshold for binary classification.")
    parser.add_argument("-m", dest="seq_min_run", type=int, default=12,
                        help="Minimal number of consecutive bases for sequence-level classification.")
    parser.add_argument("--use-all-channels", dest="use_all_channels", action='store_true',
                        help="Force using all channels from data (including additional features).")
    parser.add_argument("--dna-only", dest="dna_only", action='store_true',
                        help="Force using only DNA sequence (first 4 channels).")

    return parser.parse_args()


def get_model_input_channels(state_dict):
    """
    Infer the number of input channels expected by the model
    by inspecting the first convolutional layer weights.
    """
    for key in sorted(state_dict.keys()):
        if 'weight' in key and len(state_dict[key].shape) == 3:  # Conv1d weight shape: [out, in, kernel]
            in_channels = state_dict[key].shape[1]
            print(f"Detected model input channels from '{key}': {in_channels}")
            return in_channels
    
    print("WARNING: Could not detect input channels from model. Assuming 4 (DNA only).")
    return 4


def has_run_of_ones(arr, min_run=12):
    """
    Return True if arr (1D array-like of 0/1) contains >= min_run consecutive ones.
    """
    count = 0
    for v in np.asarray(arr).astype(int):
        if v == 1:
            count += 1
            if count >= min_run:
                return True
        else:
            count = 0
    return False


def compute_sequence_metrics(truths, preds, threshold=0.5, min_run=12):

    seq_true = []
    seq_pred = []
    seq_score = []  
    for t, p in zip(truths, preds):
        t_arr = np.asarray(t).flatten()
        p_arr = np.asarray(p).flatten()
        
        t_bin = t_arr.astype(int)
        p_bin = (p_arr >= threshold).astype(int)

        # Binary sequence label
        truth_has = has_run_of_ones(t_bin, min_run=min_run)
        pred_has = has_run_of_ones(p_bin, min_run=min_run)

        seq_true.append(1 if truth_has else 0)
        seq_pred.append(1 if pred_has else 0)

        if len(p_arr) >= min_run:
            score = max(
                np.mean(p_arr[i:i + min_run])
                for i in range(len(p_arr) - min_run + 1)
            )
        else:
            score = np.mean(p_arr)
        seq_score.append(score)

    if len(seq_true) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, seq_true, seq_pred

    seq_acc = accuracy_score(seq_true, seq_pred)
    seq_prec = precision_score(seq_true, seq_pred, zero_division=0)
    seq_rec = recall_score(seq_true, seq_pred, zero_division=0)
    seq_f1 = f1_score(seq_true, seq_pred, zero_division=0)
    try:
        seq_auc = roc_auc_score(seq_true, seq_score)
    except Exception:
        seq_auc = np.nan
    try:
        seq_aupr = average_precision_score(seq_true, seq_score)
    except Exception:
        seq_aupr = np.nan

    return seq_acc, seq_prec, seq_rec, seq_f1, seq_auc, seq_aupr, seq_true, seq_pred



def plot_signals(truths, preds, threshold, output_dir, prefix, num_samples=10):
    """Plot true vs predicted signals for sample sequences"""
    if len(truths) == 0 or len(preds) == 0:
        return
    
    num_samples = min(num_samples, len(truths))
    
    for i in range(num_samples):
        x = np.asarray(truths[i]).flatten()
        y_prob = np.asarray(preds[i]).flatten()
        y_bin = (y_prob >= threshold).astype(int)
        
        plt.figure(figsize=(10, 3))
        plt.plot(x, label="True (0/1)", linewidth=1)
        plt.plot(y_bin, label="Pred Binary", linewidth=1, alpha=0.7)
        plt.xlabel("Position")
        plt.ylabel("0 / 1")
        plt.title(f"{prefix} Sample {i+1}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(osp.join(output_dir, f"{prefix}_sample_{i+1}.pdf"))
        plt.savefig(osp.join(output_dir, f"{prefix}_sample_{i+1}.png"))
        plt.close()


def main():
    """Create the model and start the testing."""
    args = get_args()
    if torch.cuda.is_available():
        if len(args.gpu.split(',')) == 1:
            device = torch.device("cuda:" + args.gpu)
        else:
            device = torch.device("cuda:" + args.gpu.split(',')[0])
    else:
        device = torch.device("cpu")
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir is not None else args.checkpoint
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output file
    f = open(osp.join(output_dir, 'test_results.txt'), 'w')
    
    motifLen = 12
    
    #  LOAD MODEL AND DETECT INPUT CHANNELS 
    checkpoint_file = osp.join(args.checkpoint, 'model_best_ebf.pth')
    print(f"Loading model from: {checkpoint_file}")
    chk = torch.load(checkpoint_file, map_location='cpu')
    state_dict = chk['model_state_dict']
    
    # Detect expected input channels from model architecture
    model_in_channels = get_model_input_channels(state_dict)
    
    #  LOAD TEST DATA 
    print(f"\nLoading test data from: {args.data_dir}")
    Data = np.load(osp.join(args.data_dir, '%s_test.npz' % args.name))
    seqs, signals = Data['data'], Data['signal']
    
    print(f"Loaded data shape: {seqs.shape}")  # [N, channels, seq_len]
    data_channels = seqs.shape[1]
    
    #  HANDLE CHANNEL MISMATCH 
    if args.dna_only:
        print("--dna-only flag set: Using only first 4 channels (DNA sequence)")
        seqs = seqs[:, :4, :]
        use_channels = 4
    elif args.use_all_channels:
        print(f"--use-all-channels flag set: Using all {data_channels} channels")
        use_channels = data_channels
    else:
        # Auto-detect: use what the model expects
        use_channels = model_in_channels
        if data_channels != model_in_channels:
            print(f"\n{'='*60}")
            print(f"WARNING: Channel mismatch detected!")
            print(f"  Data has {data_channels} channels")
            print(f"  Model expects {model_in_channels} channels")
            if data_channels > model_in_channels:
                print(f"  -> Truncating to first {model_in_channels} channels")
                seqs = seqs[:, :model_in_channels, :]
            else:
                print(f"  -> ERROR: Data has fewer channels than model expects!")
                print(f"  -> This will cause errors. Please check your data.")
                f.write("ERROR: Data channel mismatch - test aborted\n")
                f.close()
                return
            print(f"{'='*60}\n")
    
    # Describe what channels are being used
    if use_channels == 4:
        channels_desc = "DNA sequence only (A, C, G, T)"
    elif use_channels == 5:
        channels_desc = "DNA + 1 additional features (ATAC-seq)"
    else:
        channels_desc = f"{use_channels} channels"
    
    print(f"\nUsing {use_channels} input channels: {channels_desc}")
    f.write(f"Input channels: {use_channels} ({channels_desc})\n")
    
    # Load test datasets
    test_data = EPIDataSet(seqs, signals)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
    
    # Load negative data
    Data_neg = np.load(osp.join(args.data_dir, '%s_neg.npz' % args.name))
    seqs_neg, signals_neg = Data_neg['data'], Data_neg['signal']
    
    # Apply same channel selection to negative data
    if args.dna_only or (not args.use_all_channels and seqs_neg.shape[1] > model_in_channels):
        seqs_neg = seqs_neg[:, :use_channels, :]
    
    neg_data = EPIDataSet(seqs_neg, signals_neg)
    neg_loader = DataLoader(neg_data, batch_size=1, shuffle=False, num_workers=0)
    
    #  LOAD MODEL WITH CORRECT INPUT CHANNELS 
    model = FCNsignalMod(motiflen=motifLen, n_features=1)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"Running evaluation with threshold={args.threshold}, min_run={args.seq_min_run}bp\n")
    
    #  POSITIVE TEST DATA 
    print("Evaluating positive test data...")
    p_all = []
    t_all = []
    
    for i_batch, sample_batch in enumerate(test_loader):
        X_data = sample_batch["data"].float().to(device)
        signal = sample_batch["signal"].float()
        with torch.no_grad():
            pred = model(X_data)
        
        # Convert to probabilities (assuming model outputs logits)
        pred_prob = torch.sigmoid(pred)
        
        signal_p = pred_prob.view(-1).data.cpu().numpy()
        p_all.append(signal_p)
        
        signal_t = signal.view(-1).data.cpu().numpy()
        t_all.append(signal_t)
    
    # Per-base classification metrics
    y_true_pos = np.concatenate(t_all).flatten()
    y_pred_prob_pos = np.concatenate(p_all).flatten()
    y_pred_pos = (y_pred_prob_pos >= args.threshold).astype(int)
    y_true_pos_bin = y_true_pos.astype(int)
    
    acc_pos = accuracy_score(y_true_pos_bin, y_pred_pos)
    prec_pos = precision_score(y_true_pos_bin, y_pred_pos, zero_division=0)
    rec_pos = recall_score(y_true_pos_bin, y_pred_pos, zero_division=0)
    f1_pos = f1_score(y_true_pos_bin, y_pred_pos, zero_division=0)
    try:
        auc_pos = roc_auc_score(y_true_pos_bin, y_pred_prob_pos)
    except ValueError:
        auc_pos = np.nan
    try:
        aupr_pos = average_precision_score(y_true_pos_bin, y_pred_prob_pos)
    except ValueError:
        aupr_pos = np.nan
    
    # print(f"Positive Test Per-base - ACC: {acc_pos:.3f}\tPREC: {prec_pos:.3f}\t"
          # f"REC: {rec_pos:.3f}\tF1: {f1_pos:.3f}\tAUC: {auc_pos:.3f}\tAUPR: {aupr_pos:.3f}")
    
    #  NEGATIVE TEST DATA 
    print("\nEvaluating negative test data...")
    p_all_neg = []
    t_all_neg = []
    
    for i_batch, sample_batch in enumerate(neg_loader):
        X_data = sample_batch["data"].float().to(device)
        signal = sample_batch["signal"].float()
        with torch.no_grad():
            pred = model(X_data)
        
        pred_prob = torch.sigmoid(pred)
        
        signal_p = pred_prob.view(-1).data.cpu().numpy()
        p_all_neg.append(signal_p)
        
        signal_t = signal.view(-1).data.cpu().numpy()
        t_all_neg.append(signal_t)
    
    # Per-base classification metrics
    y_true_neg = np.concatenate(t_all_neg).flatten()
    y_pred_prob_neg = np.concatenate(p_all_neg).flatten()
    y_pred_neg = (y_pred_prob_neg >= args.threshold).astype(int)
    y_true_neg_bin = y_true_neg.astype(int)
    
    acc_neg = accuracy_score(y_true_neg_bin, y_pred_neg)
    prec_neg = precision_score(y_true_neg_bin, y_pred_neg, zero_division=0)
    rec_neg = recall_score(y_true_neg_bin, y_pred_neg, zero_division=0)
    f1_neg = f1_score(y_true_neg_bin, y_pred_neg, zero_division=0)
    try:
        auc_neg = roc_auc_score(y_true_neg_bin, y_pred_prob_neg)
    except ValueError:
        auc_neg = np.nan
    try:
        aupr_neg = average_precision_score(y_true_neg_bin, y_pred_prob_neg)
    except ValueError:
        aupr_neg = np.nan
    
    # print(f"Negative Test Per-base - ACC: {acc_neg:.3f}\tPREC: {prec_neg:.3f}\t"
          # f"REC: {rec_neg:.3f}\tF1: {f1_neg:.3f}\tAUC: {auc_neg:.3f}\tAUPR: {aupr_neg:.3f}")
    
    #  COMBINED METRICS (pos + neg) 
    print("\nCombined per-base metrics...")
    
    # Combined per-base metrics
    y_true_combined = np.concatenate([y_true_pos_bin, y_true_neg_bin])
    y_pred_prob_combined = np.concatenate([y_pred_prob_pos, y_pred_prob_neg])
    y_pred_combined = (y_pred_prob_combined >= args.threshold).astype(int)
    
    acc_comb = accuracy_score(y_true_combined, y_pred_combined)
    prec_comb = precision_score(y_true_combined, y_pred_combined, zero_division=0)
    rec_comb = recall_score(y_true_combined, y_pred_combined, zero_division=0)
    f1_comb = f1_score(y_true_combined, y_pred_combined, zero_division=0)
    try:
        auc_comb = roc_auc_score(y_true_combined, y_pred_prob_combined)
    except ValueError:
        auc_comb = np.nan
    try:
        aupr_comb = average_precision_score(y_true_combined, y_pred_prob_combined)
    except ValueError:
        aupr_comb = np.nan
    
    print(f"Combined Per-base - ACC: {acc_comb:.3f}\tPREC: {prec_comb:.3f}\t"
          f"REC: {rec_comb:.3f}\tF1: {f1_comb:.3f}\tAUC: {auc_comb:.3f}\tAUPR: {aupr_comb:.3f}")
    
    #  SEQUENCE-LEVEL METRICS 
    print(f"\nComputing sequence-level metrics (>= {args.seq_min_run} bp run)...")
    
    # Positive sequences
    # seq_acc_pos, seq_prec_pos, seq_rec_pos, seq_f1_pos, seq_auc_pos, seq_aupr_pos, seq_true_pos, seq_pred_pos = \
        # compute_sequence_metrics(t_all, p_all, args.threshold, args.seq_min_run)

    
    # Negative sequences
    # seq_acc_neg, seq_prec_neg, seq_rec_neg, seq_f1_neg, seq_auc_neg, seq_aupr_neg, seq_true_neg, seq_pred_neg = \
        # compute_sequence_metrics(t_all_neg, p_all_neg, args.threshold, args.seq_min_run)
    
    # Combined sequences
    all_truths = t_all + t_all_neg
    all_preds = p_all + p_all_neg
    seq_acc, seq_prec, seq_rec, seq_f1, seq_auc, seq_aupr, seq_true, seq_pred = \
        compute_sequence_metrics(all_truths, all_preds, args.threshold, args.seq_min_run)
    
    print(f"Combined Sequence-level - ACC: {seq_acc:.3f}\tPREC: {seq_prec:.3f}\t"
          f"REC: {seq_rec:.3f}\tF1: {seq_f1:.3f}\tAUC: {seq_auc:.3f}\tAUPR: {seq_aupr:.3f}")
    
    #  PLOTTING 
    print("\nGenerating plots...")
    
    # Confusion matrices - Per-base

    cm_comb = confusion_matrix(y_true_combined, y_pred_combined)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_comb)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix - Combined Test (Per-base)")
    plt.tight_layout()
    plt.savefig(osp.join(output_dir, "confusion_matrix_combined_test.png"))
    plt.savefig(osp.join(output_dir, "confusion_matrix_combined_test.pdf"))
    plt.close()
    
    # Confusion matrix with percentages
    cm_comb_pct = cm_comb.astype('float') / cm_comb.sum(axis=1)[:, np.newaxis] * 100
    disp_pct = ConfusionMatrixDisplay(confusion_matrix=cm_comb_pct, display_labels=['Negative', 'Positive'])
    disp_pct.plot(cmap="Blues", values_format=".2f")
    plt.title("Confusion Matrix - Combined Test (Per-base, %)")
    plt.tight_layout()
    plt.savefig(osp.join(output_dir, "confusion_matrix_combined_test_percent.png"))
    plt.savefig(osp.join(output_dir, "confusion_matrix_combined_test_percent.pdf"))
    plt.close()
    
    # Sequence-level confusion matrices    
    cm_seq = confusion_matrix(seq_true, seq_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_seq)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - Combined Sequence-level (>= {args.seq_min_run} bp)")
    plt.tight_layout()
    plt.savefig(osp.join(output_dir, "confusion_matrix_combined_sequence_test.png"))
    plt.savefig(osp.join(output_dir, "confusion_matrix_combined_sequence_test.pdf"))
    plt.close()
    
    # Sequence-level confusion matrix with percentages
    cm_seq_pct = cm_seq.astype('float') / cm_seq.sum(axis=1)[:, np.newaxis] * 100
    disp_seq_pct = ConfusionMatrixDisplay(confusion_matrix=cm_seq_pct, display_labels=['Negative', 'Positive'])
    disp_seq_pct.plot(cmap="Blues", values_format=".2f")
    plt.title(f"Confusion Matrix - Combined Sequence-level (>= {args.seq_min_run} bp, %)")
    plt.tight_layout()
    plt.savefig(osp.join(output_dir, "confusion_matrix_combined_sequence_test_percent.png"))
    plt.savefig(osp.join(output_dir, "confusion_matrix_combined_sequence_test_percent.pdf"))
    plt.close()
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true_combined, y_pred_prob_combined)
    avg_prec = average_precision_score(y_true_combined, y_pred_prob_combined)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (Test, AP = {avg_prec:.3f})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(osp.join(output_dir, "precision_recall_curve_test.png"))
    plt.savefig(osp.join(output_dir, "precision_recall_curve_test.pdf"))
    plt.close()

    fpr, tpr, _ = roc_curve(y_true_combined, y_pred_prob_combined)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {auc_comb:.3f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(osp.join(output_dir, "roc_curve_test.png"))
    plt.savefig(osp.join(output_dir, "roc_curve_test.pdf"))
    plt.close()
    
    # Plot sample predictions
    plot_signals(t_all[:10], p_all[:10], args.threshold, output_dir, "pos_test", num_samples=10)
    plot_signals(t_all_neg[:10], p_all_neg[:10], args.threshold, output_dir, "neg_test", num_samples=10)
    
    #  SAVE RESULTS TO FILE 
    f.write("=" * 60 + "\n")
    f.write("BASE-RESOLUTION TFBS PREDICTION TEST RESULTS\n")
    f.write("=" * 60 + "\n\n")
    
    f.write(f"Model checkpoint: {checkpoint_file}\n")
    f.write(f"Data directory: {args.data_dir}\n")
    f.write(f"Input channels: {use_channels} ({channels_desc})\n")
    f.write(f"Classification threshold: {args.threshold}\n")
    f.write(f"Sequence min run: {args.seq_min_run} bp\n\n")
    
    f.write("=== PER-BASE CLASSIFICATION METRICS ===\n\n")
    
    f.write("Combined Test:\n")
    f.write(f"  Accuracy:  {acc_comb:.4f}\n")
    f.write(f"  Precision: {prec_comb:.4f}\n")
    f.write(f"  Recall:    {rec_comb:.4f}\n")
    f.write(f"  F1-score:  {f1_comb:.4f}\n")
    f.write(f"  AUC:       {auc_comb:.4f}\n")
    f.write(f"  AUPR:      {aupr_comb:.4f}\n")
    f.write(f"  Avg Precision: {avg_prec:.4f}\n\n")
    
    f.write(f"=== SEQUENCE-LEVEL METRICS (>= {args.seq_min_run} bp rule) ===\n\n")
    
    f.write("Combined Test:\n")
    f.write(f"  Accuracy:  {seq_acc:.4f}\n")
    f.write(f"  Precision: {seq_prec:.4f}\n")
    f.write(f"  Recall:    {seq_rec:.4f}\n")
    f.write(f"  F1-score:  {seq_f1:.4f}\n")
    f.write(f"  AUC:       {seq_auc:.4f}\n")
    f.write(f"  AUPR:      {seq_aupr:.4f}\n")
    
    f.close()
    
    print(f"\nResults saved to {osp.join(output_dir, 'test_results.txt')}")
    print("All plots generated successfully!")
    
    # Save predictions for further analysis
    np.savez_compressed(
        osp.join(output_dir, "test_predictions.npz"),
        pos_preds=np.concatenate(p_all),
        pos_truths=np.concatenate(t_all),
        neg_preds=np.concatenate(p_all_neg),
        neg_truths=np.concatenate(t_all_neg),
        threshold=args.threshold,
        seq_min_run=args.seq_min_run,
        input_channels=use_channels,
        channels_description=channels_desc
    )
    print(f"Predictions saved to {osp.join(args.checkpoint, 'test_predictions.npz')}")


if __name__ == "__main__":
    main()

