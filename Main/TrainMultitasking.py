from tqdm import tqdm
from transformers import AutoTokenizer

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import transformers.optimization as op
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score,
    roc_auc_score
)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (parent of Main/)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

sys.path.append(PROJECT_ROOT)
import Dataset.DataLoader
import Dataset.DataReader
import Model.BERTSNR2

import Utils.Metrics
import Utils.Threshold



# Helpers


def set_seed(args):
    np.random.seed(args)
    torch.manual_seed(args)


def tokenize_sequences(sequences, tokenizer):
    """Tokenize a list of raw DNA sequences once before training."""
    encoded = tokenizer(
        list(sequences),
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512,
        return_offsets_mapping=True
    )
    return encoded


def plot_and_save(
    TFsName,
    result_dir,
    train_losses,
    val_losses,
    # per-base arrays (nucleotide-resolved via offset_mapping inside the model)
    logits_token,
    pred_token,
    label_token,
    # per-sequence arrays (from the model's sequence classification head directly)
    logits_sequence,
    pred_sequence,
    label_sequence,
    threshold=0.5,
):
    """
    Save into result_dir / TFsName /:
      - loss_curve.png
      - per-base confusion matrices (count + %)
      - per-base precision-recall curve
      - per-sequence confusion matrices (count + %)
      - per-sequence precision-recall curve
      - metrics_summary.txt
      - val_outputs.npz
    """
    out_dir = os.path.join(result_dir, TFsName)
    os.makedirs(out_dir, exist_ok=True)

    
    # 1. Loss curve
    
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses,   label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{TFsName} — Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"))
    plt.close()

    
    # 2. Per-base (token-level) plots
    # DNABERT-2 uses BPE; nucleotide alignment is done inside the model
    # via offset_mapping, so logits_token / label_token are already flat
    # per-nucleotide arrays — no k-mer unwinding needed.
    
    y_true_base = label_token.astype(int)
    y_prob_base = logits_token                        # sigmoid probabilities
    y_pred_base = (y_prob_base >= threshold).astype(int)

    # Count-based confusion matrix
    cm_base = confusion_matrix(y_true_base, y_pred_base)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_base)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"{TFsName} — Per-base Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix_per_base.png"))
    plt.savefig(os.path.join(out_dir, "confusion_matrix_per_base.pdf"))
    plt.close()

    # Percentage confusion matrix
    cm_base_pct = cm_base.astype(float) / cm_base.sum(axis=1, keepdims=True) * 100
    disp_pct = ConfusionMatrixDisplay(confusion_matrix=cm_base_pct)
    disp_pct.plot(cmap="Blues", values_format=".2f")
    plt.title(f"{TFsName} — Per-base Confusion Matrix (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix_per_base_pct.png"))
    plt.savefig(os.path.join(out_dir, "confusion_matrix_per_base_pct.pdf"))
    plt.close()

    # Per-base Precision-Recall curve
    precision_b, recall_b, _ = precision_recall_curve(y_true_base, y_prob_base)
    avg_prec_base = average_precision_score(y_true_base, y_prob_base)
    plt.figure(figsize=(6, 5))
    plt.plot(recall_b, precision_b, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{TFsName} — PR Curve (Per-base, AP={avg_prec_base:.3f})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "precision_recall_curve_per_base.png"))
    plt.savefig(os.path.join(out_dir, "precision_recall_curve_per_base.pdf"))
    plt.close()

    
    # 3. Per-sequence plots
    # The model has a dedicated sequence classification head — use its
    # outputs directly, no heuristic needed.
    
    y_true_seq = label_sequence.astype(int)
    y_prob_seq = logits_sequence                      # sigmoid probabilities
    y_pred_seq = pred_sequence.astype(int)

    # Count-based confusion matrix
    cm_seq = confusion_matrix(y_true_seq, y_pred_seq)
    disp_seq = ConfusionMatrixDisplay(confusion_matrix=cm_seq)
    disp_seq.plot(cmap="Blues", values_format="d")
    plt.title(f"{TFsName} — Per-sequence Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix_per_sequence.png"))
    plt.savefig(os.path.join(out_dir, "confusion_matrix_per_sequence.pdf"))
    plt.close()

    # Percentage confusion matrix
    cm_seq_pct = cm_seq.astype(float) / cm_seq.sum(axis=1, keepdims=True) * 100
    disp_seq_pct = ConfusionMatrixDisplay(confusion_matrix=cm_seq_pct)
    disp_seq_pct.plot(cmap="Blues", values_format=".2f")
    plt.title(f"{TFsName} — Per-sequence Confusion Matrix (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix_per_sequence_pct.png"))
    plt.savefig(os.path.join(out_dir, "confusion_matrix_per_sequence_pct.pdf"))
    plt.close()

    # Per-sequence Precision-Recall curve
    precision_s, recall_s, _ = precision_recall_curve(y_true_seq, y_prob_seq)
    avg_prec_seq = average_precision_score(y_true_seq, y_prob_seq)
    plt.figure(figsize=(6, 5))
    plt.plot(recall_s, precision_s, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{TFsName} — PR Curve (Per-sequence, AP={avg_prec_seq:.3f})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "precision_recall_curve_per_sequence.png"))
    plt.savefig(os.path.join(out_dir, "precision_recall_curve_per_sequence.pdf"))
    plt.close()

    
    # 4. metrics_summary.txt
    
    try:
        base_auc = roc_auc_score(y_true_base, y_prob_base)
    except Exception:
        base_auc = float('nan')

    try:
        seq_auc = roc_auc_score(y_true_seq, y_prob_seq)
    except Exception:
        seq_auc = float('nan')

    base_acc  = accuracy_score(y_true_base, y_pred_base)
    base_prec = precision_score(y_true_base, y_pred_base, zero_division=0)
    base_rec  = recall_score(y_true_base, y_pred_base, zero_division=0)
    base_f1   = f1_score(y_true_base, y_pred_base, zero_division=0)

    seq_acc  = accuracy_score(y_true_seq, y_pred_seq)
    seq_prec = precision_score(y_true_seq, y_pred_seq, zero_division=0)
    seq_rec  = recall_score(y_true_seq, y_pred_seq, zero_division=0)
    seq_f1   = f1_score(y_true_seq, y_pred_seq, zero_division=0)

    metrics_path = os.path.join(out_dir, "metrics_summary.txt")
    with open(metrics_path, "w") as f:
        f.write("=== PER-BASE METRICS (BEST EPOCH) ===\n")
        f.write("Note: DNABERT-2 uses BPE; nucleotide alignment done via offset_mapping.\n")
        f.write(f"Accuracy  : {base_acc:.4f}\n")
        f.write(f"Precision : {base_prec:.4f}\n")
        f.write(f"Recall    : {base_rec:.4f}\n")
        f.write(f"F1-score  : {base_f1:.4f}\n")
        f.write(f"AUC       : {base_auc:.4f}\n")
        f.write(f"Avg Prec  : {avg_prec_base:.4f}\n\n")

        f.write("=== PER-SEQUENCE METRICS (BEST EPOCH) ===\n")
        f.write("Note: Labels from model's sequence classification head directly.\n")
        f.write(f"Accuracy  : {seq_acc:.4f}\n")
        f.write(f"Precision : {seq_prec:.4f}\n")
        f.write(f"Recall    : {seq_rec:.4f}\n")
        f.write(f"F1-score  : {seq_f1:.4f}\n")
        f.write(f"AUC       : {seq_auc:.4f}\n")
        f.write(f"Avg Prec  : {avg_prec_seq:.4f}\n")

    print(f"  → Metrics saved to {metrics_path}")

    
    # 5. val_outputs.npz
    
    npz_path = os.path.join(out_dir, "val_outputs.npz")
    np.savez_compressed(
        npz_path,
        # per-base token level
        token_logits=logits_token,
        token_pred_binary=y_pred_base,
        token_label=label_token,
        # per-sequence level (model head outputs)
        seq_logits=logits_sequence,
        seq_pred_binary=y_pred_seq,
        seq_label=label_sequence,
        # losses
        train_losses=np.array(train_losses),
        val_losses=np.array(val_losses),
    )
    print(f"  → Predictions saved to {npz_path}")



# Initialization


DataSetName    = 'ChIP-seq'
set_seed(1)
use_gpu = torch.cuda.is_available()
print(use_gpu)

MaxEpoch       = 10
BatchSize      = 32
t              = 0.15
ThresholdValue = 0.5


# Paths


RESULT_DIR = os.path.join(PROJECT_ROOT, 'Result')
os.makedirs(RESULT_DIR, exist_ok=True)

result_sequence = open(os.path.join(RESULT_DIR, 'Result_TFBert2_sequence.txt'), 'a')
result_sequence.write('TF_name\tAcc\tPre\tRec\tF1-S\tAUC\tAUPR\n')
result_token = open(os.path.join(RESULT_DIR, 'Result_TFBert2_token.txt'), 'a')
result_token.write('TF_name\tAcc\tPre\tRec\tF1-S\tAUC\tAUPR\n')

DATASET_PATH = os.path.join(PROJECT_ROOT, 'Dataset', 'ChIP-seq')

if not os.path.exists(DATASET_PATH):
    print(f"ERROR: Dataset path does not exist: {DATASET_PATH}")
    print(f"Script directory: {SCRIPT_DIR}")
    print(f"Project root: {PROJECT_ROOT}")
    sys.exit(1)

# Load tokenizer once — shared across all TFs
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)


# Per-TF training loop
for TF in os.listdir(DATASET_PATH):
    TFsName = TF
    print(f"\n{'='*60}\n  TF: {TFsName}\n{'='*60}")

    #  best-epoch accumulators 
    acc_token = pre_token = rec_token = f1_token = auc_token = aupr_token = 0
    acc_sequence = pre_sequence = rec_sequence = f1_sequence = auc_sequence = aupr_sequence = 0

    #  loss tracking 
    train_losses_epochs = []
    val_losses_epochs   = []

    #  best-epoch predictions (cached when AUPR improves) 
    best_logits_token    = None
    best_pred_token      = None
    best_label_token     = None
    best_logits_sequence = None
    best_pred_sequence   = None
    best_label_sequence  = None

    os.makedirs('ModelWeight/multiModel/' + TFsName, exist_ok=True)

    #  data 
    TrainSequence, TrainDenseLabel, TrainLabel, TestSequence, TestDenseLabel, TestLabel = \
        Dataset.DataReader.DataReaderBERT2(TFsName, DataSetName)
    TrainSequence, TrainDenseLabel, TrainLabel, TestSequence, TestDenseLabel, TestLabel = \
        np.array(TrainSequence), np.array(TrainDenseLabel), np.array(TrainLabel), \
        np.array(TestSequence),  np.array(TestDenseLabel),  np.array(TestLabel)

    #  tokenize once 
    TrainEncoded = tokenize_sequences(TrainSequence, tokenizer)
    TestEncoded  = tokenize_sequences(TestSequence,  tokenizer)

    TrainOffsets = TrainEncoded.pop('offset_mapping')
    TestOffsets  = TestEncoded.pop('offset_mapping')

    LossFunction  = nn.BCELoss().cuda()
    NeuralNetwork = Model.BertSNR.BERTSNR().cuda()
    t_total       = len(TrainSequence) * MaxEpoch // BatchSize

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in NeuralNetwork.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in NeuralNetwork.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8,
                            betas=(0.9, 0.999))
    scheduler = op.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(t_total * 0.1),
        num_training_steps=t_total
    )

    #  data loaders 
    TrainLoader = Dataset.DataLoader.SampleLoaderBERT2(
        input_ids=TrainEncoded['input_ids'],
        attention_mask=TrainEncoded['attention_mask'],
        offset_mapping=TrainOffsets,
        DenseLabel=torch.tensor(TrainDenseLabel),
        Label=torch.tensor(TrainLabel),
        BatchSize=BatchSize
    )

    TestLoader = Dataset.DataLoader.SampleLoaderBERT2(
        input_ids=TestEncoded['input_ids'],
        attention_mask=TestEncoded['attention_mask'],
        offset_mapping=TestOffsets,
        DenseLabel=torch.tensor(TestDenseLabel),
        Label=torch.tensor(TestLabel),
        BatchSize=BatchSize
    )

    #  training 
    earlystop = 0

    for Epoch in range(MaxEpoch):

        #  Train 
        NeuralNetwork.train()
        TrainProgressBar = tqdm(TrainLoader, desc=f'Epoch {Epoch} [train]')
        epoch_train_loss = 0.0

        for data in TrainProgressBar:
            optimizer.zero_grad()
            input_ids, attention_mask, offsets, Y1, Y2 = data
            input_ids      = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            Y1 = Y1.cuda()
            Y2 = Y2.cuda()

            Prediction_sequence, Prediction_token = NeuralNetwork(input_ids, attention_mask, offsets)

            loss_list = []
            for i, nuc_logits in enumerate(Prediction_token):
                true_len = min(len(nuc_logits), Y1.shape[1])
                loss_list.append(LossFunction(
                    nuc_logits[:true_len].squeeze(),
                    Y1[i, :true_len].to(torch.float32)
                ))
            Loss_token    = torch.stack(loss_list).mean()
            Loss_sequence = LossFunction(Prediction_sequence.squeeze(-1), Y2.to(torch.float32))
            Loss          = t * Loss_sequence + (1 - t) * Loss_token

            TrainProgressBar.set_description(f'Epoch {Epoch} loss:{Loss.cpu().item():.4f}')
            Loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_train_loss += Loss.item()

        avg_train_loss = epoch_train_loss / len(TrainLoader)
        train_losses_epochs.append(avg_train_loss)

        #  Validate 
        NeuralNetwork.eval()
        ValidProgressBar = tqdm(TestLoader, desc=f'Epoch {Epoch} [val]')

        pred_sequence   = np.array([])
        label_sequence  = np.array([])
        logits_sequence = np.array([])

        pred_token      = np.array([])
        label_token     = np.array([])
        logits_token    = np.array([])

        epoch_val_loss = 0.0

        with torch.no_grad():
            for data in ValidProgressBar:
                input_ids, attention_mask, offsets, Y1, Y2 = data
                input_ids      = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                Y2_gpu         = Y2.cuda()

                Logits_sequence, Logits_token = NeuralNetwork(input_ids, attention_mask, offsets)

                # val loss — same weighting as train
                loss_list_val = []
                for i, nuc_logits in enumerate(Logits_token):
                    true_len = min(len(nuc_logits), Y1.shape[1])
                    loss_list_val.append(LossFunction(
                        nuc_logits[:true_len].squeeze(),
                        Y1[i, :true_len].to(torch.float32).cuda()
                    ))
                val_loss_token    = torch.stack(loss_list_val).mean()
                val_loss_sequence = LossFunction(
                    Logits_sequence.squeeze(-1), Y2_gpu.to(torch.float32)
                )
                batch_val_loss = t * val_loss_sequence + (1 - t) * val_loss_token
                epoch_val_loss += batch_val_loss.item()

                # sequence-level outputs — straight from the model's head
                Prediction_sequence = Utils.Threshold.Threshold(
                    YPredicted=Logits_sequence.cpu(), ThresholdValue=ThresholdValue
                )
                logits_sequence = np.append(logits_sequence, Logits_sequence.cpu().detach().numpy())
                pred_sequence   = np.append(pred_sequence,   Prediction_sequence)
                label_sequence  = np.append(label_sequence,  Y2.cpu())

                # token-level outputs (per-nucleotide, BPE already resolved by model)
                for i, nuc_logits in enumerate(Logits_token):
                    nuc_logits_np = nuc_logits.cpu().detach().numpy().flatten()
                    nuc_pred      = (nuc_logits_np >= ThresholdValue).astype(np.int32)

                    true_label = np.array(Y1[i], dtype=np.int32)
                    true_label = true_label[:len(nuc_logits_np)]

                    logits_token = np.append(logits_token, nuc_logits_np)
                    pred_token   = np.append(pred_token,   nuc_pred)
                    label_token  = np.append(label_token,  true_label)

        avg_val_loss = epoch_val_loss / len(TestLoader)
        val_losses_epochs.append(avg_val_loss)

        #  metrics 
        Performance_sequence = np.zeros(shape=6, dtype=np.float32)
        Performance_token    = np.zeros(shape=6, dtype=np.float32)

        Performance_sequence[0], Performance_sequence[1], Performance_sequence[2], \
        Performance_sequence[3], Performance_sequence[4], Performance_sequence[5] \
            = Utils.Metrics.EvaluationMetricsSequence(
                y_pred=pred_sequence, y_true=label_sequence, y_logits=logits_sequence
            )

        Performance_token[0], Performance_token[1], Performance_token[2], \
        Performance_token[3], Performance_token[4], Performance_token[5] \
            = Utils.Metrics.EvaluationMetricsToken2(
                y_pred=pred_token, y_true=label_token, y_logits=logits_token
            )

        print(f'  Seq   metrics: {Performance_sequence}')
        print(f'  Token metrics: {Performance_token}')
        print(f'  Train loss: {avg_train_loss:.4f}  |  Val loss: {avg_val_loss:.4f}')

        #  save best 
        if Performance_token[5] > aupr_token:
            acc_sequence  = Performance_sequence[0]
            pre_sequence  = Performance_sequence[1]
            rec_sequence  = Performance_sequence[2]
            f1_sequence   = Performance_sequence[3]
            auc_sequence  = Performance_sequence[4]
            aupr_sequence = Performance_sequence[5]

            acc_token  = Performance_token[0]
            pre_token  = Performance_token[1]
            rec_token  = Performance_token[2]
            f1_token   = Performance_token[3]
            auc_token  = Performance_token[4]
            aupr_token = Performance_token[5]

            torch.save(
                NeuralNetwork.state_dict(),
                'ModelWeight/multiModel/' + TFsName + '/pytorch_model.bin'
            )

            # cache best-epoch predictions for plotting
            best_logits_token    = logits_token.copy()
            best_pred_token      = pred_token.copy()
            best_label_token     = label_token.copy()
            best_logits_sequence = logits_sequence.copy()
            best_pred_sequence   = pred_sequence.copy()
            best_label_sequence  = label_sequence.copy()

            earlystop = 0
        else:
            earlystop += 1
            if earlystop > 2:
                print(f"  Early stopping triggered at epoch {Epoch}.")
                break

    #  final report 
    print('Sequence Classification: Acc=%.3f, Pre=%.3f, Rec=%.3f, F1-S=%.3f, AUC=%.3f, AUPR=%.3f' % (
        acc_sequence, pre_sequence, rec_sequence, f1_sequence, auc_sequence, aupr_sequence))
    print('Token Classification:    Acc=%.3f, Pre=%.3f, Rec=%.3f, F1-S=%.3f, AUC=%.3f, AUPR=%.3f' % (
        acc_token, pre_token, rec_token, f1_token, auc_token, aupr_token))

    result_sequence.write(TFsName + '\t' + str(acc_sequence) + '\t' + str(pre_sequence) + '\t' +
                          str(rec_sequence) + '\t' + str(f1_sequence) + '\t' + str(auc_sequence) + '\t' +
                          str(aupr_sequence) + '\n')
    result_token.write(TFsName + '\t' + str(acc_token) + '\t' + str(pre_token) + '\t' +
                       str(rec_token) + '\t' + str(f1_token) + '\t' + str(auc_token) + '\t' +
                       str(aupr_token) + '\n')
    result_sequence.flush()
    result_token.flush()

    #  plots & saved predictions (best epoch) 
    if best_logits_token is not None:
        print(f"\nGenerating plots and saving outputs for {TFsName} ...")
        plot_and_save(
            TFsName         = TFsName,
            result_dir      = RESULT_DIR,
            train_losses    = train_losses_epochs,
            val_losses      = val_losses_epochs,
            logits_token    = best_logits_token,
            pred_token      = best_pred_token,
            label_token     = best_label_token,
            logits_sequence = best_logits_sequence,
            pred_sequence   = best_pred_sequence,
            label_sequence  = best_label_sequence,
            threshold       = ThresholdValue,
        )