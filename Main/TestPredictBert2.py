#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from pyfaidx import Fasta
from transformers import AutoTokenizer


# Setup paths

BERTSNR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BERTSNR_ROOT, 'Model'))

# Import DNABERT-2 based BertSNR model
from BertSNR2 import BERTSNR2

# DNABERT-2 tokenizer path
DNABERT2_PATH = "/home/ubuntu/data/dnabert2"


# Utilities

def read_bed_file(bed_path):
    """Read BED file with at least chr, start, end"""
    df = pd.read_csv(bed_path, sep='\t', header=None)
    if len(df.columns) >= 4:
        df.columns = ['chr', 'start', 'end', 'sequence'] + [f'col{i}' for i in range(4, len(df.columns))]
    else:
        df.columns = ['chr', 'start', 'end'] + [f'col{i}' for i in range(3, len(df.columns))]
    return df

def extract_sequences_from_fasta(bed_df, fasta_path):
    """Extract sequences from genome FASTA using pyfaidx"""
    print(f"Loading genome FASTA: {fasta_path}")
    fasta = Fasta(fasta_path)
    sequences = []
    for _, row in tqdm(bed_df.iterrows(), total=len(bed_df), desc="Extracting sequences"):
        try:
            seq = fasta[row['chr']][int(row['start']):int(row['end'])].seq
            sequences.append(seq.upper())
        except Exception as e:
            print(f"Warning: could not extract {row['chr']}:{row['start']}-{row['end']} -> {e}")
            sequences.append("")
    return sequences

def predict_batch(model, tokenizer, sequences, device, batch_size=1):
    """Predict sequence-level and base-level outputs using DNABERT-2"""
    model.eval()
    predictions, probabilities = [], []
    token_level_outputs, base_level_outputs = [], []

    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="Predicting"):
            batch_seqs = sequences[i:i+batch_size]

            # Tokenize with offset mapping for nucleotide-level alignment
            encoding = tokenizer(
                batch_seqs,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512,
                return_offsets_mapping=True
            )

            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            # offset_mapping stays on CPU (used for indexing, not passed to BERT)
            offset_mapping = encoding['offset_mapping']

            # Forward pass
            logits_seq, all_nucleotide_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                offset_mapping=offset_mapping
            )

            # Sequence-level predictions
            probs = logits_seq.cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)
            predictions.extend(preds)
            probabilities.extend(probs)

            # Base-level predictions from variable-length nucleotide logits
            for b_idx, seq in enumerate(batch_seqs):
                nuc_logits = all_nucleotide_logits[b_idx]  # (seq_len, 1) tensor
                if nuc_logits.shape[0] > 0:
                    base_pred = nuc_logits.squeeze(-1).cpu().numpy()
                else:
                    base_pred = np.zeros(len(seq))

                # token_level here refers to per-nucleotide (already mapped)
                token_level_outputs.append(base_pred)
                base_level_outputs.append(base_pred)

    return np.array(predictions), np.array(probabilities), token_level_outputs, base_level_outputs


# Main

def main():
    parser = argparse.ArgumentParser(description="Predict TFBS using BertSNR2")
    parser.add_argument('--model_path', type=str, required=True, help="Trained model .bin path")
    parser.add_argument('--bed_file', type=str, required=True, help="Input BED file")
    parser.add_argument('--fasta_file', type=str, required=True, help="Genome FASTA file (required)")
    parser.add_argument('--output_dir', type=str, default='predictions', help="Output directory")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
    parser.add_argument('--threshold', type=float, default=0.5, help="Peak probability threshold")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load tokenizer
    print("[1/4] Loading DNABERT-2 tokenizer and BertSNR model...")
    tokenizer = AutoTokenizer.from_pretrained(DNABERT2_PATH, trust_remote_code=True)

    # Load model
    model = BERTSNR()
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model loaded.")

    # Read BED
    print(f"[2/4] Reading BED file: {args.bed_file}")
    bed_df = read_bed_file(args.bed_file)
    print(f"Found {len(bed_df)} regions.")

    # Extract sequences from FASTA
    print("[3/4] Extracting sequences from genome FASTA...")
    sequences = extract_sequences_from_fasta(bed_df, args.fasta_file)
    bed_df['sequence'] = sequences

    # Remove empty sequences
    valid_mask = np.array([len(seq) > 0 for seq in sequences])
    if not all(valid_mask):
        n_invalid = np.sum(~valid_mask)
        print(f"Warning: Removing {n_invalid} regions with empty sequences")
        bed_df = bed_df[valid_mask].reset_index(drop=True)
        sequences = [sequences[i] for i in range(len(sequences)) if valid_mask[i]]

    if len(sequences) == 0:
        print("ERROR: No valid sequences to predict on!")
        sys.exit(1)

    print(f"Ready to predict on {len(sequences)} sequences.")

    # Predict
    print("[4/4] Running predictions...")
    preds, probs, token_preds, base_preds = predict_batch(
        model, tokenizer, sequences, device, batch_size=args.batch_size
    )

    bed_df['prediction'] = preds
    bed_df['prob_peak'] = probs

    # Separate peaks
    with_peak = bed_df[bed_df['prediction'] == 1]
    without_peak = bed_df[bed_df['prediction'] == 0]

    # Save standard BED files
    with_peak_path = os.path.join(args.output_dir, "with_peak.bed")
    without_peak_path = os.path.join(args.output_dir, "without_peak.bed")
    with_peak[['chr','start','end']].to_csv(with_peak_path, sep='\t', header=False, index=False)
    without_peak[['chr','start','end']].to_csv(without_peak_path, sep='\t', header=False, index=False)
    print(f"Saved BED files:\n  {with_peak_path}\n  {without_peak_path}")

    # Save detailed predictions
    all_pred_path = os.path.join(args.output_dir, "all_predictions.tsv")
    bed_df.to_csv(all_pred_path, sep='\t', index=False)

    token_base_path = os.path.join(args.output_dir, "token_base_predictions.tsv")
    with open(token_base_path, "w") as f:
        f.write("chr\tstart\tend\ttoken_level_pred\tbase_level_pred\n")
        for idx, row in bed_df.iterrows():
            token_str = ",".join(map(str, token_preds[idx]))
            base_str = ",".join(map(str, base_preds[idx]))
            f.write(f"{row['chr']}\t{row['start']}\t{row['end']}\t{token_str}\t{base_str}\n")
    print(f"Saved token/base predictions: {token_base_path}")

    # Summary
    print(f"\nTotal regions: {len(bed_df)}")
    print(f"With peak: {len(with_peak)} ({len(with_peak)/len(bed_df)*100:.1f}%)")
    print(f"Without peak: {len(without_peak)} ({len(without_peak)/len(bed_df)*100:.1f}%)")
    print(f"Mean probability: {probs.mean():.3f}")
    print(f"Probability range: [{probs.min():.3f}, {probs.max():.3f}]")

    # Show first 5 predictions
    print("\nExample predictions (first 5):")
    print(bed_df[['chr','start','end','prediction','prob_peak']].head().to_string(index=False))

if __name__ == "__main__":
    main()