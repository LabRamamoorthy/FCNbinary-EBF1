# FCNBinary

## Prerequisites and Dependencies

* PyTorch 1.1 ([Install](https://pytorch.org/))
* CUDA 9.0
* Python 3.6
* Python packages:

  * biopython
  * scikit-learn
  * pyBigWig
  * scipy
  * pandas
  * matplotlib
  * seaborn
* `bedtools`
* Reference genome FASTA file

Download the required genome FASTA file from the [UCSC Genome Browser downloads](https://hgdownload.soe.ucsc.edu/downloads.html), unzip it, and place it in the `Genome/` directory.

For example:

```text
Genome/
├── hg38.fa
└── hg38.chrom.sizes
```

For mouse/mm9-based experiments:

```text
Genome/
├── mm9.fa
└── mm9.chrom.sizes
```

---

# Data Preparation

Prepare the signal data for a specified dataset using:

```bash
python bed2signal.py -d <> -n <> -s <>
```

### Arguments

| Argument                     | Description                                                                                |
| ---------------------------- | ------------------------------------------------------------------------------------------ |
| `-d`                         | Path to the dataset                                                                        |
| `-n`                         | Name of the specified dataset                                                              |
| `-s`                         | Random seed (default is `666`)                                                             |
| `-t`, `--threshold`          | Threshold for binary classification of signal after log10 normalization (default is `0.1`) |
| `--out_dir`                  | Required output directory                                                                  |
| `--seqs_bed`                 | Required BED file containing peak coordinates                                              |
| `--pvalue_file`              | Required BigWig file containing the signal                                                 |
| `--additional_feature_files` | Optional additional BigWig feature files                                                   |
| `--genome_fasta`             | Required path to the genome FASTA file                                                     |
| `--chrom_sizes`              | Required path to the chromosome sizes file                                                 |
| `--seq_len`                  | Sequence length (default is `1001`)                                                        |

### Example

```bash
python bed2signal.py \
    -d /your_path/FCNsignal/EBF \
    -n EBF \
    -s 666
```

---

# Model Training

Train FCNsignal models on a specified dataset:

```bash
python run_signal.py -d <> -n <> -g <> -s <> -b <> -e <> -c <>
```

### Arguments

| Argument | Description                                                                      |
| -------- | -------------------------------------------------------------------------------- |
| `-d`     | Path to the specified dataset, e.g. `/your_path/FCNsignal/EBF/data`     |
| `-n`     | Name of the specified dataset, e.g. `EBF`                                       |
| `-g`     | GPU device ID (default is `0`)                                                   |
| `-s`     | Random seed                                                                      |
| `-b`     | Number of sequences per batch (default is `500`)                                 |
| `-e`     | Number of training epochs (default is `50`)                                      |
| `-c`     | Path for storing trained models, e.g. `/your_path/FCNsignal/models/EBF |

### Example

```bash
python run_signal.py \
    -d /your_path/FCNsignal/EBF/data \
    -n EBF \
    -g 0 \
    -s 666 \
    -b 500 \
    -e 50 \
    -c /your_path/FCNsignal/models/EBF
```

## Output

A trained model is stored in the specified model directory.

For example:

```text
/your_path/FCNsignal/models/EBF/model_best.pth
```

---

# Model Classification

Test a trained FCNsignal model on the specified test data:

```bash
python test_signal.py -d <> -n <> -g <> -c <>
```

### Arguments

| Argument | Description                                                                  |
| -------- | ---------------------------------------------------------------------------- |
| `-d`     | Path to the specified dataset, e.g. `/your_path/FCNsignal/EBF/data` |
| `-n`     | Name of the specified dataset, e.g. `EBF`                                   |
| `-g`     | GPU device ID (default is `0`)                                               |
| `-c`     | Path to the trained model, e.g. `/your_path/FCNsignal/models/EBF   |

### Example

```bash
python test_signal.py \
    -d /your_path/FCNsignal/EBF/data \
    -n EBF \
    -g 0 \
    -c /your_path/FCNsignal/models/EBF
```

## Output

The script generates `record.txt`, containing the results for the metrics.

---

# Locating TFBSs

Locate potential transcription factor binding regions from input BED files of arbitrary length:

```bash
python TFBS_locating.py -i <> -n <> -g <> -t <> -w <> -c <>
```

### Arguments

| Argument | Description                                                                |
| -------- | -------------------------------------------------------------------------- |
| `-i`     | Input BED file, e.g. `/your_path/FCNsignal/input.bed`                      |
| `-n`     | Name of the specified dataset, e.g. `EBF`                                 |
| `-g`     | GPU device ID (default is `0`)                                             |
| `-t`     | Threshold used to determine binding regions (default is `1.5`)             |
| `-w`     | Length of binding regions (default is `60`)                                |
| `-c`     | Path to the trained model, e.g. `/your_path/FCNsignal/models/EBF` |

### Example

```bash
python TFBS_locating.py \
    -i /your_path/FCNsignal/input.bed \
    -n EBF \
    -g 0 \
    -t 1.5 \
    -w 60 \
    -c /your_path/FCNsignal/models/EBF
```

## Output

The outputs include:

* Base-resolution predictions for the input sequences
* Potential TF binding regions in BED format
* Line plots showing the base-resolution predictions


# Test Prediction

The `test_prediction.py` script provides a command-line interface for running model inference directly on BED regions.

Unlike the original dataset-based testing workflow, this script allows the user to specify the reference genome, chromosome sizes, input BED file, trained model, output directory, and optional BigWig features directly from the command line.

## Command

```bash
python test_prediction.py \
    --chrom <> \
    --fasta <> \
    --bed <> \
    --model <> \
    --output <> \
    [--features <> ...] \
    [--threshold <>] \
    [--peak-bps <>] \
    [--window <>] \
    [--smooth]
```

## Arguments

| Argument      | Description                                                                                                         |
| ------------- | ------------------------------------------------------------------------------------------------------------------- |
| `--chrom`     | Chromosome sizes file, e.g. `Genome/mm9.chrom.sizes`                                                                |
| `--fasta`     | Reference genome FASTA file, e.g. `Genome/mm9.fa`                                                                   |
| `--bed`       | Input BED file containing genomic regions to predict                                                                |
| `--model`     | Trained model checkpoint (`.pth`)                                                                                   |
| `--output`    | Directory in which prediction results are stored                                                                    |
| `--features`  | Optional BigWig feature files. Multiple files can be supplied                                                       |
| `--threshold` | Sigmoid prediction threshold (default is `0.5`)                                                                     |
| `--peak-bps`  | Minimum number of consecutive positive bases required to classify a sequence as containing a peak (default is `12`) |
| `--window`    | Smoothing window for additional BigWig features (default is `15`)                                                   |
| `--smooth`    | Apply smoothing to additional BigWig features                                                                       |

## Example
```bash
python test_prediction.py \
    --chrom Genome/mm9.chrom.sizes \
    --fasta Genome/mm9.fa \
    --bed test_predict/input/pro-B_Rep1_Chip.bed \
    --model models/adipo_1000_0.1/model_best_ebf.pth \
    --features EBF/ATAC2.bw \
    --output test_predict/output6/afcn_Abcell_chip \
    --threshold 0.5 \
    --peak-bps 12 \
    --window 15 \
    --smooth
```


## Output

The script creates the specified output directory automatically.

Four output files are generated:

```text
test_with_peak.tsv
test_without_peak.tsv
test_with_peak_full.tsv
test_without_peak_full.tsv
```

### `test_with_peak.tsv`

Contains sequences for which the model predicts a continuous peak satisfying the specified `--peak-bps` criterion.

Example:

```text
chr    start    end    label
chr1   10000    15000  with_peak_00001
chr1   25000    30000  with_peak_00002
```

### `test_without_peak.tsv`

Contains sequences for which the model does not detect the required continuous peak.

### `test_with_peak_full.tsv`

Contains the complete base-resolution prediction for sequences classified as containing a peak.

### `test_without_peak_full.tsv`

Contains the complete base-resolution prediction for sequences classified as not containing a peak.

The prediction values are stored as comma-separated binary values.

---

# Prediction Comparison

The `compare.sh` script compares true binding regions with predicted binding regions.

The script accepts the TRUE BED file, PRED BED file, and output directory from the command line.

## Command

```bash
./compare.sh <TRUE> <PRED> <OUTDIR>
```

### Arguments

| Argument   | Description                                       |
| ---------- | ------------------------------------------------- |
| `<TRUE>`   | BED file containing the true binding regions      |
| `<PRED>`   | BED file containing the predicted binding regions |
| `<OUTDIR>` | Output directory for comparison results           |

## Example

```bash
./compare.sh \
    test_predict/input/true.bed \
    test_predict/output6/predictions.bed \
    test_predict/output6/bfcn_op9_7/com_adipo_neg
```




