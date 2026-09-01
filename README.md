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

## Recreating the Conda Environment

You can run the below command to recreate the Conda environment from the `fcnbinary.yml` file:

```bash
conda env create -f fcnbinary.yml
```

Incase you want to name the environment `fcnbinary`:

```bash
conda env create -n fcnbinary -f fcnbinary.yml
```

Activate the environment:

```bash
conda activate fcnbinary
```

To verify that the environment was created successfully:

```bash
conda env list
```

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
| `--seqs_bed`                 | Required BED file containing peak coordinates (input file)                                 |
| `--pvalue_file`              | Required BigWig file containing the signal (ChIP signal)                                   |
| `--additional_feature_files` | Optional additional BigWig feature files (ATAC signal)                                     |
| `--genome_fasta`             | Required path to the genome FASTA file                                                     |
| `--chrom_sizes`              | Required path to the chromosome sizes file                                                 |
| `--seq_len`                  | Sequence length (default is `1001`)                                                        |

### Example

```bash
python bed2signal.py \
    -d EBF \
    -n EBF \
    -s 666 \
    -t 0.1 \
    --out_dir EBF/data_binary_0.1 \
    --seqs_bed EBF/WT2_0based.bed \
    --pvalue_file EBF/p-value.bw \
    --additional_feature_files EBF/ATAC2.bw \
    --genome_fasta Genome/mm9.fa \
    --chrom_sizes Genome/mm9.chrom.sizes
    --seq_len 1001
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
    -d EBF/data_binary_0.1 \
    -n EBF \
    -g 0 \
    -s 666 \
    -b 500 \
    -e 50 \
    -c models/unet_0.1
```

---

# Model Classification

Test a trained FCNsignal model performance on the specified test data. The output is Accuracy, Precision, Recall, F1-score, AUC, AUPR, Confusion matrix .

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
| `-o`     | Path to save results  |
| `-t`     | Threshold for binary classification (default is `0.5`)  |
| `-m`     | Minimal number of consecutive bases for sequence-level classification (default is `12`) |


### Example

```bash
python test_signal.py \
    -d EBF/data_binary_0.1 \
    -n EBF \
    -g 0 \
    -c models/unet_0.1
    -o results/unet_0.1_run1
```

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
| `-t`     | Threshold for binary classification (default is `0.5`)             |
| `-w`     | Length of binding regions (default is `60`)                                |
| `-c`     | Path to the trained model, e.g. `/your_path/FCNsignal/models/EBF` |
| `--features`     | Paths to bigWig files for additional features |
| `--smooth`     | Apply smoothing to features |
| `--chrom_sizes`     | Chromosome sizes file (default is `Genome/mm9.chrom.sizes`) |

### Example

```bash
python TFBS_locating.py \
    -i input.bed \
    -n EBF \
    -g 0 \
    -t 0.5 \
    -w 60 \
    -c models/unet_0.1 \
    --features EBF\ATAC2.bw
    --smooth 
    --chrom_sizes Genome/mm9.chrom.sizes
```



# Test Prediction

The `test_prediction.py` script provides a running model inference directly on BED regions classifying the input sequesces into with_peak(has TF) and without_peak(doesn't have TF).

## Command

```bash
python test_prediction.py \
    --chrom <> \
    --fasta <> \
    --bed <> \
    --model <> \
    --output <> \
    --features <>  \
    --threshold <> \
    --peak-bps <> \
    --window <> \
    --smooth
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
    --model models/unet_0.1/model_best_ebf.pth \
    --features EBF/ATAC2.bw \
    --output test_predict/output/bcell_chip \
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
bash compare.sh <TRUE> <PRED> <OUTDIR>
```

### Arguments

| Argument   | Description                                       |
| ---------- | ------------------------------------------------- |
| `<TRUE>`   | BED file containing the true binding regions      |
| `<PRED>`   | BED file containing the predicted binding regions |
| `<OUTDIR>` | Output directory for comparison results           |

## Example

```bash
bash compare_bed.sh test_predict/Ground_truth/chip_prob/common.bed test_predict/output6/bfcn_Aadipo/test_with_peak.tsv 

```
# 5. HOMER Motif Analysis

HOMER can be used to identify known motifs in the genomic regions predicted by FusedModel.

For example, `Pred_only.bed` contains predicted regions that are not present in the corresponding true regions. HOMER can be used to search these regions for motifs similar to a specified motif.

## Command

```bash
findMotifsGenome.pl \
    <INPUT_BED> \
    mm9 \
    <OUTPUT_PREFIX> \
    -find <MOTIF_FILE> \
    -size given \
    > <OUTPUT_BED>
```

### Example

For the `Pred_only.bed` regions generated from a specific prediction run:

```bash
findMotifsGenome.pl \
    test_predict/output/unet_0.1/bed_WT2/Pred_only.bed \
    mm9 \
    test_predict/output/unet_0.1/homer/Pred_only_test \
    -find motif/motif1.similar2.motif \
    -size given \
    > test_predict/output/unet_0.1/homer/Pred_only_test.bed
```

## Arguments

| Argument             | Description                                                                                           |
| -------------------- | ----------------------------------------------------------------------------------------------------- |
| `<INPUT_BED>`        | BED file containing genomic regions to be analyzed.                                                   |
| `mm9`                | Genome assembly used for the analysis. Replace this with the appropriate genome assembly if required. |
| `<OUTPUT_PREFIX>`    | Prefix/path used by HOMER for output files.                                                           |
| `-find <MOTIF_FILE>` | HOMER motif file containing the motif(s) to search for.                                               |
| `-size given`        | Uses the original size of each BED region rather than extracting a fixed-size region.                 |
| `> <OUTPUT_BED>`     | Redirects the motif-search results to the specified output file.                                      |


# FusedModel

FusedModel is a deep learning framework for **base-resolution transcription factor binding site (TFBS) prediction** that combines:

* **FCNsignalMod** for base-resolution signal prediction
* **BERTSNR2** for sequence-based representation learning

The fused architecture combines the complementary information learned by the FCN-based signal model and the BERT-based sequence model for TFBS prediction.

---

# Prerequisites and Dependencies

## Software

* Python 3.x
* PyTorch
* CUDA-enabled GPU (recommended for training)
* `bedtools` if required by the data-preparation workflow

## Python Packages

The fused model requires the packages used by both the FCNsignal and BERTSNR2 components, including:

* PyTorch
* NumPy
* pandas
* scikit-learn
* scipy
* pyBigWig
* pyfaidx
* transformers
* tokenizers

Install the required Python packages according to your environment.

## Recreating the Conda Environment

You can run the below command to recreate the Conda environment from the `fcnbinary.yml` file:

```bash
conda env create -f fused.yml
```

Incase you want to name the environment `fused`:

```bash
conda env create -n fused -f fused.yml
```

Activate the environment:

```bash
conda activate fused
```

To verify that the environment was created successfully:

```bash
conda env list
```
---

# 1. Data Preparation

The data-preparation script generates the datasets required by both components of the fused model.

The input consists of a BED file containing peak regions and a reference genome FASTA file.

## Command

```bash
python fused_dataset.py \
    --bed_file <> \
    --genome_file <> \
    --output_path <>
```

Additional options can be used to control sequence sampling and FCN signal generation.

## Arguments

### Input and Output

| Argument        | Description                                                             |
| --------------- | ----------------------------------------------------------------------- |
| `--bed_file`    | **Required.** BED file containing peak regions                          |
| `--genome_file` | **Required.** Reference genome FASTA file                               |
| `--output_path` | **Required.** Root directory where the prepared datasets will be stored |

### Sequence and Sampling

| Argument           | Default | Description                                                                              |
| ------------------ | ------- | ---------------------------------------------------------------------------------------- |
| `--seq_length`     | `500`   | Length of each sequence window in bp                                                     |
| `--negative_ratio` | `1`     | Number of negative windows sampled per positive window                                   |
| `--expand_bp`      | `3000`  | Number of bp by which peaks are expanded when excluding regions during negative sampling |
| `--max_n_fraction` | `0.1`   | Maximum fraction of `N` bases allowed in a sequence window                               |
| `--test_ratio`     | `0.2`   | Fraction of the dataset reserved for testing                                             |

### FCN Signal

| Argument          | Default | Description                                                                                                         |
| ----------------- | ------- | ------------------------------------------------------------------------------------------------------------------- |
| `--bigwig_file`   | `None`  | Primary BigWig signal used to generate FCN per-nucleotide labels. If omitted, FCN labels default to BERTSNR2 labels |
| `--chrom_sizes`   | `None`  | Chromosome sizes file.                                                    |
| `--smooth`        | `True`  | Apply moving-average smoothing to the BigWig signal                                                                 |
| `--no_smooth`     | —       | Disable BigWig signal smoothing                                                                                     |
| `--smooth_window` | `15`    | Moving-average smoothing window                                                                                     |
| `--fcn_threshold` | `0.1`   | Log10 signal threshold used to generate binary FCN labels                                                           |
| `--additional_bigwigs` | `None`   | Path to additional signal (Eg.ATAC)                                                           |

## Example
```bash
python fused_dataset.py 
    --bed_file fused_dataset/EBF.bed 
    --genome_file Genome/mm9.fa \
    --output_path fused_dataset/EBF_500/ \
    --seq_length 500 \
    --bigwig_file EBF/p-value.bw \
    --chrom_sizes Genome/mm9.chrom.sizes \
    --smooth \
    --fcn_threshold 0.1 \
    --additional_bigwigs EBF/ATAC2.bw
```

---

# Data Preparation Output

The output directory is the root directory containing the datasets used by FusedModel.

The expected structure is:

```text
fused_dataset/
├── bertsnr2/
│   ├── train/
│   │   └── dataset.txt
│   ├── test/
│   │   └── dataset.txt
│   └── ...
│
└── fcn/
    ├── train.npz
    ├── test.npz
    └── ...
```

These files are subsequently used by the FusedModel training script.

---

# 2. Model Training

After preparing the datasets, train the fused model using the BERTSNR2 and FCNsignalMod components.

## Command

```bash
python fusedRun_signal.py \
    -d <> \
    -g <> \
    -s <> \
    -c <> \
    --bert_path <>
```

## Arguments

| Argument      | Default                      | Description                                                              |
| ------------- | ---------------------------- | ------------------------------------------------------------------------ |
| `-d`          | —                            | **Required.** Root output directory generated during data preparation    |
| `-g`          | `0`                          | GPU device ID. Use `-1` to run on CPU                                    |
| `-s`          | `42`                         | Random seed for reproducibility                                          |
| `-c`          | `./checkpoints/`             | Directory where checkpoints and training outputs are saved               |
| `--bert_path` | `/home/ubuntu/data/dnabert2` | Path to the local DNABERT-2 model snapshot used as the BERTSNR2 backbone |

The dataset directory supplied with `-d` is expected to contain:

```text
<data_dir>/bertsnr2/train/dataset.txt
<data_dir>/fcn/train.npz
```

## Example
```bash
python fusedRun_signal.py 
    -d fused_dataset/EBF_500
    -g 0 /
    -s 42 /
    -c fusedModels/EBF_500_ATAC
    --bert_path data/dnabert2
```
---

# 3. Model Evaluation

The evaluation script evaluates a trained FusedModel on the held-out test dataset.

The evaluation requires:

* BERTSNR2 test dataset
* FCN test NPZ file
* Trained FusedModel weights
* DNABERT-2 model/tokenizer

## Command

```bash
python fusedEvaluationMetrics.py \
    --bertsnr_txt <> \
    --fcn_npz <> \
    --model_weights <> \
    --bert_path <> \
    --output_csv <>
```

Additional model and evaluation parameters can be specified from the command line.

## Arguments

### Dataset and Model Paths

| Argument          | Required | Description                            |
| ----------------- | -------- | -------------------------------------- |
| `--bertsnr_txt`   | Yes      | Path to the BERTSNR2 test dataset      |
| `--fcn_npz`       | Yes      | Path to the FCN test NPZ file          |
| `--model_weights` | Yes      | Path to the trained FusedModel weights |
| `--bert_path`     | Yes      | Path to the DNABERT-2 model/tokenizer  |

### Model Parameters

| Argument     | Default | Description                    |
| ------------ | ------- | ------------------------------ |
| `--motiflen` | `10`    | Motif length used by the model |
| `--dropout`  | `0.2`   | Dropout rate                   |

### Evaluation Parameters

| Argument        | Default | Description                  |
| --------------- | ------- | ---------------------------- |
| `--max_len`     | `512`   | Maximum sequence length      |
| `--batch_size`  | `16`    | Evaluation batch size        |
| `--num_workers` | `2`     | Number of DataLoader workers |
| `--threshold`   | `0.5`   | Classification threshold     |

### Output

| Argument       | Required | Description                 |
| -------------- | -------- | --------------------------- |
| `--output_csv` | Yes      | Path to the output CSV file |


---

# Evaluation Output

The evaluation script writes the prediction results to the CSV file specified by `--output_csv`.

For example:

```text
results/
└── fused_evaluation.csv
```

The output can be used to analyze model predictions and classification performance on the held-out test set.

The classification threshold can be changed using:

```bash
--threshold 0.5
```

For example:

```bash
--threshold 0.3
```

will classify predictions using a threshold of `0.3`.

---

# Parameters and Their Effects

## Sequence Length

```bash
--seq_length 500
```

Controls the length of the sequence windows used during dataset preparation.

The default is:

```text
500 bp
```

## Negative Sampling

```bash
--negative_ratio 1
```

Controls the number of negative windows generated per positive window.

For example:

```text
negative_ratio = 1
```

produces approximately one negative window per positive window.

## Negative Exclusion Region

```bash
--expand_bp 3000
```

Expands positive peak regions when identifying genomic regions from which negative examples should not be sampled.

## N-Bases Filtering

```bash
--max_n_fraction 0.1
```

Rejects windows containing more than the specified fraction of ambiguous `N` bases.

## Test Split

```bash
--test_ratio 0.2
```

Reserves 20% of the prepared data for testing by default.

## FCN Smoothing

Smoothing is enabled by default:

```bash
--smooth
```

The smoothing window is controlled by:

```bash
--smooth_window 15
```

To disable smoothing:

```bash
--no_smooth
```

## FCN Threshold

```bash
--fcn_threshold 0.1
```

Controls the threshold applied to the log10-transformed primary BigWig signal when generating binary FCN labels.

## Evaluation Threshold

```bash
--threshold 0.5
```

Controls the threshold used to convert model predictions into binary classifications during evaluation.

---

# DNABERT-2

FusedModel uses a local DNABERT-2 snapshot as the sequence backbone for the BERTSNR2 component.

Specify the model location using:

```bash
--bert_path /path/to/dnabert2
```

# 4. Fused Model Prediction

The `fusedTestPrediction.py` script performs inference using the trained **FusedModel** on arbitrary genomic regions provided in a BED file.

The model combines the sequence representation from **BERTSNR2** with the **FCNsignalMod** component and can optionally use additional BigWig features.

## Command

```bash
python fusedTestPrediction.py \
    --fasta <FASTA_FILE> \
    --bed <BED_FILE> \
    --model <MODEL_FILE> \
    --bert <DNABERT2_PATH> \
    --output <OUTPUT_DIR> \
    --extra-bigwig <BIGWIG_FILE> \
    --threshold <THRESHOLD> \
    --peak-bps <PEAK_BPS> \
    --max-len <MAX_LEN> \
    --smooth \
    --smooth-window <SMOOTH_WINDOW>
```

## Prediction Arguments

The prediction script accepts the following required and optional arguments.

### Required Arguments

| Argument   | Description                                               |
| ---------- | --------------------------------------------------------- |
| `--fasta`  | Reference genome FASTA file.                              |
| `--bed`    | Input BED file containing genomic regions to predict.     |
| `--model`  | Path to the trained `FusedModel` weights.                 |
| `--bert`   | Path to the local DNABERT-2 model/tokenizer.              |
| `--output` | Output directory where prediction results will be stored. |

### Optional Arguments

| Argument          | Default | Description                                                                                        |
| ----------------- | ------- | -------------------------------------------------------------------------------------------------- |
| `--extra-bigwig`  | `None`  | Optional BigWig feature files used as additional input features. Multiple files can be supplied.   |
| `--threshold`     | `0.5`   | Classification threshold used to convert model probabilities into binary predictions.              |
| `--peak-bps`      | `10`    | Minimum number of consecutive positive bases required to classify a sequence as containing a peak. |
| `--max-len`       | `512`   | Maximum sequence length used by the BERT component.                                                |
| `--smooth`        | `True`  | Enable or disable smoothing of BigWig features.                                                    |
| `--smooth-window` | `15`    | Window size used for moving-average smoothing.                                                     |

## Example

```bash
python fusedTestPrediction.py \
    --fasta Genome/mm9.fa \
    --bed test_predict/input/pro-B_Rep1_Chip.bed \
    --model checkpoints/fused_model/model_best.pth \
    --bert data/dnabert2 \
    --output test_predict/output \
    --extra-bigwig EBF/ATAC2.bw
```

