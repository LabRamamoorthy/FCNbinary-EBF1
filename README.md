# BertSNR

## 1. Environment setup

#### 1.1 Create and activate a new virtual environment

```
conda create -n bertsnr2 python=3.7
conda activate bertsnr2
```

#### 1.2 Install the package and other requirements

(Required)

```
cd BertSNR2
conda install --file requirements.txt
```

## 2. Model pipeline

#### 3.1 Data processing

First, use `Dataset/createSequence.py` to generate DNA sequences from the ChIP-seq BED file and reference genome.

The available arguments are:

| Argument           | Default | Description                                                                                               |
| ------------------ | ------: | --------------------------------------------------------------------------------------------------------- |
| `--bed_file`       |       — | Input BED file containing peak coordinates                                                                |
| `--genome_file`    |       — | Reference genome FASTA file                                                                               |
| `--output_path`    |       — | Path for the generated sequence output                                                                    |
| `--seq_length`     |   `100` | Length of each DNA sequence                                                                               |
| `--negative_ratio` |     `1` | Ratio of negative to positive sequences                                                                   |
| `--expand_bp`      |  `3000` | Expand peak coordinates by this many base pairs on each side when excluding regions for negative sampling |
| `--max_n_fraction` |     `0` | Maximum fraction of `N` bases allowed in a negative window                                                |
| `--seed`           |    `42` | Random seed for reproducible negative sampling                                                            |

Example:

```
python Dataset/createSequence.py \
    --bed_file <path/to/peaks.bed> \
    --genome_file <path/to/genome.fa> \
    --output_path <path/to/output> \
    --seq_length 100 \
    --negative_ratio 1 \
    --expand_bp 3000 \
    --max_n_fraction 0 \
    --seed 42
```

After generating the DNA sequences, use `Dataset/CreateDataset.py` to generate the k-mer sequence files required for model training.

```
python Dataset/CreateDataset.py
```

#### 3.2 Train

Use `Main/TrainMultitasking.py` to train model.

#### 3.3 Predict

Use `Main/Predict.py` to predict TFBS.

## Reference

1. Z. Zhou, Y. Ji, W. Li, P. Dutta, R. Davuluri, and H. Liu, “Dnabert-2: Efficient foundation model and benchmark for multi-species genome,” arXiv preprint:2306.15006, 2023.

2. H. Luo, L. Tang, M. Zeng, R. Yin, P. Ding, L. Luo, and M. Li, “Bertsnr: an interpretable deep learning framework for single-nucleotide resolution identification of transcription factor binding sites based on dna language model,” Bioinformatics, vol. 40, no. 8, p. btae461, 2024.
