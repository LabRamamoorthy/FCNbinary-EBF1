# it is for creating dataset for DNABERT2
import os
import pandas as pd

chip_seq_root = "ChIP-seq"
file_list = os.listdir(chip_seq_root)

for tf_name in file_list:
    print(f"Processing: {tf_name}")
    dataset_path = os.path.join(chip_seq_root, tf_name)

    seq_path = os.path.join(dataset_path, "sequence.txt")
    lab_path = os.path.join(dataset_path, "label.txt")

    if not os.path.isfile(seq_path) or not os.path.isfile(lab_path):
        print(f"  Skipping {tf_name}: missing sequence.txt or label.txt")
        continue

    sequence    = pd.read_csv(seq_path, header=None)
    dense_label = pd.read_csv(lab_path, header=None)

    n      = len(sequence)
    n_test = n // 5

    baseline_path = os.path.join(dataset_path, "baseline")
    os.makedirs(baseline_path, exist_ok=True)

    with open(os.path.join(baseline_path, "seq_train.txt"), "w") as fst, \
         open(os.path.join(baseline_path, "seq_test.txt"),  "w") as fse, \
         open(os.path.join(baseline_path, "lab_train.txt"), "w") as flt, \
         open(os.path.join(baseline_path, "lab_test.txt"),  "w") as fle:

        for i in range(n):
            if i < n_test:
                fse.write(str(sequence[0][i])    + "\n")
                fle.write(str(dense_label[0][i]) + "\n")
            else:
                fst.write(str(sequence[0][i])    + "\n")
                flt.write(str(dense_label[0][i]) + "\n")

    dnabert2_path = os.path.join(dataset_path, "dnabert2")
    os.makedirs(dnabert2_path, exist_ok=True)

    header = "sequence\tdenseLabel\tlabel\n"

    with open(os.path.join(dnabert2_path, "train.txt"), "w") as f_tr, \
         open(os.path.join(dnabert2_path, "test.txt"),  "w") as f_te:

        f_tr.write(header)
        f_te.write(header)

        for i in range(n):
            raw_seq   = str(sequence[0][i])
            raw_dense = str(dense_label[0][i])
            if i % 2 == 0:   # positive sample
                label = "1"
            else:             # negative sample
                label = "0"

            line = f"{raw_seq}\t{raw_dense}\t{label}\n"

            if i < n_test:
                f_te.write(line)
            else:
                f_tr.write(line)

    print(f"  -> {n_test} test / {n - n_test} train rows -> {dnabert2_path}")

print("Done.")