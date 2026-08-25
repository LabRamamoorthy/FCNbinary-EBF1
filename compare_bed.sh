#!/bin/bash -f 1.0

TRUE=$1
PRED=$2
OUTDIR=$3

mkdir -p "${OUTDIR}"

# Remove header
tail -n +2 "${TRUE}" > "${OUTDIR}/true_noheader.bed"
tail -n +2 "${PRED}" > "${OUTDIR}/pred_noheader.bed"

# Count number of TRUE columns
N=$(head -1 "${OUTDIR}/true_noheader.bed" | awk '{print NF}')


# 1. TRUE regions overlapping PRED

bedtools intersect \
    -a "${OUTDIR}/true_noheader.bed" \
    -b "${OUTDIR}/pred_noheader.bed" \
    -wb | \
cut -f $((N+1))- | \
sort -k4,4 | \
awk '!seen[$4]++' \
> "${OUTDIR}/True_in_Pred.bed"


# 2. TRUE regions NOT in PRED

bedtools intersect \
    -a "${OUTDIR}/true_noheader.bed" \
    -b "${OUTDIR}/pred_noheader.bed" \
    -v | \
sort -k4,4 | \
awk '!seen[$4]++' \
> "${OUTDIR}/True_only.bed"


# 4. PRED-only (reverse)

bedtools intersect \
    -a "${OUTDIR}/pred_noheader.bed" \
    -b "${OUTDIR}/true_noheader.bed" \
    -v | \
sort -k4,4 | \
awk '!seen[$4]++' \
> "${OUTDIR}/Pred_only.bed"


# Summary counts (unique labels)

echo "Summary (unique labels):"

echo -n "overlap: "
cut -f4 "${OUTDIR}/True_in_Pred.bed" | wc -l

echo -n "True only: "
cut -f4 "${OUTDIR}/True_only.bed" | wc -l

echo -n "Pred_only: "
cut -f4 "${OUTDIR}/Pred_only.bed" | wc -l

echo "Done. Results saved in ${OUTDIR}/"