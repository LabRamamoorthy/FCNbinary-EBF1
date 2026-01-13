TRUE=$1
PRED=$2

OUTDIR="test_predict/output/dynamic_pos_0.4/bedtools_results"
mkdir -p ${OUTDIR}

# Remove header
tail -n +2 ${TRUE} > ${OUTDIR}/true_noheader.bed
tail -n +2 ${PRED} > ${OUTDIR}/pred_noheader.bed

# Count number of TRUE columns
N=$(head -1 ${OUTDIR}/true_noheader.bed | awk '{print NF}')

# Intersect and remove TRUE columns
bedtools intersect \
    -a ${OUTDIR}/true_noheader.bed \
    -b ${OUTDIR}/pred_noheader.bed \
    -wb | \
cut -f $((N+1))- \
> ${OUTDIR}/True_in_Pred.bed

# 2. TRUE bases NOT covered by PRED
bedtools intersect \
    -a ${OUTDIR}/true_noheader.bed \
    -b ${OUTDIR}/pred_noheader.bed \
    -v \
    > ${OUTDIR}/True_not_in_Pred.bed

# 3. TRUE-only (essentially same as #2)
cp ${OUTDIR}/True_not_in_Pred.bed ${OUTDIR}/True_only.bed

# 4. PRED-only (reverse perspective)
bedtools intersect \
    -a ${OUTDIR}/pred_noheader.bed \
    -b ${OUTDIR}/true_noheader.bed \
    -v \
    > ${OUTDIR}/Pred_only.bed

echo "Done. Results saved in ${OUTDIR}/"


# TRUE=$1
# PRED=$2

# # Output directory
# OUTDIR="test_predict/output/EBF_ATAC/bedtools_results"
# mkdir -p ${OUTDIR}

# # Remove header and run bedtools
# tail -n +2 ${TRUE} > ${OUTDIR}/true_noheader.bed
# tail -n +2 ${PRED} > ${OUTDIR}/pred_noheader.bed

# # 1. True bases covered by predicted peaks
# bedtools intersect -a ${OUTDIR}/true_noheader.bed -b ${OUTDIR}/pred_noheader.bed -u > ${OUTDIR}/True_in_Pred.bed

# # 2. True bases NOT covered by predicted peaks
# bedtools intersect -a ${OUTDIR}/true_noheader.bed -b ${OUTDIR}/pred_noheader.bed -v > ${OUTDIR}/True_not_in_Pred.bed

# # 3. True-only (no overlap)
# bedtools intersect -a ${OUTDIR}/true_noheader.bed -b ${OUTDIR}/pred_noheader.bed -v > ${OUTDIR}/True_only.bed

# # 4. Pred-only (no overlap)
# bedtools intersect -a ${OUTDIR}/pred_noheader.bed -b ${OUTDIR}/true_noheader.bed -v > ${OUTDIR}/Pred_only.bed

# echo "Bedtools intersections complete. Results saved in ${OUTDIR}/"