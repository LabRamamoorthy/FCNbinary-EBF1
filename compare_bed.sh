TRUE=$1
PRED=$2

# Output directory
OUTDIR="test_predict/output/AWT2_features/bedtools_results"
mkdir -p ${OUTDIR}

# Remove header and run bedtools
tail -n +2 ${TRUE} > ${OUTDIR}/true_noheader.bed
tail -n +2 ${PRED} > ${OUTDIR}/pred_noheader.bed

# 1. True bases covered by predicted peaks
bedtools intersect -a ${OUTDIR}/true_noheader.bed -b ${OUTDIR}/pred_noheader.bed -u > ${OUTDIR}/True_in_Pred.bed

# 2. True bases NOT covered by predicted peaks
bedtools intersect -a ${OUTDIR}/true_noheader.bed -b ${OUTDIR}/pred_noheader.bed -v > ${OUTDIR}/True_not_in_Pred.bed

# 3. True-only (no overlap)
bedtools intersect -a ${OUTDIR}/true_noheader.bed -b ${OUTDIR}/pred_noheader.bed -v > ${OUTDIR}/True_only.bed

# 4. Pred-only (no overlap)
bedtools intersect -a ${OUTDIR}/pred_noheader.bed -b ${OUTDIR}/true_noheader.bed -v > ${OUTDIR}/Pred_only.bed

echo "✅ Bedtools intersections complete. Results saved in ${OUTDIR}/"
