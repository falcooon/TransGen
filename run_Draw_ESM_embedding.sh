##!/usr/bin/env bash
#set -euo pipefail
#
#PYTHON="/opt/conda/envs/esm_feats/bin/python"
#
## ===== 可改参数区 =====
#PY_SCRIPT="Batch_draw_esm_embedding.py"     # 你的 python 脚本文件名
#GPU="0"
#BATCH_SIZE=32
#MODEL="esm2_t33_650M_UR50D"
#FP16="--fp16"
#SORT="--sort_by_len"
#OVERWRITE="--overwrite"
#BATCH_PATH="/workspace/zhangzh/evodiff-main/output/"
#
## ===== 输入文件 & 对应输出文件夹 =====
#EXPS=("exp4_oadm_deepchem" \
#        "exp3_oadm_molt5" \
#        "exp2_oadm_UniMol" \
#        "exp1_oadm_RDKit" \
##        "exp3_oadm_UniMol_soft_prompt" \
#        "exp2_oadm_UniMol_adaln" \
#        "exp1_oadm_UniMol_cross_attn")
#FASTA_BASENAME="generated_samples_string.fasta"
#
## ===== 主流程 =====
#for exp in "${EXPS[@]}"; do
#  outdir="${BATCH_PATH}/${exp}/gent_seq"
#  input_fasta="${outdir}/${FASTA_BASENAME}"
#  out_h5="${outdir}/generated_seq_ESM2_embedding.h5"
#
#
#  echo "==== Running: ${fasta} -> ${out_h5} ===="
#  python "${PY_SCRIPT}" \
#    --fasta "${input_fasta}" \
#    --out_h5 "${out_h5}" \
#    --gpu "${GPU}" \
#    --batch_size "${BATCH_SIZE}" \
#    --model "${MODEL}" \
#    ${FP16} \
#    ${SORT} \
#    ${OVERWRITE}
#
#  echo "==== Done: ${out_h5} ===="
#done
#
#echo "All done."

#!/usr/bin/env bash
set -euo pipefail

PYTHON="/opt/conda/envs/esm_feats/bin/python"

# ===== 可改参数区 =====
PY_SCRIPT="Batch_draw_esm_embedding.py"
GPU="6"
BATCH_SIZE=32
MODEL="esm2_t33_650M_UR50D"
FP16="--fp16"
SORT="--sort_by_len"
OVERWRITE="--overwrite"
BATCH_PATH="/workspace/zhangzh/evodiff-main/output/"
#FASTA_BASENAME="generated_samples_string.fasta"
FASTA_BASENAME="generated_samples_string.fasta"

# ===== 实验列表 =====
EXPS=(
      "exp4_oadm_deepchem"
      "exp3_oadm_molt5"
      "exp2_oadm_UniMol"
      "exp1_oadm_RDKit"
      "exp3_oadm_UniMol_soft_prompt"
      "exp2_oadm_UniMol_adaln"
      "exp1_oadm_UniMol_cross_attn"
      )


OUTPUT=(
    "/workspace/zhangzh/evodiff-main/output/ESM2_align_exp23_oadm_molt5_adaln_true/gent_seq"
    )

## ===== 主流程 =====
#for exp in "${EXPS[@]}"; do
#  outdir="${BATCH_PATH}/${exp}/gent_seq"
##  outdir="/workspace/zhangzh/evodiff-main/output/gent_seq"
#  input_fasta="${outdir}/${FASTA_BASENAME}"
#  out_h5="${outdir}/generated_seq_ESM2_embedding.h5"
#  log_dir="${outdir}/logs"
#  mkdir -p "${log_dir}"
#  log_file="${log_dir}/esm_embed_${exp}_$(date +%Y%m%d_%H%M%S).log"

# ===== 主流程 =====
for outdir in "${OUTPUT[@]}"; do
#  outdir="${BATCH_PATH}/${exp}/gent_seq"
#  outdir="/workspace/zhangzh/evodiff-main/output/gent_seq"
  exp=111
#  input_fasta="${outdir}/${FASTA_BASENAME}"
  input_fasta="${outdir}/${FASTA_BASENAME}"
  out_h5="${outdir}/generated_seq_ESM2_embedding.h5"
#  out_h5="${outdir}/generated_seq_ESM2_embedding.h5"
  log_dir="${outdir}/logs"
  mkdir -p "${log_dir}"
  log_file="${log_dir}/esm_embed_${exp}_$(date +%Y%m%d_%H%M%S).log"

  echo "==== [$(date '+%F %T')] Start exp=${exp} ====" | tee -a "${log_file}"
  echo "PYTHON=${PYTHON}" | tee -a "${log_file}"
  echo "CONDA_PREFIX=${CONDA_PREFIX:-N/A}" | tee -a "${log_file}"
  echo "GPU=${GPU}  BATCH_SIZE=${BATCH_SIZE}  MODEL=${MODEL}  FP16=${FP16}" | tee -a "${log_file}"
  echo "input_fasta=${input_fasta}" | tee -a "${log_file}"
  echo "out_h5=${out_h5}" | tee -a "${log_file}"

  if [[ ! -f "${input_fasta}" ]]; then
    echo "!! [$(date '+%F %T')] Missing FASTA, skip: ${input_fasta}" | tee -a "${log_file}"
    echo "==== [$(date '+%F %T')] End exp=${exp} (SKIP) ====" | tee -a "${log_file}"
    echo
    continue
  fi

  # 关键：把 python 的 stdout+stderr 同时写入 log + 终端
  CUDA_VISIBLE_DEVICES="${GPU}" \
  "${PYTHON}" "${PY_SCRIPT}" \
    --fasta "${input_fasta}" \
    --out_h5 "${out_h5}" \
    --gpu "${GPU}" \
    --batch_size "${BATCH_SIZE}" \
    --model "${MODEL}" \
    ${FP16} \
    ${SORT} \
    ${OVERWRITE} 2>&1 | tee -a "${log_file}"

  echo "==== [$(date '+%F %T')] Done exp=${exp} ====" | tee -a "${log_file}"
  echo
done

echo "All done."
