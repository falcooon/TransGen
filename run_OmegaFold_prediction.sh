##!/bin/bash
#set -e
#
#BASE_DIR="/workspace/zhangzh/evodiff-main/output"
#CONDA_ENV="omegafold"
#
#run_fold () {
#  GPU=$1
#  OUTDIR=$2   # 实验目录名（BASE_DIR 下的子目录）
#  fasta_input="${BASE_DIR}/${OUTDIR}/gent_seqgenerated_samples_string.fasta"
#  output_path="${BASE_DIR}/${OUTDIR}/gent_seq/OmegaFold"
#
#  mkdir -p "$output_path"
#  echo "Launching: GPU=$GPU OUT=${BASE_DIR}/${OUTDIR}"
#
#  CUDA_VISIBLE_DEVICES=$GPU conda run -n $CONDA_ENV python -m omegafold \
#    "$fasta_input" "$output_path" \
#    --device cuda:0 --allow_tf32 True \
#    > "${output_path}/omegafold.log" 2>&1 &
#
#  echo $! > "${output_path}/pid.txt"
#}
#
## ====== 任务（像你那样一行一个）======
#run_fold 0 "ESM2_align_exp11_oadm_RDKit"
#run_fold 1 "ESM2_align_exp12_oadm_UniMol"
#run_fold 2 "ESM2_align_exp13_oadm_molt5"
#run_fold 3 "ESM2_align_exp14_oadm_deepchem"
#
#echo "All launched."
#echo "Check logs: tail -f ${BASE_DIR}/*/gent_seq/OmegaFold/omegafold.log"
#echo "Check pids: cat  ${BASE_DIR}/*/gent_seq/OmegaFold/pid.txt"




#!/bin/bash
set -e

BASE_DIR="/workspace/zhangzh/evodiff-main/output"
CONDA_ENV="omegafold"

run_fold () {
  GPU=$1
  OUTDIR=$2   # 实验目录名（BASE_DIR 下的子目录）
  fasta_input="${OUTDIR}/generated_samples_string.fasta"
  output_path="${OUTDIR}/OmegaFold"

  mkdir -p "$output_path"
  echo "Launching: GPU=$GPU OUT=${BASE_DIR}/${OUTDIR}"

  CUDA_VISIBLE_DEVICES=$GPU conda run -n $CONDA_ENV python -m omegafold \
    "$fasta_input" "$output_path" \
    --device cuda:0 --allow_tf32 True \
    > "${output_path}/omegafold.log" 2>&1 &

  echo $! > "${output_path}/pid.txt"
}

# ====== 任务（像你那样一行一个）======
run_fold 1 "/workspace/zhangzh/evodiff-main/output/ESM2_align_exp23_oadm_molt5_adaln_true/gent_seq/Rhea_61768_Rhea_61776"


echo "All launched."
echo "Check logs: tail -f ${BASE_DIR}/*/gent_seq/OmegaFold/omegafold.log"
echo "Check pids: cat  ${BASE_DIR}/*/gent_seq/OmegaFold/pid.txt"
