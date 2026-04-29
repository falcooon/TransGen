#!/bin/bash

# simple_parallel.sh
# 最简单直接的并行运行方式
set -e
PYTHON="/opt/conda/envs/evodiff/bin/python"


BASE_CMD="$PYTHON -u train.py \
  --config_fpath ./config/config38M.json \
  --gpus 1 --nodes 1 \
  --warmup --checkpoint_freq 30 --log_freq 200"


BASE_CMD_PRETRAIN="$PYTHON -u train.py \
  --config_fpath ./config/config38M.json \
  --gpus 1 --nodes 1 \
  "

run_exp () {
  GPU=$1
  PORT=$2
  OUT=$3
  shift 3
  mkdir -p "$OUT"
  echo "Launching: GPU=$GPU PORT=$PORT OUT=$OUT args=$*"
  CUDA_VISIBLE_DEVICES=$GPU $BASE_CMD \
    --master_port $PORT \
    --out_fpath $OUT \
    "$@" \
    > "$OUT/training.log" 2>&1 &
  echo $! > "$OUT/pid.txt"
}


# 20251218 对比不同的分子特征对生成模型的影响
#run_exp 0 8899 ./output/exp1_oadm_RDKit  --mask oadm   --reaction RDKit  --condition_insert add --ligand_feats_dim 2048
#run_exp 1 8900 ./output/exp2_oadm_UniMol --mask oadm --reaction UniMol  --condition_insert add  --ligand_feats_dim 512
#run_exp 2 8901 ./output/exp3_oadm_molt5  --mask oadm     --reaction molt5  --condition_insert add  --ligand_feats_dim 512
#run_exp 3 8902 ./output/exp4_oadm_deepchem  --mask oadm --reaction deepchem  --condition_insert add  --ligand_feats_dim 1613


# 20251219 使用相同模型对比同的特征注入方式有什么影响
#run_exp 0 8899 ./output/exp1_oadm_UniMol_cross_attn/  --mask oadm   --reaction UniMol  --condition_insert cross_attn --ligand_feats_dim 512
#run_exp 1 8900 ./output/exp2_oadm_UniMol_adaln/ --mask oadm --reaction UniMol  --condition_insert adaln  --ligand_feats_dim 512
#run_exp 2 8901 ./output/exp3_oadm_UniMol_soft_prompt/  --mask oadm     --reaction UniMol  --condition_insert soft_prompt  --ligand_feats_dim 512


## 20251229 对比不同的分子特征对生成模型的影响
#run_exp 0 8801 ./output/ESM2_align_exp11_oadm_RDKit/  --mask oadm   --reaction RDKit  --condition_insert add --ligand_feats_dim 2048 --align_ESM2 True --state_dict /workspace/zhangzh/evodiff-main/output/checkpoint649838_454.tar
#run_exp 1 8802 ./output/ESM2_align_exp12_oadm_UniMol/ --mask oadm --reaction UniMol  --condition_insert add  --ligand_feats_dim 512 --align_ESM2 True --state_dict /workspace/zhangzh/evodiff-main/output/checkpoint649838_454.tar
#run_exp 2 8803 ./output/ESM2_align_exp13_oadm_molt5/  --mask oadm     --reaction molt5  --condition_insert add  --ligand_feats_dim 512 --align_ESM2 True --state_dict /workspace/zhangzh/evodiff-main/output/checkpoint649838_454.tar
#run_exp 3 8804 ./output/ESM2_align_exp14_oadm_deepchem/  --mask oadm --reaction deepchem  --condition_insert add  --ligand_feats_dim 1613 --align_ESM2 True --state_dict /workspace/zhangzh/evodiff-main/output/checkpoint649838_454.tar


# 20251230 在20251229显示，molt5的ESM2聚类效果较好，使用不同的特征注入方式的影响
#run_exp 0 8801 ./output/ESM2_align_exp21_oadm_molt5_soft_prompt/   --reaction molt5  --condition_insert soft_prompt --ligand_feats_dim 512 --align_ESM2 True --state_dict /workspace/zhangzh/evodiff-main/output/checkpoint649838_454.tar
#run_exp 1 8802 ./output/ESM2_align_exp22_oadm_molt5_cross_attn/ --reaction molt5  --condition_insert cross_attn  --ligand_feats_dim 512 --align_ESM2 True --state_dict /workspace/zhangzh/evodiff-main/output/checkpoint649838_454.tar
#run_exp 2 8803 ./output/ESM2_align_exp23_oadm_molt5_adaln_true/    --reaction molt5  --condition_insert adaln  --ligand_feats_dim 512 --align_ESM2 True --state_dict /workspace/zhangzh/evodiff-main/output/pretrain_hav_ESM2_feats_align/checkpoint649838_454.tar
#run_exp 0 8804 ./output/Stage2_None_Condition/ --reaction None --align_ESM2 True --state_dict /workspace/zhangzh/evodiff-main/output/checkpoint649838_454.tar --data_dir /root/Reaction_DATASETS/

# 20251231 在20251229显示，molt5的ESM2聚类效果较好，且使用adaln的这种融合方式更好一点，所以这里进行一次反应使用多少信息的对比实验
#run_exp 3 8805 ./output/ESM2_align_exp31_oadm_molt5_adaln_substrate/ --reaction molt5  --Target_condition substrate --condition_insert adaln  --ligand_feats_dim 512 --align_ESM2 True --state_dict /workspace/zhangzh/evodiff-main/output/checkpoint649838_454.tar
#run_exp 3 8806 ./output/ESM2_align_exp32_oadm_molt5_adaln_acceptor/ --reaction molt5  --Target_condition acceptor --condition_insert adaln  --ligand_feats_dim 512 --align_ESM2 True --state_dict /workspace/zhangzh/evodiff-main/output/checkpoint649838_454.tar
#run_exp 3 8806 ./output/ESM2_align_exp32_oadm_molt5_adaln_dap/ --reaction molt5  --Target_condition dap --condition_insert adaln  --ligand_feats_dim 512 --align_ESM2 True --state_dict /workspace/zhangzh/evodiff-main/output/checkpoint649838_454.tar


run_exp 3 8803 ./output/ESM2_align_exp23_oadm_molt5_adaln_true/    --reaction molt5  --condition_insert adaln  --ligand_feats_dim 512 --align_ESM2 True --state_dict /workspace/zhangzh/evodiff-main/output/pretrain_hav_ESM2_feats_align/checkpoint649838_454.tar --data_dir /root/Reaction_DATASETS/


echo "All launched."
echo "Check logs: tail -f ./output/exp*/training.log"
echo "Check pids: cat ./output/exp*/pid.txt"




