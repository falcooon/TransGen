#
############################################预训练的生成###################################################################
##!/bin/bash
#set -e
#
#PYTHON="/opt/conda/envs/evodiff/bin/python"
#SCRIPT_PATH="generate.py"
#
#BASE_ARGS="--model-type oa_dm_38M \
#           --config_fpath /workspace/zhangzh/evodiff-main/config/config38M.json \
#           --num-seqs 1000 \
#           --Target_condition reaction"
#
#
#declare -a EXPERIMENTS=(
#    "1 8901 /workspace/zhangzh/evodiff-main/output/pretrain_hav_ESM2_feats_align/gent_seq/ 1000 True /workspace/zhangzh/evodiff-main/output/checkpoint648586_453.tar /root/DATASETS/"
#    "1 8900 /workspace/zhangzh/evodiff-main/output/pretrain_non_ESM2_feats_align/gent_seq/ 1000 True /workspace/zhangzh/evodiff-main/output/pretrain_non_ESM2_feats_align/checkpoint1364752_361.tar /root/DATASETS/"
#)
#
#
#
#echo "开始执行实验任务流..."
#echo "Host: $(hostname) | Time: $(date '+%F %T')"
#echo
#
#for exp_config in "${EXPERIMENTS[@]}"; do
#    # 1) 处理 sleep 指令
#    if [[ $exp_config == sleep* ]]; then
#        wait
#        val=$(echo "$exp_config" | awk '{print $2}')
#        echo "休眠 $val 秒..."
#        sleep "$val"
#        continue
#    fi
#
#    # 2) 解析参数
#    IFS=' ' read -r -a cfg <<< "$exp_config"
#
#
#    GPU_ID=${cfg[0]}
#    PORT=${cfg[1]}
#    OUT_DIR=${cfg[2]}
#    SEQ_NUM=${cfg[3]}
#    ALIGN_ESM2=${cfg[4]}
#    SD_PATH=${cfg[5]}
#    DATA_DIR=${cfg[6]}
#
#
#    # 3) 创建目录
#    mkdir -p "$OUT_DIR"
#    LOG_FILE="${OUT_DIR%/}/run_generate.log"   # 去掉末尾可能的 /
#
#    echo ">>> 启动任务: GPU=$GPU_ID, Reaction=$ALIGN_ESM2"
#    echo "    OUT_DIR=$OUT_DIR"
#    echo "    LOG=$LOG_FILE"
#
#    # 4) 后台运行 + 记录 PID
#    CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON $SCRIPT_PATH \
#        --gpus 0 \
#        --out_fpath "$OUT_DIR" \
#        --state_dict "$SD_PATH" \
#        --data_dir "$DATA_DIR" \
#        --num-seqs "$SEQ_NUM" \
#        --align_ESM2 "$ALIGN_ESM2" \
##        --port "$PORT" \
#        > "$LOG_FILE" 2>&1 &
#
#    PID=$!
#    echo "    ✅ 已启动: PID=$PID"
#    echo "$PID" > "${OUT_DIR%/}/pid.txt"
#
#    # 5) 立刻检查一下进程是否活着
#    sleep 1
#    if ps -p "$PID" > /dev/null 2>&1; then
#        echo "    🟢 进程运行中 (ps ok)"
#    else
#        echo "    🔴 进程未在运行！请检查日志：$LOG_FILE"
#    fi
#
#    echo
#    sleep 3
#done
#
#echo "所有任务已启动，进入等待..."
#echo "你可以用下面命令查看："
#echo "  nvidia-smi"
#echo "  ps -ef | grep generate.py"
#echo "  tail -f <OUT_DIR>/run_generate.log"
#echo
#
## 等待全部后台任务结束
#wait
#echo "所有实验已完成！ Time: $(date '+%F %T')"
#





###################################根据不同反应特征生成序列#################################################################
#!/bin/bash

# generate_parallel.sh
# 简洁的并行生成脚本
set -e

PYTHON="/opt/conda/envs/evodiff/bin/python"
SCRIPT="generate.py"

# 基础命令（去掉需要覆盖的参数）
BASE_CMD="$PYTHON $SCRIPT \
  --model-type oa_dm_38M \
  --config_fpath /workspace/zhangzh/evodiff-main/config/config38M.json \
  "

run_exp () {
  GPU=$1
  OUT=$2
  shift 2
  mkdir -p "$OUT"
  echo "Launching: GPU=$GPU OUT=$OUT args=$*"

  CUDA_VISIBLE_DEVICES=$GPU $BASE_CMD \
    --out_fpath $OUT \
    "$@" \
    > "$OUT/run_generate.log" 2>&1 &

  echo $! > "$OUT/pid.txt"
  echo "PID: $! saved to $OUT/pid.txt"
  echo ""
}

# 2025.12.31 使用不同分子特征的生成实验
#run_exp 0 /workspace/zhangzh/evodiff-main/output/ESM2_align_exp11_oadm_RDKit/gent_seq/react_per_generate_20_seq/ \
#  --state_dict /workspace/zhangzh/evodiff-main/output/ESM2_align_exp11_oadm_RDKitcheckpoint108462_499.tar \
#  --ligand_feats_dim 2048 \
#  --align_ESM2 True \
#  --reaction RDKit \
#  --data_dir /root/Reaction_DATASETS/
#
#run_exp 1 /workspace/zhangzh/evodiff-main/output/ESM2_align_exp12_oadm_UniMol/gent_seq/react_per_generate_20_seq/ \
#  --state_dict /workspace/zhangzh/evodiff-main/output/ESM2_align_exp12_oadm_UniMolcheckpoint108462_499.tar \
#  --ligand_feats_dim 512 \
#  --align_ESM2 True \
#  --reaction UniMol \
#  --data_dir /root/Reaction_DATASETS/
#
#run_exp 2 /workspace/zhangzh/evodiff-main/output/ESM2_align_exp13_oadm_molt5/gent_seq/react_per_generate_20_seq/ \
#  --state_dict /workspace/zhangzh/evodiff-main/output/ESM2_align_exp13_oadm_molt5checkpoint108462_499.tar \
#  --ligand_feats_dim 512 \
#  --align_ESM2 True \
#  --reaction molt5 \
#  --data_dir /root/Reaction_DATASETS/
#


## 2025.1.6 使用不同的特征注入方式
#run_exp 0 /workspace/zhangzh/evodiff-main/output/ESM2_align_exp21_oadm_molt5_soft_prompt/gent_seq/ \
#  --state_dict /workspace/zhangzh/evodiff-main/output/ESM2_align_exp21_oadm_molt5_soft_prompt/checkpoint108462_499.tar \
#  --ligand_feats_dim 512 \
#  --align_ESM2 True \
#  --reaction molt5 \
#  --condition_insert soft_prompt\
#  --data_dir /root/Reaction_DATASETS/
#
#run_exp 1 /workspace/zhangzh/evodiff-main/output/ESM2_align_exp22_oadm_molt5_cross_attn/gent_seq/ \
#  --state_dict /workspace/zhangzh/evodiff-main/output/ESM2_align_exp22_oadm_molt5_cross_attn/checkpoint108462_499.tar \
#  --ligand_feats_dim 512 \
#  --align_ESM2 True \
#  --reaction molt5 \
#  --condition_insert cross_attn\
#  --data_dir /root/Reaction_DATASETS/

#run_exp 2 /workspace/zhangzh/evodiff-main/output/ESM2_align_exp23_oadm_molt5_adaln/gent_seq/\
#  --state_dict /workspace/zhangzh/evodiff-main/output/ESM2_align_exp23_oadm_molt5_adaln/checkpoint108462_499.tar \
#  --ligand_feats_dim 512 \
#  --align_ESM2 True \
#  --reaction molt5 \
#  --condition_insert adaln\
#  --data_dir /root/Reaction_DATASETS/

#
# 2025.1.6 使用不同层面的反应信息
#run_exp 1 /workspace/zhangzh/evodiff-main/output/ESM2_align_exp31_oadm_molt5_adaln_substrate/gent_seq/ \
#  --state_dict /workspace/zhangzh/evodiff-main/output/ESM2_align_exp31_oadm_molt5_adaln_substrate/checkpoint108462_499.tar \
#  --ligand_feats_dim 512 \
#  --align_ESM2 True \
#  --reaction molt5 \
#  --condition_insert adaln\
#  --Target_condition substrate\
#  --data_dir /root/Reaction_DATASETS/
#
#run_exp 2 /workspace/zhangzh/evodiff-main/output/ESM2_align_exp32_oadm_molt5_adaln_acceptor/gent_seq/ \
#  --state_dict /workspace/zhangzh/evodiff-main/output/ESM2_align_exp32_oadm_molt5_adaln_acceptor/checkpoint108462_499.tar \
#  --ligand_feats_dim 512 \
#  --align_ESM2 True \
#  --reaction molt5 \
#  --condition_insert adaln\
#  --Target_condition acceptor\
#  --data_dir /root/Reaction_DATASETS/

#run_exp 3 /workspace/zhangzh/evodiff-main/output/ESM2_align_exp33_oadm_molt5_adaln_dap/gent_seq/\
#  --state_dict /workspace/zhangzh/evodiff-main/output/ESM2_align_exp33_oadm_molt5_adaln_dap/checkpoint108462_499.tar \
#  --ligand_feats_dim 512 \
#  --align_ESM2 True \
#  --reaction molt5 \
#  --condition_insert adaln\
#  --Target_condition dap\
#  --data_dir /root/Reaction_DATASETS/

#run_exp 3 /workspace/zhangzh/evodiff-main/output/ESM2_align_exp33_oadm_molt5_adaln_dap/gent_seq/\
#  --state_dict /workspace/zhangzh/evodiff-main/output/ESM2_align_exp33_oadm_molt5_adaln_dap/checkpoint108462_499.tar \
#  --ligand_feats_dim 512 \
#  --align_ESM2 True \
#  --reaction molt5 \
#  --condition_insert adaln\
#  --Target_condition dap\
#  --data_dir /root/Reaction_DATASETS/

run_exp 1 /workspace/zhangzh/evodiff-main/output/ESM2_align_exp23_oadm_molt5_adaln_true/gent_seq/ \
  --state_dict /workspace/zhangzh/evodiff-main/output/ESM2_align_exp23_oadm_molt5_adaln_true/checkpoint66030_499.tar \
  --ligand_feats_dim 512 \
  --align_ESM2 True \
  --reaction molt5 \
  --condition_insert adaln\
  --Target_condition reaction\
  --data_dir /root/Reaction_DATASETS/



echo "所有实验已启动！"
echo ""
echo "监控方式："
echo "1. 查看GPU使用情况: nvidia-smi"
echo "2. 查看进程: ps aux | grep generate.py"
echo "3. 实时查看日志:"
echo "   tail -f /workspace/zhangzh/evodiff-main/output/ESM2_align_exp*/gent_seq/run_generate.log"
echo "4. 检查PID文件:"
echo "   cat /workspace/zhangzh/evodiff-main/output/ESM2_align_exp*/gent_seq/pid.txt"
echo ""
echo "等待所有实验完成..."
wait
echo ""
echo "所有实验已完成！时间: $(date '+%F %T')"
