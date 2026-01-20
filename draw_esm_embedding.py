# import torch
# import esm
#
# import pandas as pd
# import os
# import numpy as np
# from tqdm import tqdm
# import h5py
#
# ems_pt = r'/mnt/zhangzh/ESM_PT/esm2_t33_650M_UR50D.pt'
# pugtdb_uniprot_path = r'/mnt/zhangzh/pUGTdb/20251128_collection/download_PF00201_taxonomy33090/download_data/Uniprot_pUGTdb_concat(pretrain).csv'
# lmdb_path = r"/mnt/zhangzh/pUGTdb/20251128_collection/download_PF00201_taxonomy33090/download_data/Uniprot_pUGTdb_ESM_embedding.lmdb"
# SAVE_PATH = r"/mnt/zhangzh/pUGTdb/20251128_collection/download_PF00201_taxonomy33090/download_data/pUGTdb_Uniprot_ESM_embedding.h5"
#
#
# BATCH_SIZE = 4
#
# torch.cuda.empty_cache() # empty caches
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# torch.cuda.set_device(0)
#
# model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
# # model, alphabet = esm.pretrained.load_model_and_alphabet(ems_pt)
# batch_converter = alphabet.get_batch_converter()
# model.eval()
# model = model.cuda()
#
#
# data = pd.read_csv(pugtdb_uniprot_path)
# data_list = []
# for i in range(len(data)):
#     data_list.append((data.iloc[i]['id'], data.iloc[i]['seq']))
#
# with h5py.File(SAVE_PATH, 'w') as f:
#     grp = f.create_group("emb")
#
#     for i in tqdm(range(0,data.shape[0],BATCH_SIZE)):
#
#         batch_labels, batch_strs, batch_tokens = batch_converter(data_list[i: i + BATCH_SIZE])
#         batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
#         batch_tokens = batch_tokens.cuda()
#         with torch.no_grad():
#             results = model(batch_tokens, repr_layers=[33], return_contacts=True)
#             token_representations = results["representations"][33].cpu().numpy().astype(np.float16)
#
#
#         for j, tokens_len in enumerate(batch_lens):
#             seq_rep = token_representations[j,1:tokens_len-1]
#             seq_id = batch_labels[j]
#             seq = batch_strs[j]
#             grp.create_dataset(seq_id, data=seq_rep)
#
#         del results, token_representations, batch_tokens
#         torch.cuda.empty_cache()

import torch
import esm
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import h5py
from torch.cuda.amp import autocast  # 引入混合精度加速

# ================= 配置区 =================
# 建议显存允许的情况下尽可能调大
# 对于 650M 模型：
# 24G 显存 (3090/4090) 建议 BATCH_SIZE = 30 ~ 40
# 16G 显存 (T4/V100)   建议 BATCH_SIZE = 16 ~ 20
BATCH_SIZE = 32
pugtdb_uniprot_path = r'/root/Reaction_DATASETS/PF00201_taxonomy_Uniprot_Level2(condition_generation_standard_aa)+expertments+SMILES_ID.csv'
SAVE_PATH = r"/root/Reaction_DATASETS/pUGTdb_Uniprot_ESM_embedding_.h5"
# SAVE_PATH = r"/mnt/zhangzh/pUGTdb/20251128_collection/download_PF00201_taxonomy33090/download_data/pUGTdb_Uniprot_ESM_embedding_temp.h5"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# ================= 准备工作 =================
print("加载模型...")
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()
model = model.cuda()

print("读取并处理数据...")
data = pd.read_csv(pugtdb_uniprot_path)
# 优化 1: 快速构建列表 (比 iloc 快 100 倍)
# 直接将两列 zip 起来转 list
data_list = list(zip(data['seq_id'], data['seq']))

# 优化 2: 核心！按序列长度排序 (Sort by length)
# 这能极大减少 Padding 的数量，显著提升 Transformer 的推理速度
# 稍后写入时我们不在乎顺序，因为是字典格式
print("正在对序列按长度排序以加速推理...")

# h5_file = h5py.File(SAVE_PATH, 'r')
# mb_group = h5_file['emb']
# print(f"总共有 {len(mb_group)} 条序列")
# mb_group['A0A0K0PVW1']

data_list.sort(key=lambda x: len(x[1]), reverse=False)

# ================= 推理循环 =================
print(f"开始推理 (Batch Size: {BATCH_SIZE})...")

# 使用 'w' 模式每次都会重新创建文件
with h5py.File(SAVE_PATH, 'w') as f:
    grp = f.create_group("emb")

    # 循环
    for i in tqdm(range(0, len(data_list), BATCH_SIZE), mininterval=1.0):
        # 截取 batch
        batch_data = data_list[i: i + BATCH_SIZE]

        # ESM 处理
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)

        # 获取有效长度 (在 CPU 算即可)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        # 移至 GPU
        batch_tokens = batch_tokens.cuda()

        # 优化 3: 混合精度 (AMP) + 去掉 unnecessary 计算
        with torch.no_grad():
            with autocast():  # 开启 fp16 计算，加速 Tensor Core
                # 优化 4: return_contacts=False (必须！否则不仅慢还爆显存)
                results = model(batch_tokens, repr_layers=[33], return_contacts=False)

            # 此时还在 GPU，先切片再转 CPU，减少传输量
            token_representations = results["representations"][33]

        # 处理并保存
        for j, tokens_len in enumerate(batch_lens):
            # 获取 ID
            seq_id = str(batch_labels[j])

            # 截取 (去掉 <cls>, <eos>)
            # 注意：length 需要转为 int
            length = int(tokens_len.item())

            # 转 CPU -> Numpy -> fp16
            seq_rep = token_representations[j, 1: length - 1].cpu().numpy().astype(np.float16)

            # 写入 H5
            if seq_id in grp:
                # 处理重复 ID 的情况（虽然排序了，但原始数据可能有重复）
                pass
            else:
                grp.create_dataset(seq_id, data=seq_rep)

        # 优化 5: 删除 empty_cache()
        # PyTorch 的缓存分配器很聪明，频繁 empty_cache 会导致
        # CPU 和 GPU 强制同步，严重拖慢速度。只有 OOM 时才需要加。
        del results, token_representations, batch_tokens
        # torch.cuda.empty_cache()  <-- 删掉这一行！