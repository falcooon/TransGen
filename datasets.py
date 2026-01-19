from typing import Union
from pathlib import Path
import lmdb
import subprocess
import string
import json
import os
from os import path
import pickle as pkl
from scipy.spatial.distance import squareform, pdist, hamming, cdist
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

from sequence_models.utils import Tokenizer, parse_fasta
from sequence_models.constants import trR_ALPHABET, DIST_BINS, PHI_BINS, THETA_BINS, OMEGA_BINS, STOP, PAD, \
    PROTEIN_ALPHABET
from sequence_models.gnn import bins_to_vals
from sequence_models.pdb_utils import process_coords
import h5py
#
class UniportpUGTdbDataset(Dataset):
    """
    Dataset that pulls from UniRef/Uniclust downloads.

    The data folder should contain the following:
    - 'consensus.fasta': consensus sequences, no line breaks in sequences
    - 'splits.json': a dict with keys 'train', 'valid', and 'test' mapping to lists of indices
    - 'lengths_and_offsets.npz': byte offsets for the 'consensus.fasta' and sequence lengths
    """

    def __init__(self, data_dir, arg, split):
        self.data_dir = data_dir
        self.split = split
        with open(data_dir + 'splits1.json', 'r') as f:
            self.indices = json.load(f)[self.split]
        metadata = np.load(self.data_dir + 'lengths_and_offsets1.npz')
        self.offsets = metadata['seq_offsets']
        self.namesets = metadata['name_offsets']
        self.h5_path = data_dir +"pUGTdb_Uniprot_ESM_embedding_.h5"
        self.h5_file = None
        self.emb_group = None

        if arg.reaction != 'None':
            self.reaction_feats_path = data_dir + f"smiles_{arg.reaction}_embedding1.h5"
            self.reaction_feats = None
            self.index_pd = pd.read_csv(data_dir + "Level2_expertments_SMILES_ID1.csv")
        else:
            self.reaction_feats_path = None
            self.reaction_feats = None
            self.index_pd = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
            self.emb_group = self.h5_file['emb']
        if self.reaction_feats_path is not None:
            self.reaction_feats = h5py.File(self.reaction_feats_path, 'r')

        idx = self.indices[idx]
        offset = self.offsets[idx]
        nameset = self.namesets[idx]


        with open(self.data_dir + 'consensus1.fasta') as f:
            f.seek(offset)
            consensus = f.readline()[:-1]
            f.seek(nameset)
            seq_name = f.readline().strip()[1:]
            esm_emb = self.emb_group[seq_name][:]
            if self.index_pd is not None:
                row = self.index_pd[self.index_pd['seq_id'] == seq_name]
                # # print( row)
                # # print(row.keys())
                # # print(f"sdfdas:{str(row['acceptor_id'])},fdcadsc:{type(row['acceptor_id'])}，{seq_name}")
                # print("asdf:",self.reaction_feats_path,row['acceptor_id'],seq_name)
                ac_feats = self.reaction_feats[row['acceptor_id'].values[0]]
                do_feats = self.reaction_feats[row['donor_id'].values[0]]
                get_feats = self.reaction_feats[row['get_loop_id'].values[0]]
                lo_feats = self.reaction_feats[row['lost_loop_id'].values[0]]
                H_id = row['H_id'].values[0]

                try:
                    H_feats = np.zeros_like(ac_feats)
                except:
                    H_feats = self.reaction_feats[H_id]

                reaction_feats = np.vstack([ac_feats,do_feats,get_feats,lo_feats,H_feats])

                return (consensus,esm_emb,seq_name,reaction_feats)
        return (consensus,esm_emb,seq_name)




#
# import json
# import numpy as np
# import h5py
# import pandas as pd
# import torch
# from torch.utils.data import Dataset
# from tqdm import tqdm  # 如果没有安装 tqdm，可以 pip install tqdm，或者删除相关代码
#
#
# class UniportpUGTdbDataset(Dataset):
#     """
#     针对 pUGTdb 优化的全内存加载 Dataset。
#     核心策略：以空间换时间，启动时将所有数据读入 RAM，消除训练时的 IO 延迟。
#     """
#
#     def __init__(self, data_dir, arg, split):
#         self.data_dir = data_dir
#         self.split = split
#
#         # 1. 加载索引文件
#         print(f"[{split}] Loading splits and metadata...")
#         with open(data_dir + 'splits.json', 'r') as f:
#             self.indices = json.load(f)[self.split]
#
#         metadata = np.load(self.data_dir + 'lengths_and_offsets.npz')
#         seq_offsets = metadata['seq_offsets']
#         name_offsets = metadata['name_offsets']
#
#         # ---------------------------------------------------------------
#         # 内存缓存初始化
#         # ---------------------------------------------------------------
#         self.cache_consensus = {}  # 存储序列字符串
#         self.cache_seqname = {}  # 存储序列名称
#         self.cache_esm = {}  # 存储 ESM Embedding (Numpy Array)
#         self.cache_reaction = {}  # 存储反应特征 (如果启用)
#
#         # ---------------------------------------------------------------
#         # 步骤 1: 预加载 FASTA 序列到内存
#         # ---------------------------------------------------------------
#         print(f"[{split}] Pre-loading FASTA sequences into RAM (Total: {len(self.indices)})...")
#         # 使用 tqdm 显示进度条，如果不想用可以去掉 tqdm()
#         with open(self.data_dir + 'consensus.fasta', 'r') as f:
#             for idx in tqdm(self.indices, desc="Loading FASTA"):
#                 # 读取序列
#                 f.seek(seq_offsets[idx])
#                 self.cache_consensus[idx] = f.readline()[:-1]
#
#                 # 读取名称
#                 f.seek(name_offsets[idx])
#                 self.cache_seqname[idx] = f.readline().strip()[1:]
#
#         # ---------------------------------------------------------------
#         # 步骤 2: 预加载 ESM Embedding 到内存
#         # ---------------------------------------------------------------
#         self.h5_path = data_dir + "pUGTdb_Uniprot_ESM_embedding_.h5"
#         print(f"[{split}] Pre-loading ESM Embeddings into RAM...")
#
#         with h5py.File(self.h5_path, 'r') as f:
#             emb_group = f['emb']
#             loaded_count = 0
#
#             # 我们只加载当前 split 需要的 embedding
#             for idx in tqdm(self.indices, desc="Loading ESM H5"):
#                 seq_name = self.cache_seqname[idx]
#
#                 if seq_name in emb_group:
#                     # [:] 是关键，它强制将数据从 H5 对象复制为内存中的 Numpy 数组
#                     self.cache_esm[seq_name] = emb_group[seq_name][:]
#                     loaded_count += 1
#                 else:
#                     # 记录缺失情况，稍后补零
#                     pass
#
#         print(f"[{split}] Successfully loaded {loaded_count} embeddings.")
#
#         # ---------------------------------------------------------------
#         # 步骤 3: 预加载 Reaction Embedding (如果启用)
#         # ---------------------------------------------------------------
#         if arg.reaction != 'None':
#             print(f"[{split}] Processing Reaction features (Mode: {arg.reaction})...")
#
#             # 3.1 加载 CSV 映射表 -> 字典
#             df = pd.read_csv(data_dir + "Level2_expertments_SMILES_ID.csv")
#             self.index_map = df.set_index('seq_id').to_dict('index')
#
#             # 3.2 加载 Reaction H5 到内存
#             reaction_h5_path = data_dir + f"smiles_{arg.reaction}_embedding.h5"
#             print(f"[{split}] Pre-loading Reaction H5 into RAM...")
#
#             with h5py.File(reaction_h5_path, 'r') as f:
#                 # 遍历所有 key 加载到字典
#                 for k in f.keys():
#                     self.cache_reaction[str(k)] = f[k][:]
#
#             self.use_reaction = True
#         else:
#             self.use_reaction = False
#             self.index_map = None
#
#     def __len__(self):
#         return len(self.indices)
#
#     def __getitem__(self, idx):
#         # 获取真实的 dataset index
#         real_idx = self.indices[idx]
#
#         # --- 极速读取: 纯内存字典查询 (O(1) 复杂度) ---
#         consensus = self.cache_consensus[real_idx]
#         seq_name = self.cache_seqname[real_idx]
#
#         # 获取 ESM Embedding
#         if seq_name in self.cache_esm:
#             esm_emb = self.cache_esm[seq_name]
#         else:
#             # 兜底策略：如果 H5 里确实没有这个蛋白，返回全 0 向量
#             # 假设维度是 1280 (ESM-2 650M)，根据实际情况调整
#             esm_emb = np.zeros(1280, dtype=np.float32)
#
#             # 处理 Reaction (如果启用)
#         if self.use_reaction:
#             if seq_name in self.index_map:
#                 row_data = self.index_map[seq_name]
#
#                 # 从内存 cache 获取，极快
#                 # 使用 .get() 并提供默认值 None，方便后续处理
#                 ac_feats = self.cache_reaction.get(str(row_data['acceptor_id']))
#                 do_feats = self.cache_reaction.get(str(row_data['donor_id']))
#                 get_feats = self.cache_reaction.get(str(row_data['get_loop_id']))
#                 lo_feats = self.cache_reaction.get(str(row_data['lost_loop_id']))
#                 H_feats = self.cache_reaction.get(str(row_data['H_id']))
#
#                 # 维度检查与补全 (防止某个 ID 缺失导致报错)
#                 # 假设 reaction feature 维度是 2048
#                 feat_dim = 2048
#                 # 尝试从存在的特征中推断维度
#                 if ac_feats is not None: feat_dim = ac_feats.shape[0]
#
#                 if ac_feats is None: ac_feats = np.zeros(feat_dim, dtype=np.float32)
#                 if do_feats is None: do_feats = np.zeros(feat_dim, dtype=np.float32)
#                 if get_feats is None: get_feats = np.zeros(feat_dim, dtype=np.float32)
#                 if lo_feats is None: lo_feats = np.zeros(feat_dim, dtype=np.float32)
#                 if H_feats is None: H_feats = np.zeros(feat_dim, dtype=np.float32)
#
#                 # 堆叠
#                 reaction_feats = np.vstack([ac_feats, do_feats, get_feats, lo_feats, H_feats])
#
#                 return (consensus, esm_emb, seq_name, reaction_feats)
#             else:
#                 # 没找到对应的 reaction 数据
#                 pass
#
#         return (consensus, esm_emb, seq_name)


#
# class UniRefDataset(Dataset):
#     """
#     Dataset that pulls from UniRef/Uniclust downloads.
#
#     The data folder should contain the following:
#     - 'consensus.fasta': consensus sequences, no line breaks in sequences
#     - 'splits.json': a dict with keys 'train', 'valid', and 'test' mapping to lists of indices
#     - 'lengths_and_offsets.npz': byte offsets for the 'consensus.fasta' and sequence lengths
#     """
#
#     def __init__(self, data_dir: str, split: str, structure=False, pdb=False, coords=False, bins=False,
#                  p_drop=0.0, max_len=2048):
#         self.data_dir = data_dir
#         self.split = split
#         self.structure = structure
#         self.coords = coords
#         with open(data_dir + 'splits.json', 'r') as f:
#             self.indices = json.load(f)[self.split]
#         metadata = np.load(self.data_dir + 'lengths_and_offsets.npz')
#         self.offsets = metadata['seq_offsets']
#         self.namesets = metadata['name_offsets']
#         with open(data_dir + "reaction_feats.pkl", "rb") as f:
#             self.feats_all = pickle.load(f)
#         self.pdb = pdb
#         self.bins = bins
#         if self.pdb or self.bins:
#             self.n_digits = 6
#         else:
#             self.n_digits = 8
#         if self.coords:
#             with open(data_dir + 'coords.pkl', 'rb') as f:
#                 self.structures = pkl.load(f)
#         self.p_drop = p_drop
#         self.max_len = max_len
#
#     def __len__(self):
#         return len(self.indices)
#
#     def __getitem__(self, idx):
#         idx = self.indices[idx]
#         offset = self.offsets[idx]
#         nameset = self.namesets[idx]
#         with open(self.data_dir + 'consensus.fasta') as f:
#             f.seek(offset)
#             consensus = f.readline()[:-1]
#             f.seek(nameset)
#             reaction_name = f.readline()[1:-1]
#             reaction_feats = self.feats_all[reaction_name]
#         if len(consensus) - self.max_len > 0:
#             start = np.random.choice(len(consensus) - self.max_len)
#             stop = start + self.max_len
#         else:
#             start = 0
#             stop = len(consensus)
#         if self.coords:
#             coords = self.structures[str(idx)]
#             dist, omega, theta, phi = process_coords(coords)
#             dist = torch.tensor(dist).float()
#             omega = torch.tensor(omega).float()
#             theta = torch.tensor(theta).float()
#             phi = torch.tensor(phi).float()
#         elif self.structure:
#             sname = 'structures/{num:{fill}{width}}.npz'.format(num=idx, fill='0', width=self.n_digits)
#             fname = self.data_dir + sname
#             if path.isfile(fname):
#                 structure = np.load(fname)
#             else:
#                 structure = None
#             if structure is not None:
#                 if np.random.random() < self.p_drop:
#                     structure = None
#                 elif self.pdb:
#                     dist = torch.tensor(structure['dist']).float()
#                     omega = torch.tensor(structure['omega']).float()
#                     theta = torch.tensor(structure['theta']).float()
#                     phi = torch.tensor(structure['phi']).float()
#                     if self.bins:
#                         dist, omega, theta, phi = trr_bin(dist, omega, theta, phi)
#                 else:
#                     dist, omega, theta, phi = bins_to_vals(data=structure)
#             if structure is None:
#                 dist, omega, theta, phi = bins_to_vals(L=len(consensus))
#         if self.structure or self.coords:
#             consensus = consensus[start:stop]
#             dist = dist[start:stop, start:stop]
#             omega = omega[start:stop, start:stop]
#             theta = theta[start:stop, start:stop]
#             phi = phi[start:stop, start:stop]
#             return consensus, dist, omega, theta, phi
#         consensus = consensus[start:stop]
#         return (consensus,reaction_feats)


#
# class UniRefDataset(Dataset):
#     """
#     Dataset that pulls from UniRef/Uniclust downloads.
#
#     The data folder should contain the following:
#     - 'consensus.fasta': consensus sequences, no line breaks in sequences
#     - 'splits.json': a dict with keys 'train', 'valid', and 'test' mapping to lists of indices
#     - 'lengths_and_offsets.npz': byte offsets for the 'consensus.fasta' and sequence lengths
#     """
#
#     def __init__(self, data_dir: str, split: str, structure=False, pdb=False, coords=False, bins=False,
#                  p_drop=0.0, max_len=2048):
#         self.data_dir = data_dir
#         self.split = split
#         self.structure = structure
#         self.coords = coords
#         with open(data_dir + 'splits.json', 'r') as f:
#             self.indices = json.load(f)[self.split]
#         metadata = np.load(self.data_dir + 'lengths_and_offsets.npz')
#         self.offsets = metadata['seq_offsets']
#
#         self.pdb = pdb
#         self.bins = bins
#         if self.pdb or self.bins:
#             self.n_digits = 6
#         else:
#             self.n_digits = 8
#         if self.coords:
#             with open(data_dir + 'coords.pkl', 'rb') as f:
#                 self.structures = pkl.load(f)
#         self.p_drop = p_drop
#         self.max_len = max_len
#
#     def __len__(self):
#         return len(self.indices)
#
#     def __getitem__(self, idx):
#         idx = self.indices[idx]
#         offset = self.offsets[idx]
#         with open(self.data_dir + 'consensus.fasta') as f:
#             f.seek(offset)
#             consensus = f.readline()[:-1]
#
#         if len(consensus) - self.max_len > 0:
#             start = np.random.choice(len(consensus) - self.max_len)
#             stop = start + self.max_len
#         else:
#             start = 0
#             stop = len(consensus)
#         if self.coords:
#             coords = self.structures[str(idx)]
#             dist, omega, theta, phi = process_coords(coords)
#             dist = torch.tensor(dist).float()
#             omega = torch.tensor(omega).float()
#             theta = torch.tensor(theta).float()
#             phi = torch.tensor(phi).float()
#         elif self.structure:
#             sname = 'structures/{num:{fill}{width}}.npz'.format(num=idx, fill='0', width=self.n_digits)
#             fname = self.data_dir + sname
#             if path.isfile(fname):
#                 structure = np.load(fname)
#             else:
#                 structure = None
#             if structure is not None:
#                 if np.random.random() < self.p_drop:
#                     structure = None
#                 elif self.pdb:
#                     dist = torch.tensor(structure['dist']).float()
#                     omega = torch.tensor(structure['omega']).float()
#                     theta = torch.tensor(structure['theta']).float()
#                     phi = torch.tensor(structure['phi']).float()
#                     if self.bins:
#                         dist, omega, theta, phi = trr_bin(dist, omega, theta, phi)
#                 else:
#                     dist, omega, theta, phi = bins_to_vals(data=structure)
#             if structure is None:
#                 dist, omega, theta, phi = bins_to_vals(L=len(consensus))
#         if self.structure or self.coords:
#             consensus = consensus[start:stop]
#             dist = dist[start:stop, start:stop]
#             omega = omega[start:stop, start:stop]
#             theta = theta[start:stop, start:stop]
#             phi = phi[start:stop, start:stop]
#             return consensus, dist, omega, theta, phi
#         consensus = consensus[start:stop]
#         return (consensus,)



