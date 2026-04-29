import os
import numpy as np
import json
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import random
import torch
import torch_geometric
import torch_sparse
from torch_geometric.nn import MessagePassing

import esm

# model = esm.pretrained.esmfold_v1()
# model = model.eval().cuda()

from tqdm import tqdm
from esm.inverse_folding.util import score_sequence
device = torch.device("cuda")


model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
model = model.to(device)
model = model.eval()

root_file = r'/workspace/zhangzh/ESMfold_'
# root_file = r'/home/zhangzh/Generated_seq'
# file_list = os.listdir(root_file)
file_list = [
    # "/root/REFER_PDB",
    # "/workspace/zhangzh/ZymCTRL/OmegaFold",
    "/workspace/zhangzh/GENzyme/generated/all_protein",
    "/workspace/zhangzh/REXzyme_aa/generated/OmegaFold",
    "/workspace/zhangzh/ProCALM-main/generated/OmegaFold",
    "/workspace/zhangzh/evodiff-main/output/ESM2_align_exp23_oadm_molt5_adaln_true/gent_seq/OmegaFold"
]
for ff in file_list:
    sc_dict = dict({})
    # ff = os.path.join(ff, r'predict_informations')
    pdb_list = os.listdir(ff)
    if len(pdb_list) > 200:
        pdb_list = random.sample(pdb_list, 200)
    for pdb in tqdm(pdb_list):
        pdb_name = os.path.join(ff, pdb)
        structure = esm.inverse_folding.util.load_structure(pdb_name)
        coords, native_seq = esm.inverse_folding.util.extract_coords_from_structure(structure)
        ll_fullseq, ll_withcoord = score_sequence(model, alphabet, coords, native_seq)
        coords = torch.tensor(coords).to(device)
        sampled_seq = model.sample(coords, temperature=1e-6)
        recovery = np.mean([(a == b) for a, b in zip(native_seq, sampled_seq)])
        pdb_dict = dict({'Sequence recovery':f'{recovery:.2f}',
                         'average log-likelihood':f'{ll_fullseq:.2f}',
                         'scPerplexity':f'{np.exp(-ll_fullseq):.2f}'})
        sc_dict[pdb] = pdb_dict
    sc_save_path = os.path.join(ff, r'sc_perplexity.json')
    with open(sc_save_path, 'w') as file:
        json.dump(sc_dict, file, indent=4)
    print("保存成功！！！！！！！！！！！！！！！！！！！！！！！！")
