import json
import math
import os
from collections import Counter
import subprocess

from tqdm import tqdm

import numpy as np
import pandas as pd
import random
from Bio import SeqIO
import sys
from utils.ligand_init import *

def split(input,output):
        all_names = []
        splits = {'train': [], 'test': [], 'valid': [], 'rtest': []}
        print('Getting names...')
        with open(input) as f:
            for line in f:
                if line[0] == '>':
                    all_names.append(line[1:-1].split(' ')[0])

        used_files = []
        remian_list = []
        n = len(all_names)
        print('Splitting...')
        order = list(range(n))
        random.shuffle(order)
        #
        # splits['rtest'] = order[0:1000]
        # splits['valid'] = order[1000:2000]
        # splits['test'] = order[2000:3000]
        # splits['train'] = order[3000:]
        rest = []
        for i in tqdm(range(n)):
            if np.random.random() < 5e-3:
                splits['rtest'].append(i)
            elif np.random.random() < 2e-3:
                splits['test'].append(i)
            else:
                rest.append(i)

        splits['valid'] = rest[:10000]
        splits['train'] = rest[10000:]



        for k in splits:
            print(k + ': %d sequences' %len(splits[k]))
        with open(output, 'w') as f:
            json.dump(splits, f)




def get_offsets(input,output):
    results = {}
    results['name_offsets'] = []
    results['seq_offsets'] = []
    results['ells'] = []

    with open(input, 'r') as file:
        # 使用 sum 函数来统计行数
        total_lines = sum(1 for line in file)
    length = total_lines // 2+1

    with tqdm(total=length) as pbar:
        with open(input, 'r') as f:
            results['name_offsets'].append(f.tell())
            line = f.readline()
            while line:
                if line[0] != '>':
                    results['name_offsets'].append(f.tell())     #序列的起始位置
                    results['ells'].append(len(line[:-1]))       #序列的长度
                else:
                    results['seq_offsets'].append(f.tell())      #>头的起始位置
                    pbar.update(1)
                line = f.readline()
    results['ells'].append(len(line[:-1]))
    results['name_offsets'] = np.array(results['name_offsets'])
    results['seq_offsets'] = np.array(results['seq_offsets'])
    results['ells'] = np.array(results['ells'])
    np.savez_compressed(output, **results)




def flatten_fasta(input_p, output_p):
    with open(input_p, 'r') as file:
        # 使用 sum 函数来统计行数
        total_lines = sum(1 for line in file)
    with tqdm(total=total_lines//2 +1 ) as pbar:
        with open(input_p, 'r') as f_in, open(output_p, 'w') as f_out:
            seq = ''
            for line in f_in:
                if line[0] == '>':
                    if len(seq) > 0:
                        f_out.write(seq + '\n')
                        seq = ''
                        pbar.update(1)
                    f_out.write(line)
                else:
                    seq += line[:-1]

def flatten_fasta_excel(input_p, output_p):
    df = pd.read_excel(input_p)
    total_lines = df.shape[0]
    with tqdm(total=total_lines//2 +1 ) as pbar:
        ligand_dict = ligand_init(ligand_smiles)
        with open(output_p, 'w') as f_out:
            seq = ''
            for line in f_in:
                if line[0] == '>':
                    if len(seq) > 0:
                        f_out.write(seq + '\n')
                        seq = ''
                        pbar.update(1)
                    f_out.write(line)
                else:
                    seq += line[:-1]

#
def read_and_save_fasta(input_fasta, output_fasta, num_lines=20000):
    with open(input_fasta, 'r') as infile, open(output_fasta, 'w') as outfile:
        # 读取指定数量的行并写入新的文件
        for i, line in enumerate(infile):
            if i >= num_lines:
                break
            outfile.write(line)


from Bio import SeqIO
def remove_sequence_from_fasta(input_fasta, sequence_to_remove):
    filtered_records = []
    for record in SeqIO.parse(input_fasta, "fasta"):
        if str(record.seq) == sequence_to_remove:
            continue  # 跳过要删除的序列
        filtered_records.append(record)
    # 将保留下来的序列写入新的FASTA文件
    SeqIO.write(filtered_records, input_fasta, "fasta")


# 删除序列
def remove_duplicate(input_path,fasta_file):
    # file_list = os.listdir(input_path)
    # for i in file_list:
    #     sub_fasta = os.path.join( os.path.join(input_path,i), 'Sequence.fasta')
    #     for i,record in enumerate(SeqIO.parse(sub_fasta, "fasta")):
    #         remove_sequence_from_fasta(fasta_file, str(record.seq))

    length_seq = []
    end_hang = []
    star_seq = []
    for i, record in enumerate(SeqIO.parse(fasta_file, "fasta")):
        length_seq.append(len(str(record.seq)))
        end_hang.append(str(record.seq)[-1])
        if '*' in str(record.seq) and '*' != str(record.seq)[-1]:
            star_seq.append(str(record.seq))
    # remove_sequence_from_fasta(fasta_file)

    print("最后一个字母是什么：",dict(Counter(end_hang)))
    print("所有序列的平均长度为：", sum(length_seq)/len(length_seq))
    print("一共多少个样本：", len(length_seq))
    frequency_dict = dict(Counter(length_seq))
    import matplotlib.pyplot as plt
    # 数据字典
    # 创建图形和子图
    keys = sorted(frequency_dict.keys())
    values = [frequency_dict[key] for key in keys]
    fig, ax1 = plt.subplots(figsize=(6*2, 2.9*2))
    # 绘制直方图    fig, ax = plt.subplots(figsize=(6*2, 2.5*2))
    ax1.bar(keys, values, color='lightblue', label='Frequency')
    ax1.plot(keys, values, color='blue', marker='o', linestyle='-', label='Line Plot')
    ax1.set_xlabel('The numbers of AA in the Sequence',fontsize = 20)
    ax1.set_ylabel('Frequency',fontsize = 20)
    ax1.tick_params(axis='both', labelsize=16,color = 'Black')
    # plt.title('Sample length distribution of pUGTdb',fontsize = 30)
    ax1.legend(loc='upper right',fontsize = 20)
    ax1.text(0.05, 0.95, f'Average length:{"{:.2f}".format(sum(length_seq)/len(length_seq))}', transform=ax1.transAxes, fontsize=20,
            verticalalignment='top')
    plt.show()


    print("完成删除！！！")

    import numpy as np
    import os
import pickle

def merge_npy():
    input_folder = r"E:\pUGTdb\3D_molecule"
    data_dict = dict({})# 替换为你的文件路径
    for file_name in tqdm(os.listdir(input_folder)):
        if file_name[-1] == '@':
            continue
        file_path = os.path.join(input_folder, file_name)
        conf_path = os.path.join(file_path, "conformer_feats")
        for d3 in os.listdir(conf_path):
            mo_path = os.path.join(conf_path, d3)
            # print(mo_path)
            feat_name = f"{file_name}@{d3}"            #分子的特征是 名字@conformer_0-101
            feat_name = feat_name.split('.')[0]
            f = np.load(mo_path)
            if feat_name not in data_dict.keys():
                data_dict[feat_name] = f
    # np.save(output_file, data_dict)
    output_file = "E:\pUGTdb\molecule_finetuning\molecule_feats.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(data_dict,f)



def merge_fasta(input_folder, output_file):
    cc = r'E:\pUGTdb\3D_molecule'
    with open(input_folder, "rb") as f:
        loaded_data = pickle.load(f)
    feat_mo = loaded_data.keys()
    with open(output_file, 'w') as f_out:
        for feat in feat_mo:
            file_name = feat.split("@")[0]
            file_path = os.path.join(cc, file_name)
            file_path = os.path.join(file_path, 'Sequence.fasta')
            with open(file_path, 'r') as f_in:
                for line in f_in:
                    if line.startswith('>'):
                        new_header = line.strip()
                        temp_header = f"{new_header}@@{feat}"
                        f_out.write(temp_header + '\n')
                    else:
                        f_out.write(line)

# merge_npy()
# 示例用法
# output_file = r"E:\pUGTdb\molecule_finetuning\seq_ligand.fasta"
# input_folder = r'E:\pUGTdb\molecule_finetuning\molecule_feats.pkl'# 替换为合并后的输出文件路径
# merge_fasta(input_folder, output_file)

# # 输入文件和输出文件路径
# input_fasta = r'E:\pUGTdb\pUGTdb_completeUGTs.pep'
# output_fasta = r"E:\uniref50\uniref50_1w\uniref50_1w.fasta"
# # 读取前 10000 行并保存
# read_and_save_fasta(input_fasta, output_fasta, num_lines=20000)

# i_p = r'E:\pUGTdb\pUGTdb_completeUGTs_non_dupliate.fasta'
# o_p = r'E:\pUGTdb\3D_molecule_train\consensus.fasta'
# flatten_fasta(i_p,o_p)

i_p = r'/workspace/zhangzh/pUGTdb/molecule_finetuning/reaction_fine_tune/chebi_reactions_V4-5.xlsx'
o_p = r'/workspace/zhangzh/pUGTdb/molecule_finetuning/reaction_fine_tune/consensus.fasta'
flatten_fasta_excel(i_p,o_p)
#
#
i_p = r'E:\pUGTdb\3D_molecule_train\consensus.fasta'
o_p = r'E:\pUGTdb\3D_molecule_train\lengths_and_offsets.npz'
get_offsets(i_p,o_p)
#
#
i_p = r'E:\pUGTdb\pUGTdb_completeUGTs_non_dupliate.fasta'
o_p = r'E:\pUGTdb\3D_molecule_train\splits.json'
split(i_p,o_p)

# input_path = r'E:\pUGTdb\3D_molecule'
# fasta_file = r'E:\pUGTdb\pUGTdb_completeUGTs.fasta'
# fasta_file = r'E:\pUGTdb\pUGTdb_completeUGTs_non_dupliate.fasta'
# remove_duplicate(input_path,fasta_file)
#


# cc = r'E:\pUGTdb\molecule_finetuning\molecule_feats.pkl'
# with open(cc, "rb") as f:
#     loaded_data = pickle.load(f)
# print(loaded_data)

