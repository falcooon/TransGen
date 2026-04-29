import os
import sys
import json
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
# from openbabel import openbabel
from tqdm import tqdm


def generate_skid_style_pdb(smiles, output_filename="ligand.pdb"):
    """
    结合 SKiD GitHub 仓库逻辑：
    1. 使用 RDKit 进行 3D 构象生成和 MMFF94 能量优化。
    2. (可选) 使用 OpenBabel 进行最终格式标准化（如果需要）。
    """
    try:
        # --- 第一阶段：使用 RDKit 生成高质量 3D 结构 ---

        # 1. 解析 SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # 2. 添加显式氢 (论文核心步骤)
        mol = Chem.AddHs(mol)

        # 3. 生成 3D 坐标 (Embedding)
        # ETKDG 是目前最标准的生成算法
        params = AllChem.ETKDG()
        params.randomSeed = 0xf00d  # 设置随机种子以保证结果可复现
        embed_res = AllChem.EmbedMolecule(mol, params)

        if embed_res == -1:
            # 如果失败，尝试随机坐标
            AllChem.EmbedMolecule(mol, useRandomCoords=True)

        # 4. MMFF94 力场能量最小化 (论文核心步骤)
        # SKiD 使用此步骤确保分子构象符合物理化学规律
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, mmffVariant='MMFF94')
        else:
            # 如果缺少 MMFF94 参数，回退到 UFF 力场 (常见处理逻辑)
            AllChem.UFFOptimizeMolecule(mol)

        # --- 第二阶段：输出 PDB ---

        # 方法 A: 直接使用 RDKit 输出 (最简单)
        Chem.MolToPDBFile(mol, output_filename)

        # 方法 B: (高级) 如果 RDKit 输出的 PDB 在后续对接软件(GNINA)中兼容性不佳，
        # SKiD 可能会调用 OpenBabel 进行转换。以下演示如何用 OpenBabel 转换：
        # (通常在命令行中运行: obabel -isdf input.sdf -opdb -O output.pdb)

        print(f"成功生成文件: {output_filename}")

    except Exception as e:
        print(f"Error processing {smiles}: {e}")


react_df = pd.read_csv(r'/root/Reaction_DATASETS/unique_reaction.csv')
predicted_structure_path = r'/workspace/zhangzh/GENzyme/generated/all_protein'
parent_folder = os.path.dirname(predicted_structure_path)
foldseek_path = os.path.join(parent_folder, 'foldseek')
gnina_path = os.path.join(parent_folder, 'gnina')
if not os.path.exists(gnina_path):
    os.makedirs(gnina_path)
for sub_foldseek in tqdm(os.listdir(foldseek_path)):
    sub_foldseek_path = os.path.join(foldseek_path, sub_foldseek)
    sdf_list = os.listdir(sub_foldseek_path)
    if 'ref_ligand.sdf' in sdf_list:
        ref_sdf_path = os.path.join(sub_foldseek_path, 'ref_ligand.sdf')
        print(ref_sdf_path)
        pdb_path = os.path.join(predicted_structure_path, sub_foldseek)
        react_name = "['" + sub_foldseek[:10].replace("_", ":") + "']"
        substrate_smile = react_df[react_df['reaction_rhea_ids'] == react_name]['acceptor'].values[0]
        substrate_smile_save_path = os.path.join(parent_folder, 'temp_substrate_pdb.pdb')
        generate_skid_style_pdb(substrate_smile, substrate_smile_save_path)
        gnina_path_save = os.path.join(gnina_path, f'{sub_foldseek}_docked.sdf')
        print("pdb_path:", pdb_path)
        print("substrate_smile_save_path:", substrate_smile_save_path)
        print("ref_sdf_path:", ref_sdf_path)
        print("gnina_path_save:", gnina_path_save)

        # !cd /workspace/zhangzh/test_docking && conda run -n gnina_env ./gnina -r "$pdb_path" -l "$substrate_smile_save_path" --autobox_ligand "$ref_sdf_path" -o "$gnina_path_save" --seed 0
        # !conda run -n gnina_env /workspace/zhangzh/gnina -r "{pdb_path}" -l "{substrate_smile_save_path}" --autobox_ligand "{ref_sdf_path}" -o "{gnina_path_save}" --seed 0

        # !cd /workspace/zhangzh/test_docking && conda run -n gnina_env /workspace/zhangzh/gnina -r "{pdb_path}" -l "{substrate_smile_save_path}" --autobox_ligand "{ref_sdf_path}" -o "{gnina_path_save}" --seed 0

        # !cd conda run -n gnina_env /workspace/zhangzh/gnina -r "{pdb_path}" -l "{substrate_smile_save_path}" --autobox_ligand "{ref_sdf_path}" -o "{gnina_path_save}" --seed 0


