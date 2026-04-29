#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import h5py
from tqdm import tqdm

import torch
from torch.cuda.amp import autocast
import esm


def read_fasta(fasta_path):
    """
    读取 FASTA，返回 [(seq_id, seq_str), ...]
    - 支持多行序列
    - seq_id 取 '>' 后第一段（空格前）
    """
    data_list = []
    seq_id = None
    seq_chunks = []
    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                # flush previous
                if seq_id is not None:
                    seq = "".join(seq_chunks).replace(" ", "").replace("\t", "")
                    data_list.append((seq_id, seq))
                # new header
                header = line[1:].strip()
                seq_id = header.split()[0] if header else f"seq_{len(data_list)}"
                seq_chunks = []
            else:
                seq_chunks.append(line)
        # last
        if seq_id is not None:
            seq = "".join(seq_chunks).replace(" ", "").replace("\t", "")
            data_list.append((seq_id, seq))
    return data_list


def load_esm_model(model_name):
    """
    目前只给 650M 一个选项；你也可以扩展更多 ESM2 模型。
    """
    if model_name == "esm2_t33_650M_UR50D":
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        repr_layer = 33
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    return model, alphabet, repr_layer


def main():
    parser = argparse.ArgumentParser(description="ESM2 embedding to H5 from FASTA (length-sorted, AMP).")
    parser.add_argument("--fasta", default=r'/workspace/zhangzh/evodiff-main/output/exp4_oadm_deepchem/gent_seq/generated_samples_string.fasta', type=str, help="Input FASTA file.")
    parser.add_argument("--out_h5", default=r'/workspace/zhangzh/evodiff-main/output/exp4_oadm_deepchem/gent_seq/generated_samples_string_ESM_embedding.h5', type=str, help="Output .h5 path.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--gpu", default="0", type=str, help="CUDA device id(s), e.g. '0' or '0,1'.")
    parser.add_argument("--model", default="esm2_t33_650M_UR50D", type=str, help="ESM model name.")
    parser.add_argument("--fp16", action="store_true", help="Enable AMP fp16 autocast.")
    parser.add_argument("--sort_by_len", action="store_true", help="Sort sequences by length to reduce padding.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing h5.")
    args = parser.parse_args()

    # ---- CUDA / device ----
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is not available. Please check your GPU / drivers / torch install.")

    # ---- Load model ----
    print("Loading ESM model...")
    model, alphabet, repr_layer = load_esm_model(args.model)
    batch_converter = alphabet.get_batch_converter()
    model.eval().to(device)

    # ---- Read FASTA ----
    print(f"Reading FASTA: {args.fasta}")
    data_list = read_fasta(args.fasta)
    if len(data_list) == 0:
        raise ValueError("No sequences found in FASTA.")
    print(f"Total sequences: {len(data_list)}")

    # ---- Sort by length (optional) ----
    if args.sort_by_len:
        print("Sorting sequences by length to reduce padding...")
        data_list.sort(key=lambda x: len(x[1]), reverse=False)

    # ---- Prepare output ----
    if os.path.exists(args.out_h5):
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {args.out_h5}. Use --overwrite to overwrite.")
        os.remove(args.out_h5)

    print(f"Start inference (batch_size={args.batch_size}, fp16={args.fp16})")
    with h5py.File(args.out_h5, "w") as f:
        grp = f.create_group("emb")

        for i in tqdm(range(0, len(data_list), args.batch_size), mininterval=1.0):
            batch_data = data_list[i : i + args.batch_size]
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)

            # valid lengths on CPU
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

            batch_tokens = batch_tokens.to(device)

            with torch.no_grad():
                if args.fp16:
                    with autocast():
                        results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
                else:
                    results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)

                token_representations = results["representations"][repr_layer]

            # save each seq
            for j, tokens_len in enumerate(batch_lens):
                seq_id = str(batch_labels[j])
                length = int(tokens_len.item())

                # strip <cls> and <eos>
                # token_representations: [B, T, C]
                rep = token_representations[j, 1 : length - 1]

                # move to cpu, store fp16
                rep_np = rep.detach().cpu().numpy().astype(np.float16)

                # avoid duplicate ids
                if seq_id in grp:
                    # 你也可以选择在这里追加后缀，或者覆盖
                    continue
                grp.create_dataset(seq_id, data=rep_np, compression="gzip", compression_opts=4)

            # free references
            del results, token_representations, batch_tokens

    print(f"Done. Saved to: {args.out_h5}")


if __name__ == "__main__":
    main()
