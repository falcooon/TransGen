import argparse
import json
import os
from datetime import datetime, timedelta
import pathlib

import numpy as np
import torch
import torch.multiprocessing as mp
from pygments.lexer import default
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.cuda.amp import GradScaler

from evodiff.model import ByteNetLMTime
from evodiff.utils import Tokenizer
from torch.utils.data import Subset
from sequence_models.samplers import SortishSampler, ApproxBatchSampler
from sequence_models.datasets import UniportpUGTdbDataset
from sequence_models.constants import MSA_ALPHABET
from evodiff.collaters import OAMaskCollater
from evodiff.losses import OAMaskedCrossEntropyLoss, MSEPaddingLoss
from sequence_models.metrics import MaskedAccuracy
from sequence_models.utils import warmup 
import sys
from torch.utils.tensorboard import SummaryWriter

import wandb


sys.setrecursionlimit(1000) # must be as large as diffusion timesteps for Q_bar calculation

### SET RANDOM SEEDS ###
torch.cuda.empty_cache() # empty caches
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
home = str(pathlib.Path.home())


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config_fpath', type=str, default = r'./config/config640M.json')
    parser.add_argument('--config_fpath', type=str, default = r'./config/config38M.json')
    parser.add_argument('--out_fpath', type=str,  default=r'./output/')
    parser.add_argument('--data_dir', default=r'/root/Reaction_DATASETS/')  # r'/root/DATASETS/'   r'/root/Reaction_DATASETS/'

    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    # parser.add_argument('-g', '--gpus', default=1, type=int,
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('-off', '--offset', default=0, type=int,
                        help='Number of GPU devices to skip.')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--tie_weights', action='store_true')
    parser.add_argument('--task', default=None)
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--aml', action='store_true')
    parser.add_argument('--decay', action='store_true')
    parser.add_argument('--final_norm', action='store_true')

    # parser.add_argument('-sd', '--state_dict', default=None)
    parser.add_argument('-sd', '--state_dict', default=r'/workspace/zhangzh/evodiff-main/output/checkpoint649838_454.tar')
    # parser.add_argument('-sd', '--state_dict', default=r'/workspace/zhangzh/evodiff-main/output/ESM2_align_exp11_oadm_RDKitcheckpoint42952_197.tar')

    parser.add_argument('--wangb', type=bool ,default=True)   # True False
    parser.add_argument('--align_ESM2', type=bool ,default=True)   # True False
    parser.add_argument('--reaction', type=str ,default='molt5')   # None  RDKit UniMol molt5  deepchem
    parser.add_argument('--Target_condition', default='reaction') #reaction 5 dap(donor acceptor product) 3 substrate 2 acceptor 1
    parser.add_argument('--condition_insert', default='adaln')   # add cross_attn adaln soft_prompt None
    parser.add_argument('--ligand_feats_dim', type=int, default=512) #RDKit 2048, Molt5 512, deepchem 1613, Unimol 512


    parser.add_argument('--norm_first', action='store_true') # turns norm_first on in transformer model
    parser.add_argument('--mini_run', action='store_true') # Set to True if running on subset of data
    parser.add_argument('--mask', type=str, default='oadm')  # Set to True if running on subset of data
    parser.add_argument('--warmup', action='store_true',default=True)  # Set to True if running on subset of data
    parser.add_argument('--checkpoint_freq', type=float, default=10)  # in minutes
    parser.add_argument('--log_freq', type=float, default=10)  # in steps
    parser.add_argument('--reweighting_term', type=float, default=0)  # lambda reweighting term from Austin D3PM
    parser.add_argument('--random_seed', type=int, default=0)  # lambda reweighting term from Austin D3PM
    parser.add_argument('--pretrained', action='store_true',default=False) # ONLY USE THIS FLAG FOR FIRST RUN OF PRETRAIN
    parser.add_argument('--master_port', type=str, default='8877')

    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    if args.aml:
        pass
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = args.master_port
    if args.Target_condition == 'reaction':
        args.insert_emb_dim = 5
    elif args.Target_condition == 'dap':
        args.insert_emb_dim = 3
    elif args.Target_condition == 'substrate':
        args.insert_emb_dim = 2
    elif args.Target_condition == 'acceptor':
        args.insert_emb_dim = 1

    #print(args.world_size, args.gpus, args.nodes)
    # mp.spawn(train, nprocs=args.gpus, args=(args,))
    train(0, args)

def train(gpu, args):
    torch.random.manual_seed(args.random_seed)

    np.random.seed(int(args.random_seed))
    rank = args.nr * args.gpus + gpu
    print("nr", args.nr, "gpus", args.gpus, "gpu", gpu, "rank", rank)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank)
    torch.cuda.set_device(gpu + args.offset)
    device = torch.device('cuda:' + str(gpu + args.offset))
    with open(args.config_fpath, 'r') as f:
        config = json.load(f)
    n_tokens = len(MSA_ALPHABET)
    d_embed = config['d_embed']
    d_model = config['d_model']
    n_layers = config['n_layers']
    kernel_size = config['kernel_size']
    r = config['r']
    if 'slim' in config:
        slim = config['slim']
    else:
        slim = True
    if 'activation' in config:
        activation = config['activation']
    else:
        activation = 'relu'
    if 'accumulate' in config:
        iters_to_accumulate = config['accumulate']
    else:
        iters_to_accumulate = 1 # dont accumulate
    bucket_size = config['bucket_size']
    max_tokens = config['max_tokens']
    max_batch_size = config['max_batch_size']
    epochs = config['epochs']
    lr = config['lr']
    warmup_steps = config['warmup_steps']
    if 'rank' in config:
        weight_rank = config['rank']
    else:
        weight_rank = None
    if args.task is not None:
        config['task'] = args.task
    if args.dataset is not None:
        config['dataset'] = args.dataset

    ptjob = False
    data_dir = args.data_dir

    # ----------------------------------------------------------
    ### COLLATORS ###
    # ----------------------------------------------------------
    tokenizer = Tokenizer()
    collater = OAMaskCollater(tokenizer=tokenizer)
    diffusion_timesteps = None # Not input to model

    if rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(args.out_fpath, 'runs'))

    is_main = (rank == 0)

    if is_main and args.wangb:
        # 建议用 out_fpath 做 group，方便多次实验归类
        run_name = os.path.basename(os.path.normpath(args.out_fpath))  # e.g. RDKit_fintuning
        wandb.init(
            project="evodiff-ugt",  # 你自己项目名
            name=run_name,  # 本次 run 名（也可加时间戳）
            dir=args.out_fpath,  # wandb 本地缓存目录
            # id='run-20251224_084007-ohdu7b3m',  # 复用旧 id
            config={
                "config_fpath": args.config_fpath,
                "out_fpath": args.out_fpath,
                "reaction": args.reaction,
                "Target_condition": args.Target_condition,
                "condition_insert": args.condition_insert,
                "ligand_feats_dim": args.ligand_feats_dim,
                "mask": args.mask,
                "dropout": args.dropout,
                "weight_decay": args.weight_decay,
                "lr": lr,
                "epochs": epochs,
                "kernel_size": kernel_size,
                "n_layers": n_layers,
                "d_model": d_model,
                "d_embed": d_embed,
                "bucket_size": bucket_size,
                "max_tokens": max_tokens,
                "max_batch_size": max_batch_size,
                "warmup_steps": warmup_steps,
                "gpus": args.gpus,
                "world_size": args.world_size,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            },
        )

    causal = False
    # ----------------------------------------------------------
    ### DATALOADER ###
    # ----------------------------------------------------------
    metadata = np.load(data_dir + 'lengths_and_offsets.npz')
    ds_train = UniportpUGTdbDataset(data_dir,args, 'train')
    train_idx = ds_train.indices

    len_train = metadata['ells'][train_idx]
    train_sortish_sampler = SortishSampler(len_train, bucket_size, num_replicas=args.world_size, rank=rank)
    train_sampler = ApproxBatchSampler(train_sortish_sampler, max_tokens, max_batch_size, len_train)
    dl_train = DataLoader(dataset=ds_train,
                      batch_sampler=train_sampler,
                      num_workers=4,
                      collate_fn=collater,
                      pin_memory=True,  # ### 修改 3: 加速 CPU 到 GPU 的内存传输
                      prefetch_factor=2,  # ### 修改 4: 提前加载 2 个 batch
                      persistent_workers=True  # #
                      )
    # if rank == 0:
    ds_valid = UniportpUGTdbDataset(data_dir,args, 'valid')
    valid_idx = ds_valid.indices
    len_valid = metadata['ells'][valid_idx]
    valid_sortish_sampler = SortishSampler(len_valid, 1000, num_replicas=1, rank=0)
    valid_sampler = ApproxBatchSampler(valid_sortish_sampler, max_tokens // 2, max_batch_size, len_valid)
    dl_valid = DataLoader(dataset=ds_valid,
                      batch_sampler=valid_sampler,
                      num_workers=4,
                      collate_fn=collater,
                      pin_memory=True,  # ### 修改 3: 加速 CPU 到 GPU 的内存传输
                      prefetch_factor=2,  # ### 修改 4: 提前加载 2 个 batch
                      persistent_workers=True  # #
                     )
    # ----------------------------------------------------------
    # Initiate model
    # ----------------------------------------------------------
    padding_idx = tokenizer.pad_id  # PROTEIN_ALPHABET.index(PAD)
    masking_idx = tokenizer.mask_id
    print('Using {} as padding index'.format(padding_idx))
    print('Using {} as masking index'.format(masking_idx))

    model = ByteNetLMTime(n_tokens, d_embed, d_model, n_layers, kernel_size, r,
                      causal=causal, padding_idx=masking_idx, rank=weight_rank, dropout=args.dropout,
                      tie_weights=args.tie_weights, final_ln=args.final_norm, slim=slim, activation=activation,
                      timesteps=diffusion_timesteps,args= args)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    # if args.reaction != 'None':
    #     # 1. 冻结所有参数
    #     for param in model.parameters():
    #         param.requires_grad = False
    #
    #     # 2. 只解冻反应相关的参数
    #     for name, param in model.named_parameters():
    #         # 只要名字里带 reaction_encoder 或 react_gate 就解冻
    #         if 'reaction_encoder' in name or 'react_gate' in name:
    #             param.requires_grad = True
    #             print(f"Unfrozen (Training): {name}")
    #
    #     # 3. 定义优化器 (学习率可以用正常的 1e-4 或 5e-4)
    #     # 因为只训练这一小部分参数，不容易过拟合，可以使用稍微大一点的学习率让它快速收敛
    #     optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)



    if args.state_dict is not None:
        print('Loading weights from ' + args.state_dict + '...')
        sd = torch.load(args.state_dict, map_location=torch.device('cpu'))
        msd = sd['model_state_dict']
        msd = {k.split('module.')[1]: v for k,v in msd.items()}
        model.load_state_dict(msd, strict=False)
        if args.reaction == 'None' and 1==2:
        # if args.reaction == 'None':
            optimizer.load_state_dict(sd['optimizer_state_dict'])
            initial_epoch = sd['epoch'] + 1
            total_steps = sd['step']
            total_tokens = sd['tokens']
        else:
            initial_epoch = 0
            total_steps = 0
            total_tokens = 0
    else:
        initial_epoch = 0
        total_steps = 0
        total_tokens = 0
    model = model.to(device)
    scaler = torch.GradScaler()
    model = DDP(model)
    # ----------------------------------------------------------
    # Loss Function
    # ----------------------------------------------------------
    if args.warmup:
        # scheduler = LambdaLR(optimizer, warmup(warmup_steps), verbose=False)  #特定torch出错
        scheduler = LambdaLR(optimizer, warmup(warmup_steps))
    else:
        raise Exception("add --warmup flag to runtime")
    loss_func = OAMaskedCrossEntropyLoss(reweight=True)

    accu_func = MaskedAccuracy()
    # ----------------------------------------------------------
    # Run
    # ----------------------------------------------------------

    def epoch(model, train,current_epoch = 0, current_step=0, current_tokens=0):
        start_time = datetime.now()
        if train:
            model = model.train()
            loader = dl_train
            t = 'Training:'
        else:
            model = model.eval()
            loader = dl_valid
            t = 'Validating:'
        losses = []
        nll_losses = []
        accus = []
        ns = []
        num_seqs = []
        chunk_time = datetime.now()
        n_seen = 0
        tokens_trained = current_tokens
        if train:
            n_total = len(ds_train)
        else:
            n_total = len(ds_valid)
        for i, batch in enumerate(loader):
            # restarting from a checkpoint
            if train and i == 1 and e == initial_epoch and args.state_dict is not None and not args.pretrained:
            # if train and i == 1 and e == initial_epoch and args.state_dict is not None :
                print("Restarting from checkpoint")
                # optimizer.load_state_dict(sd['optimizer_state_dict'])
                # scheduler.load_state_dict(sd['scheduler_state_dict'])
            if args.align_ESM2:
                new_loss, new_nll_loss,esm_loss, new_accu, new_n, new_seqs, new_processed = step(model, batch, train)
            else:
                new_loss, new_nll_loss, new_accu, new_n, new_seqs, new_processed = step(model, batch, train)
            if train:
                dist.reduce(new_loss, 0, op=dist.ReduceOp.SUM)
                dist.reduce(new_nll_loss, 0, op=dist.ReduceOp.SUM)
                dist.reduce(new_accu, 0, op=dist.ReduceOp.SUM)
                dist.reduce(new_n, 0, op=dist.ReduceOp.SUM)
                dist.reduce(new_seqs, 0, op=dist.ReduceOp.SUM)
            losses.append(new_loss.item())
            nll_losses.append(new_nll_loss.item())
            accus.append(new_accu.item())
            ns.append(new_n.item())
            num_seqs.append(new_seqs.item())
            n_seen += new_seqs.item()
            total_n = sum(ns)
            r_loss = sum(losses) / total_n
            r_nll_loss = sum(nll_losses) / total_n
            raccu = sum(accus) / total_n
            if hasattr(model.module.embedder, 'react_gate'):
                gate_value = model.module.embedder.react_gate.item()
            if train:
                nsteps = current_step + i + 1
                tokens_trained += new_processed.item()
            else:
                nsteps = i
            if rank == 0:
                if ptjob:
                    end = '\n'
                    start = ''
                else:
                    start = ''
                    end = '\n'
                try:
                    print(
                        start + '%s Epoch %d of %d Step %d ntokens %d Example %d of %d loss = %.4f nll loss = %.4f accu = %.4f gate_value = %.4f'
                        % (t, e + 1, epochs, nsteps, tokens_trained, n_seen, n_total, r_loss, r_nll_loss, raccu,
                           gate_value),
                        end=end)
                except:
                    print(
                        start + '%s Epoch %d of %d Step %d ntokens %d Example %d of %d loss = %.4f nll loss = %.4f accu = %.4f'
                        % (t, e + 1, epochs, nsteps, tokens_trained, n_seen, n_total, r_loss, r_nll_loss, raccu),
                        end=end)
            if train:
                losses = losses[-999:]
                accus = accus[-999:]
                ns = ns[-999:]
                num_seqs = num_seqs[-999:]
                nll_losses = nll_losses[-999:]
                if nsteps % args.log_freq == 0:  # write to checkpoint frequency
                    if rank == 0:
                        with open(args.out_fpath + 'train-metrics.csv', 'a') as f:
                            f.write(','.join(
                                [str(r_loss), str(r_nll_loss), str(raccu), str(int(current_tokens)), str(nsteps),
                                 str(e)]))
                            f.write('\n')
                        writer.add_scalar('Train/Loss', r_loss, nsteps)
                        writer.add_scalar('Train/NLL_Loss', r_nll_loss, nsteps)
                        writer.add_scalar('Train/Accuracy', raccu, nsteps)
                        writer.add_scalar('Train/Tokens', int(current_tokens), nsteps)
                        # writer.add_scalar('Gate/Value', gate_value, nsteps)

                        if is_main:
                            wandb.log({
                                "train/loss": r_loss,
                                "train/nll_loss": r_nll_loss,
                                "train/acc": raccu,
                                "train/tokens": int(tokens_trained),
                                "epoch": e + 1,
                            }, step=nsteps)
                            try:
                                wandb.log({
                                    "train/gate_value": float(gate_value)
                                }, step=nsteps)
                            except:
                                pass

                            # 可选：记录当前学习率（很有用）
                            wandb.log({
                                "train/lr": optimizer.param_groups[0]["lr"]
                            }, step=nsteps)



                # if ((datetime.now() - chunk_time) > timedelta(minutes=args.checkpoint_freq)) or (n_seen == n_total):
                if n_seen == n_total:
                    if rank == 0:
                        print('Writing to checkpoint at', chunk_time)
                        with torch.no_grad():
                            if rank == 0:
                                ckpt_fpath = f"{args.out_fpath}checkpoint{nsteps}_{current_epoch}.tar"
                                torch.save({
                                    'step': nsteps,
                                    'tokens': tokens_trained,
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'scheduler_state_dict': scheduler.state_dict(),
                                    'epoch': e
                                }, ckpt_fpath)

                                # if is_main:     #不需要使用wandb进行模型保存
                                #     wandb.save(ckpt_fpath)

                                _ = epoch(model, False, current_step=nsteps, current_tokens=tokens_trained)
                        chunk_time = datetime.now()
        if not train and rank == 0:
            with open(args.out_fpath + 'valid-metrics.csv', 'a') as f:
                f.write(','.join(
                    [str(r_loss), str(r_nll_loss), str(raccu), str(int(current_tokens)), str(current_step),
                     str(e)]))
                f.write('\n')
            # 添加 TensorBoard 验证日志
            writer.add_scalar('Valid/Loss', r_loss, current_step)
            writer.add_scalar('Valid/NLL_Loss', r_nll_loss, current_step)
            writer.add_scalar('Valid/Accuracy', raccu, current_step)

            if is_main:
                wandb.log({
                    "valid/loss": r_loss,
                    "valid/nll_loss": r_nll_loss,
                    "valid/acc": raccu,
                    "epoch": e + 1,
                }, step=current_step)

            print('Validation complete in ' + str(datetime.now() - start_time))

        elif rank == 0:
            print('Epoch complete in ' + str(datetime.now() - start_time))
        if rank == 0:
            writer.close()
        return i, tokens_trained




    def step(model, batch, train):

        react_emb = None
        if len(batch) == 5:
            src, timestep, tgt, mask, src_emb = batch
        elif len(batch) == 6:
            src, timestep, tgt, mask, src_emb, react_emb = batch
        mask = mask.to(device)

        timestep = timestep.to(device)
        src = src.to(device)
        tgt = tgt.to(device)
        input_mask = (src != padding_idx).float()
        src_emb = src_emb.to(device)
        if react_emb is not None:
            react_emb = react_emb[:,:args.insert_emb_dim,:]
            react_emb = react_emb.to(device)

        n_tokens = mask.sum()

        n_processed = input_mask.sum()
        n_seqs = torch.tensor(len(src), device=device)
        # step through model
        if train:
            optimizer.zero_grad() # reset gradients of model parameters

        # Enables autocasting for the forward pass (model + loss)
        with torch.amp.autocast(device_type='cuda'):
            if args.align_ESM2:
                outputs,model_feats = model(src, src_emb, react_emb = react_emb,input_mask=input_mask.unsqueeze(-1))
            else:
                outputs = model(src, src_emb, react_emb = react_emb,input_mask=input_mask.unsqueeze(-1))
            # (self, x, src_emb, react_emb=None, input_mask=None, train_test = None)


            ce_loss, nll_loss = loss_func(outputs, tgt, mask, timestep, input_mask)  # sum(loss per token)
            loss = ce_loss
            if args.align_ESM2:
                esm2_loss = MSEPaddingLoss(model_feats, src_emb, input_mask)
                loss = loss + esm2_loss
            accu = accu_func(outputs, tgt, mask) * n_tokens
        if train:
            # Exit the context manager before backward()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scale = scaler.get_scale()
            scaler.update()
            skip_scheduler = (scale > scaler.get_scale())
            if not skip_scheduler:
                scheduler.step()

        if loss <= 0 or loss >= 1000000:
            print(loss)
            print(timestep)
            print([tokenizer.untokenize(t) for t in tgt])
            print([tokenizer.untokenize(s) for s in src])
            # import pdb; pdb.set_trace()
        #print("lvb", lvb_loss, "ce", ce_loss, "loss", loss, "tokens", n_tokens, "timestep", timestep)
        if args.align_ESM2:
            return loss, nll_loss, esm2_loss, accu, n_tokens, n_seqs, n_processed
        return loss, nll_loss, accu, n_tokens, n_seqs, n_processed

    n_parameters = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print('%d model parameters' %n_parameters)
        print('%d training sequences' %len(len_train))
        print('%d validation sequences' %len(len_valid))
    for e in range(initial_epoch, epochs):
        # if not args.mini_run:
        train_sortish_sampler.set_epoch(e + 1)

        s, t = epoch(model, True,current_epoch =  e, current_step=total_steps, current_tokens=total_tokens)
        total_steps += s
        total_tokens += t

if __name__ == '__main__':
    main()

