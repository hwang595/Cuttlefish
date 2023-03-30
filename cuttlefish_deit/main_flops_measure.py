# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import math

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset
from engine import train_one_epoch, evaluate
from losses import DistillationLoss
from samplers import RASampler
import models
from timm_models import create_model_factorized
import utils

#from ptflops import get_model_complexity_info 
# might be a better option
from fvcore.nn import FlopCountAnalysis, flop_count_table, flop_count_str


def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)

    # Model parameters
    parser.add_argument('--model-vanilla', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--model', default='lowrank_deit_small_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Cuttlefish arguments
    parser.add_argument('--fr-warmup-epochs', type=int, default=300,
                        help='full-rank warmup epochs')
    parser.add_argument('--factorized-mode', type=str, default="adaptive",
                        help='training mode: adaptive|pufferfish|')
    parser.add_argument('--rank-ratio', type=float, default=0.25,
                        help='for fixed rank ratio used in pufferfish')

    # factorized training arguments
    parser.add_argument('--factorized-lr-decay', type=float, default=3.0,
                        help='decaying the init lr when switched to lr training is usually helpful, `factorized-lr-decay` controls that.')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def layer_rank_stable_detector(epoch, layer_est_ranks, num_layers_remain):
    __predefined_widow_size = 5
    __base_thresold = 0.411
    __threshold_scaler = 2.5

    if num_layers_remain in range(5, 10):
        __grad_threshold = __base_thresold * __threshold_scaler
    elif num_layers_remain in range(5):
        __grad_threshold = __base_thresold  * (__threshold_scaler * 2)
    else:
        __grad_threshold = __base_thresold

    if epoch < __predefined_widow_size:
        return False
    else:
        smooth_est_ranks = utils.exponential_moving_average(layer_est_ranks, points=__predefined_widow_size)
        grad_smooth_est_ranks = np.absolute(np.gradient(smooth_est_ranks))
        print("############## grad_smooth_est_ranks: {}, np.mean(grad_smooth_est_ranks[-5:]): {:.3f}, layers remain: {}, threshold: {}".format(
                            grad_smooth_est_ranks,
                             np.mean(grad_smooth_est_ranks[-__predefined_widow_size:]), num_layers_remain, __grad_threshold))
        return np.mean(grad_smooth_est_ranks[-__predefined_widow_size:]) <= __grad_threshold


def decompose_vanilla_model(vanilla_model, low_rank_model, est_rank, args):
    collected_weights = []
    rank_counter = 0
    
    for p_index, (name, param) in enumerate(vanilla_model.state_dict().items()):
        if "deit" in args.model_vanilla:
            if args.factorized_mode == "adaptive":
                condition = len(param.size()) == 2 and p_index not in range(0, 4) and p_index != 150 and ".attn.proj." not in name
            elif args.factorized_mode == "pufferfish":
                condition = len(param.size()) == 2 and p_index not in range(0, 77) and p_index != 150
            else:
                raise NotImplementedError("Unsupported factorized mode ...")
        elif "resmlp" in args.model_vanilla:
            if args.factorized_mode == "adaptive":
                condition = len(param.size()) == 2 and ".mlp_channels." in name
            elif args.factorized_mode == "pufferfish":
                condition = len(param.size()) == 2 and p_index not in range(0, 214) and ".mlp_channels." in name
            else:
                raise NotImplementedError("Unsupported factorized mode ...")
        else:
            raise NotImplementedError("Unsupported arch type ...")

        if condition: # 0 blocks
        #if len(param.size()) == 2 and p_index not in range(0, 54) and p_index != 150: # 4 blocks
        #if len(param.size()) == 2 and p_index not in range(0, 64) and p_index != 150:
        #if len(param.size()) == 2 and p_index not in range(0, 75) and p_index != 150:
            rank = min(param.size()[0], param.size()[1])

            if args.factorized_mode == "adaptive":
                sliced_rank = est_rank[rank_counter]
                rank_counter += 1
            elif args.factorized_mode == "pufferfish":
                sliced_rank = int(rank * args.rank_ratio)
            else:
                raise NotImplementedError("Unsupported factorized mode ...")

            u, s, v = torch.svd(param)
            u_weight = torch.matmul(u, torch.diag(torch.sqrt(s)))
            v_weight = torch.matmul(torch.diag(torch.sqrt(s)), v.t()).t()
            u_weight_sliced, v_weight_sliced = u_weight[:, 0:sliced_rank], v_weight[:, 0:sliced_rank]
            #collected_weights.append(u_weight_sliced)
            collected_weights.append(v_weight_sliced.t())
            collected_weights.append(u_weight_sliced)
        else:
            collected_weights.append(param)
         
    reconstructed_state_dict = {}
    model_counter = 0
    for p_index, (name, param) in enumerate(low_rank_model.state_dict().items()):
        if utils.get_rank() == 0:
            print("p_index: {}, name: {}, param size: {}, collected weight size: {}".format(
                                                                                        p_index,
                                                                                        name,
                                                                                        param.size(), 
                                                                                        collected_weights[model_counter].size()))
        if "_u.bias" in name:
            reconstructed_state_dict[name] = param
        else:
            assert param.size() == collected_weights[model_counter].size()
            reconstructed_state_dict[name] = collected_weights[model_counter]
            model_counter += 1
    low_rank_model.load_state_dict(reconstructed_state_dict)



def rank_estimation(epoch, vanilla_model, adjust_rank_scale=None,
                    est_rank_tracker=None, 
                    layers_to_factorize=None,
                    layer_stable_tracker=None,
                    args=None
                    ):
    #__rr_lower_bound = 1/16
    __rr_lower_bound = 0.5
    est_rank_list = []
    ori_rank_list = []

    if epoch == -1:
        # at the very beginning, we calculate the rank adjustment ceof
        adjust_rank_scale = []
    
    num_layers_remain = len(layer_stable_tracker) - sum(layer_stable_tracker)
    for p_index, (name, param) in enumerate(vanilla_model.state_dict().items()):
        if "deit" in args.model_vanilla:
            condition = len(param.size()) == 2 and p_index not in range(0, 4) and p_index != 150
        elif "resmlp" in args.model_vanilla:
            #condition = len(param.size()) == 2 and ".mlp_channels." in name
            condition = len(param.size()) == 2 and (".mlp_channels." in name or ".linear_tokens." in name)
        else:
            raise NotImplementedError("Unsupported arch type ...")

        if condition:
            ori_rank = min(param.size()[0], param.size()[1])

            if "deit" in args.model_vanilla:
                if ".attn.proj." not in name:
                    ori_rank_list.append(ori_rank)
            elif "resmlp" in args.model_vanilla:
                if ".linear_tokens." not in name:
                    ori_rank_list.append(ori_rank)
            else:
                raise NotImplementedError("Unsupported arch type ...")

            u, s, v = torch.svd(param)
            estimated_rank = int(torch.sum(s ** 2).item() / (torch.max(s).item() ** 2))

            if name in layers_to_factorize:
                layer_index_in_factorized_layers = layers_to_factorize.index(name)
                est_rank_tracker[layer_index_in_factorized_layers].append(estimated_rank)
                if not layer_stable_tracker[layer_index_in_factorized_layers]: # only look at layers that's not stable
                    layer_stable_tracker[layer_index_in_factorized_layers] = layer_rank_stable_detector(epoch, 
                                            layer_est_ranks=est_rank_tracker[layer_index_in_factorized_layers],
                                            num_layers_remain=num_layers_remain)            

            print("#### Epoch: {}, Param index: {}, Param name: {}, Ori rank: {}, Est rank: {}".format(
                    epoch, p_index, 
                    name,
                    ori_rank, 
                    estimated_rank))

            if "deit" in args.model_vanilla:
                if ".attn.proj." not in name:
                    est_rank_list.append(estimated_rank)
            elif "resmlp" in args.model_vanilla:
                if ".linear_tokens." not in name:
                    est_rank_list.append(estimated_rank)
            else:
                raise NotImplementedError("Unsupported arch type ...")

            if epoch == -1:
                adjust_rank_scale.append(ori_rank/estimated_rank)

    if all(layer_stable_tracker):
        switch_epoch = epoch + 1
    else:
        switch_epoch = args.epochs + 1

    print("@@@ Epoch: {}, Layer stable tracker: {}, swithc epoch: {}".format(
                epoch, layer_stable_tracker, switch_epoch
        ))

    if epoch == -1:
        return est_rank_list, adjust_rank_scale, switch_epoch
    else:
        # scale the stable rank
        adjusted_rank = []
        for er, ars, ori_rank in zip(est_rank_list, adjust_rank_scale, ori_rank_list):
            adjusted_rank.append(int(ori_rank * __rr_lower_bound))
            #adjusted_rank.append(ori_rank) # for sanity check
            #if int(er * ars) > ori_rank:
            #    adjusted_rank.append(ori_rank)
            #elif int(er * ars) < int(ori_rank * __rr_lower_bound):
            #    adjusted_rank.append(int(ori_rank * __rr_lower_bound))
            #else:
            #    adjusted_rank.append(int(math.ceil(er * ars)))
        return adjusted_rank, switch_epoch



def main(args):
    utils.init_distributed_mode(args)

    print(args)

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    print(f"Creating model: {args.model}")

    model_vanilla = create_model(
        args.model_vanilla,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )


    # measuring model FLOPs
    model_vanilla.eval()
    input = torch.randn(3, 224, 224)[None,:,:,:].float()

    flop = FlopCountAnalysis(model_vanilla, input)
    
    print("@ Measuring the FLOPs of vanilla model ...")
    print(flop_count_table(flop, max_depth=4))
    print(flop_count_str(flop))
    print(flop.total())

    # with torch.cuda.device(0):
    #     print("* Measuring FLOPs of vanilla Model ...")
    #     model_vanilla_dummy = create_model(
    #         args.model_vanilla,
    #         pretrained=False,
    #         num_classes=args.nb_classes,
    #         drop_rate=args.drop,
    #         drop_path_rate=args.drop_path,
    #         drop_block_rate=None,
    #     )
    #     macs, params = get_model_complexity_info(model_vanilla_dummy, (3, 224, 224), as_strings=True,
    #                                            print_per_layer_stat=True, verbose=True)
    #     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #     print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    #     del model_vanilla_dummy

    layers_to_factorize = []
    #for block_index in range(4, 12):
    if "deit" in args.model_vanilla:
        for block_index in range(0, 12):
            for block_content in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"):
            #for block_content in ("attn.qkv", "mlp.fc1", "mlp.fc2"):                
                layers_to_factorize.append("blocks.{}.{}.weight".format(block_index, block_content))
    elif "resmlp" in args.model_vanilla:
        for block_index in range(0, 36):
            #for block_content in ("mlp_channels.fc1", "mlp_channels.fc2"):
            for block_content in ("linear_tokens", "mlp_channels.fc1", "mlp_channels.fc2"):
                layers_to_factorize.append("blocks.{}.{}.weight".format(block_index, block_content))
    else:
        raise NotImplementedError("Unsuported Arch type ...")

    est_rank_tracker = [[] for _ in range(len(layers_to_factorize))]
    layer_stable_tracker = [False for _ in range(len(layers_to_factorize))]

    if utils.get_rank() == 0:
        print("")
        print("@@@@ Vanilla Model: {}".format(model_vanilla))
        print("")

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)

    #model.to(device)
    model_vanilla.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        # model_ema = ModelEma(
        #     model,
        #     decay=args.model_ema_decay,
        #     device='cpu' if args.model_ema_force_cpu else '',
        #     resume='')

        model_vanilla_ema = ModelEma(
            model_vanilla,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    #model_without_ddp = model
    model_vanilla_without_ddp = model_vanilla

    if args.distributed:
        model_vanilla = torch.nn.parallel.DistributedDataParallel(model_vanilla, device_ids=[args.gpu])
        model_vanilla_without_ddp = model_vanilla.module

    n_parameters_vanilla = sum(p.numel() for p in model_vanilla.parameters() if p.requires_grad)

    print('number of params vanilla:', n_parameters_vanilla)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr

    #optimizer = create_optimizer(args, model_without_ddp)
    optimizer_vanilla = create_optimizer(args, model_vanilla_without_ddp)

    loss_scaler = NativeScaler()

    #lr_scheduler, _ = create_scheduler(args, optimizer)
    lr_scheduler_vanilla, _ = create_scheduler(args, optimizer_vanilla)

    criterion = LabelSmoothingCrossEntropy()

    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    print(f"Start training for {args.epochs} epochs")
    
    # estimate rank in the beginning
    if args.factorized_mode == "adaptive":
        est_rank, adjust_rank_scale, args.fr_warmup_epochs = rank_estimation(epoch=-1, vanilla_model=model_vanilla_without_ddp,
                                                                        est_rank_tracker=est_rank_tracker,
                                                                        layers_to_factorize=layers_to_factorize,
                                                                        layer_stable_tracker=layer_stable_tracker,
                                                                        args=args
                                                                    )
    elif args.factorized_mode == "pufferfish":
        est_rank = []
    else:
        raise NotImplementedError("Unsupported facotorized mode ...")
    
    start_time = time.time()
    max_accuracy = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        if epoch in range(args.fr_warmup_epochs):
            train_stats = train_one_epoch(
                model_vanilla, criterion, data_loader_train,
                optimizer_vanilla, device, epoch, loss_scaler,
                args.clip_grad, model_vanilla_ema, mixup_fn,
                set_training_mode=args.finetune == ''  # keep in eval mode during finetuning
            )
            lr_scheduler_vanilla.step(epoch)

            if args.factorized_mode == "adaptive":
                # estimate rank after epoch training
                est_rank, args.fr_warmup_epochs = rank_estimation(epoch=epoch, vanilla_model=model_vanilla_without_ddp, 
                                                                    adjust_rank_scale=adjust_rank_scale,
                                                                    est_rank_tracker=est_rank_tracker,
                                                                    layers_to_factorize=layers_to_factorize,
                                                                    layer_stable_tracker=layer_stable_tracker,
                                                                    args=args
                                                                    )
                if epoch == 2:
                    args.fr_warmup_epochs = 3
        elif epoch == args.fr_warmup_epochs:
            print("!!!!!! Evaluate once before factorizing vanilla model ...")
            test_stats = evaluate(data_loader_val, model_vanilla, device)
            model = create_model_factorized(
                    args.model,
                    pretrained=False,
                    num_classes=args.nb_classes,
                    drop_rate=args.drop,
                    drop_path_rate=args.drop_path,
                    drop_block_rate=None,
                    est_rank=est_rank
                )
            # measuring model FLOPs
            model.eval()
            input = torch.randn(3, 224, 224)[None,:,:,:].float()

            flop = FlopCountAnalysis(model, input)
            
            print("@ Measuring the FLOPs of factorized model ...")
            print(flop_count_table(flop, max_depth=4))
            print(flop_count_str(flop))
            print(flop.total())
            model.to(device)

            # with torch.cuda.device(0):
            #     print("* Measuring FLOPs of Factorized Model ...")
            #     model_factorized_dummy = create_model(
            #         args.model,
            #         pretrained=False,
            #         num_classes=args.nb_classes,
            #         drop_rate=args.drop,
            #         drop_path_rate=args.drop_path,
            #         drop_block_rate=None,
            #         est_rank=est_rank
            #     )
            #     macs, params = get_model_complexity_info(model_factorized_dummy, (3, 224, 224), as_strings=True,
            #                                            print_per_layer_stat=True, verbose=True)
            #     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            #     print('{:<30}  {:<8}'.format('Number of parameters: ', params))
            #     del model_factorized_dummy
            exit()

            _ = decompose_vanilla_model(
                                    vanilla_model=model_vanilla_without_ddp, 
                                    low_rank_model=model, 
                                    est_rank=est_rank,
                                    args=args
                                    )
            # ema
            model_ema = None
            if args.model_ema:
                # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
                model_ema = ModelEma(
                     model,
                     decay=args.model_ema_decay,
                     device='cpu' if args.model_ema_force_cpu else '',
                     resume='')

            if args.distributed:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
                model_without_ddp = model.module

            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print("@@@@ Adapt Factorized Model: {}".format(model))
            print('number of params, factorized model:', n_parameters)
            
            # create optimizer
            linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
            args.lr = linear_scaled_lr / args.factorized_lr_decay
            args.min_lr = args.min_lr / 1.0

            optimizer = create_optimizer(args, model_without_ddp)
            lr_scheduler, _ = create_scheduler(args, optimizer)

            del model_vanilla
            del model_vanilla_without_ddp
            del model_vanilla_ema
            del optimizer_vanilla
            del lr_scheduler_vanilla

            print("###### Evaluate once before training factorized model ...")
            test_stats = evaluate(data_loader_val, model, device)

            #lr_scheduler.step(epoch - 1)
            train_stats = train_one_epoch(
                model, criterion, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                args.clip_grad, model_ema, mixup_fn,
                set_training_mode=args.finetune == ''  # keep in eval mode during finetuning
            )
            # decompose model
            lr_scheduler.step(epoch)
        else:
            train_stats = train_one_epoch(
                model, criterion, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                args.clip_grad, model_ema, mixup_fn,
                set_training_mode=args.finetune == ''  # keep in eval mode during finetuning
            )
            lr_scheduler.step(epoch)

        
        if args.output_dir:
            if epoch in range(args.fr_warmup_epochs):
                pass
            else:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)

        if epoch in range(args.fr_warmup_epochs):
            #test_stats = evaluate(data_loader_val, model_vanilla, device)
            pass
        elif epoch in range(args.fr_warmup_epochs, 260):
            if epoch % 20 == 0:
                test_stats = evaluate(data_loader_val, model, device)

                print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
                max_accuracy = max(max_accuracy, test_stats["acc1"])
                print(f'Max accuracy: {max_accuracy:.2f}%')
        else:
            test_stats = evaluate(data_loader_val, model, device)

            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            max_accuracy = max(max_accuracy, test_stats["acc1"])
            print(f'Max accuracy: {max_accuracy:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)