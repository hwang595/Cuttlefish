import argparse
import os
import random
import shutil
import time
import warnings
import logging
import json

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
#import torchvision.models as models
import models

from torch.optim import lr_scheduler
from warmup_scheduler import GradualWarmupScheduler


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# helper function because otherwise non-empty strings
# evaluate as True
def bool_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--vanilla-arch', type=str, default='resnet50')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--full-rank-warmup', type=bool_string, default=True,
                        help='if or not to use full-rank warmup')
parser.add_argument('--fr-warmup-epoch', type=int, default=15,
                        help='number of full rank epochs to use')
parser.add_argument('--end-epoch-validation', type=bool_string, default=True,
                        help='to conduct a model validation at the end of each epoch.')
#parser.add_argument('--base-lr', '--base-learning-rate', default=0.1, type=float,
#                    metavar='BLR', help='initial base learning rate, only use when conducting learning rate warmup')
parser.add_argument('--lr-decay-period', nargs='+', type=int)
parser.add_argument('-ldf', '--lr-decay-factor', default=0.1, type=float,
                    help='the decay factor to use when epoch is in lr-decay-period (default: 0.1)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-rf', '--rank-factor', default=4, type=int,
                    metavar='N', help='the rank factor that is going to use in the low rank models')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--model-save-dir', default='/mnt', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--mode', default='vanilla', type=str,
                    help='to or not to use low rank training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--est-rank', default=False, type=bool,
                    help='Wether or not to estimate the rank of the weight during the training process.')
parser.add_argument('--lr-warmup', default=False, type=bool,
                    help='Wether or not to use the learning rate warmup.')
parser.add_argument('-we', '--warmup-epoch', default=5, type=int,
                    help='the epoch that we conduct warmup (default: 5)')
parser.add_argument('--re-warmup', type=bool_string, default=False,
                    help='to rerun warmup or start from scratch')
parser.add_argument('-mpl', '--multiplier', default=16, type=int, 
                    help='the scale we want to conduct for large-batch training e.g. 4, 8, 16, 32, etc (default: 16)')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc1 = 0


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def param_counter(model):
    num_params = 0
    for param_index, (param_name, param) in enumerate(model.named_parameters()):
        num_params += param.numel()
    return num_params


def decompose_weights(model, low_rank_model, rank_factor, args):
    # SVD version
    reconstructed_aggregator = []

    for item_index, (param_name, param) in enumerate(model.state_dict().items()):
        #if len(param.size()) == 4 and item_index not in range(0, 258) and "downsample" not in param_name and "conv3" not in param_name:
        if len(param.size()) == 4 and item_index not in range(0, 258):
            # resize --> svd --> two layer
            param_reshaped = param.view(param.size()[0], -1)
            rank = min(param_reshaped.size()[0], param_reshaped.size()[1])
            u, s, v = torch.svd(param_reshaped)
            #logger.info("### Weights norm u: {}, s: {}, v:{}, w_index: {}, w:{}, w name: {}, w shape: {}".format(torch.norm(u),
            #                                                            torch.norm(s),
            #                                                            torch.norm(v),
            #                                                            item_index,
            #                                                            torch.norm(param),
            #                                                            param_name,
            #                                                            param.size()
            #                                                            ))


            sliced_rank = int(rank/rank_factor)
            u_weight = u * torch.sqrt(s)
            v_weight = torch.sqrt(s) * v
            u_weight_sliced, v_weight_sliced = u_weight[:, 0:sliced_rank], v_weight[:, 0:sliced_rank]

            u_weight_sliced_shape, v_weight_sliced_shape = u_weight_sliced.size(), v_weight_sliced.size()

            #model_weight_v = u_weight.view(u_weight_sliced_shape[0],
            model_weight_v = u_weight_sliced.view(u_weight_sliced_shape[0],
                                                  u_weight_sliced_shape[1], 1, 1)
            
            #model_weight_u = v_weight.t().view(v_weight_sliced_shape[1], 
            model_weight_u = v_weight_sliced.t().view(v_weight_sliced_shape[1], 
                                                      param.size()[1], 
                                                      param.size()[2], 
                                                      param.size()[3])

            #if "downsample" in param_name:
            #    print("@@@@ U size: {}, V size: {}".format(model_weight_u.size(), model_weight_v.size()))
            reconstructed_aggregator.append(model_weight_u)
            reconstructed_aggregator.append(model_weight_v)
        else:
            reconstructed_aggregator.append(param)
            

    model_counter = 0
    reload_state_dict = {}

    for item_index, (param_name, param) in enumerate(low_rank_model.state_dict().items()):
        print("#### {}, {}, recons agg: {}， param: {}".format(item_index, param_name, 
                                                                                reconstructed_aggregator[model_counter].size(),
                                                                               param.size()))
        if "_extra_bns" in args.arch:
            if "bn1_u" in param_name or "bn2_u" in param_name or "bn3_u" in param_name:
                reload_state_dict[param_name] = param
            else:
                assert (reconstructed_aggregator[model_counter].size() == param.size())
                reload_state_dict[param_name] = reconstructed_aggregator[model_counter]
                model_counter += 1            
        else:
            assert (reconstructed_aggregator[model_counter].size() == param.size())
            reload_state_dict[param_name] = reconstructed_aggregator[model_counter]
            model_counter += 1

    low_rank_model.load_state_dict(reload_state_dict)
    return low_rank_model


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        logger.info("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        logger.info("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        logger.info("=> creating model '{}'".format(args.arch))
        if args.mode == "lowrank":
            model = models.__dict__[args.arch](rank_factor=args.rank_factor)
        else:
            model = models.__dict__[args.arch]()

    model_vanilla = models.__dict__[args.vanilla_arch]()
    logger.info("@@@ Num Params: Vanilla Model: {}, Hybrid Model: {}".format(param_counter(model_vanilla),
                                                                             param_counter(model)))
    #model_vanilla = torch.nn.DataParallel(model_vanilla).cuda()

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            
            ## handle vanilla model
            model_vanilla.cuda(args.gpu)
            model_vanilla = torch.nn.parallel.DistributedDataParallel(model_vanilla, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)

            ## handle vanilla model
            model_vanilla = torch.nn.parallel.DistributedDataParallel(model_vanilla)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

        ## handle vanilla model
        model_vanilla = model_vanilla.cuda(args.gpu)        
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()

            ## handle vanilla model
            model_vanilla.features = torch.nn.DataParallel(model_vanilla.features)
            model_vanilla.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

            ## handle vanilla model
            model_vanilla = torch.nn.DataParallel(model_vanilla).cuda()                        


    # define loss function (criterion) and optimizer
    #criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion = LabelSmoothingLoss(classes=1000, smoothing=0.1).cuda(args.gpu)

    weight_decay = args.weight_decay
    if weight_decay:
        parameters = add_weight_decay(model, weight_decay)
        parameters_vanilla = add_weight_decay(model_vanilla, weight_decay)
        weight_decay = 0.
    else:
        parameters = model.parameters()
        parameters_vanilla = model_vanilla.parameters()

    #logger.info("@@@@@@@ Parameters: {}, weight_decay: {}".format(parameters, weight_decay))
    #optimizer = torch.optim.SGD(model.parameters(), args.lr,
    optimizer = torch.optim.SGD(parameters, args.lr,
                                momentum=args.momentum,
                                #weight_decay=args.weight_decay)
                                weight_decay=weight_decay)

    optimizer_vanilla = torch.optim.SGD(parameters_vanilla, args.lr,
                                momentum=args.momentum,
                                #weight_decay=args.weight_decay)
                                weight_decay=weight_decay)


    if args.lr_warmup:
        #scheduler_multi_step = lr_scheduler.MultiStepLR(optimizer, milestones=[e - args.warmup_epoch - 1 for e in args.lr_decay_period], gamma=args.lr_decay_factor)
        #scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=args.multiplier, total_epoch=args.warmup_epoch, after_scheduler=scheduler_multi_step)
        scheduler_multi_step = lr_scheduler.MultiStepLR(optimizer_vanilla, milestones=[e - args.warmup_epoch - 1 for e in args.lr_decay_period], gamma=args.lr_decay_factor)
        scheduler_warmup = GradualWarmupScheduler(optimizer_vanilla, multiplier=args.multiplier, total_epoch=args.warmup_epoch, after_scheduler=scheduler_multi_step)
    else:
        #scheduler_multi_step = lr_scheduler.MultiStepLR(optimizer, milestones=[e for e in args.lr_decay_period], gamma=args.lr_decay_factor)
        scheduler_multi_step = lr_scheduler.MultiStepLR(optimizer_vanilla, milestones=[e for e in args.lr_decay_period], gamma=args.lr_decay_factor)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
            and args.rank % ngpus_per_node == 0):
        logger.info("What it looks like for the (lowrank) model : {}".format(model))
        logger.info("")
        logger.info("What it looks like for the (vanilla) model : {}".format(model_vanilla))


    if args.re_warmup:
        pass
    else:
        args.start_epoch = args.fr_warmup_epoch

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        #adjust_learning_rate(optimizer, epoch, args)
        if epoch in range(args.fr_warmup_epoch):
            for param_group in optimizer_vanilla.param_groups:
                logger.info("Epoch: {}, Current Effective lr: {}".format(epoch, param_group['lr']))
                break        
        else:
            for param_group in optimizer.param_groups:
                logger.info("Epoch: {}, Current Effective lr: {}".format(epoch, param_group['lr']))
                break


        if args.full_rank_warmup and epoch in range(args.fr_warmup_epoch):
            logger.info("Epoch: {}, Warmuping ...".format(epoch))
            # warm-up training
            epoch_time_comm_comp = train(train_loader, model_vanilla, criterion, optimizer_vanilla, epoch, args)
            logger.info("@ Warming-up Epoch: {}, Comm+Comp time cost: {}".format(epoch, epoch_time_comm_comp))
        elif args.full_rank_warmup and epoch == args.fr_warmup_epoch:
            logger.info("Epoch: {}, swtiching to low rank model ...".format(epoch))
            # with open("checkpoint-epoch14.pth.tar", "rb") as ckpt_file:
            #     save_state = torch.load(ckpt_file)
            # model_vanilla.load_state_dict(save_state['state_dict'])
            # logger.info("##### Done loading pretrained model weights ...")

            torch.cuda.synchronize()
            decompose_start = time.time()
            model = decompose_weights(model=model_vanilla, 
                              low_rank_model=model, 
                              rank_factor=args.rank_factor, args=args)
            torch.cuda.synchronize()
            decompose_dur = time.time() - decompose_start
            logger.info("#### Cost for decomposing the weights: {} ....".format(decompose_dur))


            weight_decay = args.weight_decay
            if weight_decay:
                parameters = add_weight_decay(model, weight_decay)
                weight_decay = 0.
            else:
                parameters = model.parameters()

            if args.lr_warmup:
                optimizer = torch.optim.SGD(parameters, args.lr*args.multiplier,
                                            momentum=args.momentum,
                                            #weight_decay=args.weight_decay)
                                            weight_decay=weight_decay)
            else:
                optimizer = torch.optim.SGD(parameters, args.lr,
                                            momentum=args.momentum,
                                            #weight_decay=args.weight_decay)
                                            weight_decay=weight_decay)
            scheduler_multi_step = lr_scheduler.MultiStepLR(optimizer, 
                                                            milestones=[e-args.fr_warmup_epoch for e in args.lr_decay_period], 
                                                            gamma=args.lr_decay_factor)
            epoch_time_comm_comp = train(train_loader, model, criterion, optimizer, epoch, args)
            logger.info("@ Low-rank Training Epoch: {}, Comm+Comp time cost: {}".format(epoch, epoch_time_comm_comp))
        else:
            logger.info("Epoch: {}, low rank training ...".format(epoch))
            epoch_time_comm_comp = train(train_loader, model, criterion, optimizer, epoch, args)
            logger.info("@ Low-rank Training Epoch: {}, Comm+Comp time cost: {}".format(epoch, epoch_time_comm_comp))


        # train for one epoch
        #train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        if args.end_epoch_validation:
            if args.full_rank_warmup and epoch in range(args.fr_warmup_epoch):
                acc1 = validate(val_loader, model_vanilla, criterion, args)
            else:
                acc1 = validate(val_loader, model, criterion, args)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
        else:
            is_best = False

        #if args.est_rank:
        #    if ((epoch+1) % 3 == 0):
        #        estimate_rank(model, epoch)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                #and args.rank % ngpus_per_node == 0):
                and args.rank == 0):
            if args.end_epoch_validation:
                if ((epoch+1) % 1 == 0):
                    if epoch in range(args.warmup_epoch):
                        save_checkpoint({
                            'epoch': epoch + 1,
                            'arch': args.arch,
                            'state_dict': model_vanilla.state_dict(),
                            'best_acc1': best_acc1,
                            'optimizer' : optimizer_vanilla.state_dict(),
                        #}, is_best, filename=args.model_save_dir+"/"+"checkpoint-epoch{}.pth.tar".format(epoch+1))
                        }, is_best, filename=os.path.join(args.model_save_dir, "checkpoint-epoch{}.pth.tar".format(epoch+1)))
                    else:
                        save_checkpoint({
                            'epoch': epoch + 1,
                            'arch': args.arch,
                            'state_dict': model.state_dict(),
                            'best_acc1': best_acc1,
                            'optimizer' : optimizer.state_dict(),
                        #}, is_best, filename=args.model_save_dir+"/"+"checkpoint-epoch{}.pth.tar".format(epoch+1))
                        }, is_best, filename=os.path.join(args.model_save_dir, "checkpoint-epoch{}.pth.tar".format(epoch+1)))
            else:
                if (epoch in range(80, 90)):
                    if args.mode == "lowrank":
                        save_checkpoint({
                            'epoch': epoch + 1,
                            'arch': args.arch,
                            'state_dict': model.state_dict(),
                            'best_acc1': best_acc1,
                            'optimizer' : optimizer.state_dict(),
                        #}, is_best, filename=args.model_save_dir+"/"+"checkpoint-epoch{}.pth.tar".format(epoch+1))
                        }, is_best, filename=os.path.join(args.model_save_dir, "checkpoint-epoch{}.pth.tar".format(epoch+1)))   
                    elif args.mode == "vanilla":
                        save_checkpoint({
                            'epoch': epoch + 1,
                            'arch': args.vanilla_arch,
                            'state_dict': model_vanilla.state_dict(),
                            'best_acc1': best_acc1,
                            'optimizer' : optimizer_vanilla.state_dict(),
                        #}, is_best, filename=args.model_save_dir+"/"+"checkpoint-epoch{}.pth.tar".format(epoch+1))
                        }, is_best, filename=os.path.join(args.model_save_dir, "checkpoint-epoch{}.pth.tar".format(epoch+1)))
                    else:
                        raise NotImplementedError("Unsupported program mode ...")                

        # if ((epoch+1) % 1 == 0):
        #     save_checkpoint({
        #         'epoch': epoch+1,
        #         'arch': args.arch,
        #         'state_dict': model.state_dict(),
        #         'best_acc1': best_acc1,
        #         'optimizer' : optimizer.state_dict(),
        #     }, is_best, filename="checkpoint-epoch{}.pth.tar".format(epoch+1))
        if args.full_rank_warmup:
            # learning rate schedule with full-rank warmup
            if args.lr_warmup:
                if epoch in range(args.fr_warmup_epoch):
                    scheduler_warmup.step()
                else:
                    scheduler_multi_step.step()
            else:
                scheduler_multi_step.step()
        else:
            # learning rate schedule without full-rank warmup
            if args.lr_warmup:
                scheduler_warmup.step()
            else:
                scheduler_multi_step.step()
            

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    comm_and_comp_time = 0.0

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        #torch.cuda.synchronize()
        #forward_start = time.time()
        iter_start.record()
        output = model(images)

        loss = criterion(output, target)

        torch.cuda.synchronize()
        #forward_dur = time.time() - forward_start


        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()

        #torch.cuda.synchronize()
        #backward_start = time.time()
        loss.backward()

        #torch.cuda.synchronize()
        #backward_dur = time.time() - backward_start
        
        optimizer.step()
        #logger.info("Forward cost: {}, Backward cost: {}".format(forward_dur, backward_dur))
        #comm_and_comp_time += forward_dur+backward_dur
        iter_end.record()
        torch.cuda.synchronize()
        iter_time = float(iter_start.elapsed_time(iter_end))/1000.0
        comm_and_comp_time += iter_time

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return comm_and_comp_time


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch in range(0, 30):
        if args.lr_warmup:
            # we adopt the learning rate warmup rule from the Facebook paper: 
            # https://research.fb.com/wp-content/uploads/2017/06/imagenet1kin1h5.pdf
            __pre_computed_lrs = [args.base_lr + (args.lr-args.base_lr) / 5 * i for i in range(6)]
            if epoch in range(0, 6):
                lr = __pre_computed_lrs[epoch]
            else:
                lr = args.lr
        else:
            lr = args.lr
    elif epoch in range(30, 60):
        lr = args.lr / 10
    elif epoch in range(60, 80):
        lr = ((args.lr / 10) / 10)
    elif epoch in range(80, 90):
        lr = ((args.lr / 10) / 10 / 10)

    logger.info("Current Effective lr: {}".format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()