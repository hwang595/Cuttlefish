from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset

import time
import random
import numpy as np
import itertools

import os

from utils import logger, rank_estimation, bool_string, decompose_weights, \
                    add_weight_decay, apply_fd, norm_calculator, param_counter, \
                    RESNET18_FR_BLOCKS_IDX_MAP
#import models
from lowrank_vgg import FullRankVGG19, PufferfishVGG19, LowRankVGG19Adapt
from resnet_cifar10 import *

from ptflops import get_model_complexity_info

best_acc = 0  # best test accuracy

CUDA_DEVICE_COUNT=0


def train(train_loader, model, criterion, optimizer, epoch, fd=True, coef=1e-4, fact_list=(), device=None):
    model.train()
    epoch_timer = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        iter_start.record()
        
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # add Frob. decay:
        if fd:
            apply_fd(model, weight_decay=coef, factor_list=fact_list)

        optimizer.step()
        iter_end.record()

        torch.cuda.synchronize()
        iter_comp_dur = float(iter_start.elapsed_time(iter_end))/1000.0

        epoch_timer += iter_comp_dur

        if batch_idx % 40 == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]  Loss: {:.6f}'.format(
               epoch, batch_idx * len(data), len(train_loader.dataset),
               100. * batch_idx / len(train_loader), loss.item()))
        
    return epoch_timer


def validate(test_loader, model, criterion, epoch, args, device):
    global best_acc

    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    assert total == len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    logger.info('\nEpoch: {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(epoch, 
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    if not args.evaluate:
        if acc > best_acc:
            logger.info('###### Saving..')
            state = {
                'net': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }

            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/{}_seed{}_best.pth'.format(args.arch, args.seed))
            best_acc = acc
    return best_acc


def seed(seed):
    # seed = 1234
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #TODO: Do we need deterministic in cudnn ? Double check
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    logger.info("Seeded everything")


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Pufferfish-2 Cifar-10')
    parser.add_argument('-a', '--arch', default='resnet18')
    parser.add_argument('--dataset', type=str, default='cifar10',
                    help='which dataset to use.')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--seed', type=int, default=42,
                        help='the random seed to use in the experiment for reproducibility')
    parser.add_argument('--test-batch-size', type=int, default=300, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='weight decay coefficient.')
    parser.add_argument('--full-rank-warmup', type=bool_string, default=True,
                            help='if or not to use full-rank warmup')
    parser.add_argument('--fr-warmup-epoch', type=int, default=15,
                            help='number of full rank epochs to use')
    parser.add_argument('-re', '--resume', default=False, type=bool_string,
                        help='wether or not to resume from a checkpoint.')
    parser.add_argument('-eva', '--evaluate', type=bool_string, default=False,
                        help='wether or not to evaluate the model after loading the checkpoint.')
    parser.add_argument('-fd', '--frob-decay', type=bool_string, default=True,
                        help='wether or not to enable Frobenius decay.')
    parser.add_argument('--extra-bns', type=bool_string, default=True,
                        help='wether or not to enable the extra BNs.')
    parser.add_argument('-rr', '--rank-ratio', default=4, type=int,
                        metavar='N', help='the rank factor that is going to use in the low rank models')
    parser.add_argument('-cp', '--ckpt_path', type=str, default="./checkpoint/vgg19_best.pth",
                        help='path to the checkpoint to resume.')
    parser.add_argument('--rank-est-metric', default='scaled-stable-rank', type=str,
                        help='we can do scaled-stable-rank or vanilla-stable-rank.')

    # training mode
    parser.add_argument('--mode', type=str, default='vanilla',
                        help='use full rank or low rank models')

    # for large-batch training
    parser.add_argument('--scale-factor', default=4, type=int,
                        help='the factor to scale the batch size.')
    parser.add_argument('--lr-warmup-epochs', type=int, default=5,
                        help='num of epochs to warmup the learning rate for large-batch training.')


    args = parser.parse_args()
    if args.frob_decay and args.extra_bns:
        raise ValueError(
                "Can Enable Frob Decay and Extra BNs at the Same Time!!!"
                )
    torch.manual_seed(args.seed)

    device = torch.device("cuda:{}".format(CUDA_DEVICE_COUNT) if torch.cuda.is_available() else "cpu")
    logger.info("Benchmarking over device: {}".format(device))
    logger.info("Args: {}".format(args))


    if args.mode == "vanilla":
        args.fr_warmup_epoch = args.epochs

    # let's enable cudnn benchmark
    seed(seed=args.seed)

    milestone_epochs = [int(0.5*args.epochs), int(0.75*args.epochs)]
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #normalize
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    # data prep for test set
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        #normalize
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # adjust dataset
    if args.dataset == "cifar10":
        data_obj = datasets.CIFAR10
        _num_classes = 10
    elif args.dataset == "cifar100":
        data_obj = datasets.CIFAR100
        _num_classes = 100
    elif args.dataset == "svhn":
        data_obj = datasets.SVHN
        _num_classes = 10
    else:
        raise NotImplementedError("Unsupported Dataset ...")

    # load training and test set here:
    if args.dataset in ("cifar10", "cifar100"):
        training_set = data_obj(root='./{}_data'.format(args.dataset), train=True,
                                            download=True, transform=transform_train)
    elif args.dataset == "svhn":
        training_set = data_obj(root='./{}_data'.format(args.dataset), split="train",
                                            download=True, transform=transform_train)
    else:
        raise NotImplementedError("Unsupported Dataset ...")
    
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size,
                                              num_workers=4,
                                              shuffle=True,
                                              pin_memory=True)
    if args.dataset in ("cifar10", "cifar100"):
        testset = data_obj(root='./{}_data'.format(args.dataset), train=False,
                                           download=True, transform=transform_test)
    elif args.dataset == "svhn": 
        testset = data_obj(root='./{}_data'.format(args.dataset), split="test",
                                           download=True, transform=transform_test)
    else:
        raise NotImplementedError("Unsupported Dataset ...")

    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
                                             num_workers=4,
                                             shuffle=False,
                                             pin_memory=True)

    if args.arch == "resnet18":
        name_layers_to_factorize = ("layer2.1.conv1", "layer2.1.conv2",
                                    "layer3.0.conv1", "layer3.0.conv2", 
                                    "layer3.1.conv1", "layer3.1.conv2", 
                                    "layer4.0.conv1", "layer4.0.conv2", 
                                    "layer4.1.conv1", "layer4.1.conv2")
        layers_to_factorize = [s + ".weight" for s in name_layers_to_factorize]
        if args.mode == "vanilla":
            pass
        elif args.mode in ("lowrank", "baseline", "pufferfish"):
            if args.mode == "lowrank":
                model = None
            elif args.mode == "baseline":
                model = LowrankResNet18(rank_ratio=args.rank_ratio, num_fr_blocks=0, num_classes=_num_classes).to(device)
            elif args.mode == "pufferfish":
                model = PufferfishResNet18(num_classes=_num_classes).to(device)
            else:
                raise NotImplementedError("Unsupported training mode ...")
        else:
            raise NotImplementedError("unsupported mode ...")
        vanilla_model = ResNet18(num_classes=_num_classes).to(device)
    elif args.arch == "vgg19":
        layers_to_factorize = [s + ".weight" for s in (
                                   "block2.0", "block2.3", "block2.6", "block2.9",  
                                   "block3.0", "block3.3", "block3.6", "block3.9",
                                   "block3.13", "block3.16", "block3.19", "block3.22",
                                   )]
        if args.mode == "vanilla":
            pass
        elif args.mode in ("lowrank", "baseline"):
            model = None
        elif args.mode == "pufferfish":
            model = PufferfishVGG19(num_classes=_num_classes).to(device)          
        else:
            raise NotImplementedError("unsupported mode ...")
        vanilla_model = FullRankVGG19(num_classes=_num_classes).to(device)
    else:
        raise NotImplementedError("Unsupported network architecture ...")

  
    est_rank_tracker = [[] for _ in range(len(layers_to_factorize))]
    layer_stable_tracker = [False for _ in range(len(layers_to_factorize))]

    with torch.cuda.device(CUDA_DEVICE_COUNT):
        if args.mode in ("baseline", "pufferfish"):
            lowrank_macs, lowrank_params = get_model_complexity_info(
                                                model, (3, 32, 32), as_strings=True,
                                               print_per_layer_stat=True, verbose=True
                                            )
            logger.info("============> Lowrank Model info: {}, num params: {}, Macs: {}".format(model, param_counter(model), lowrank_macs))
        
        vanilla_macs, vanilla_params = get_model_complexity_info(
                                               vanilla_model, (3, 32, 32), as_strings=True,
                                               print_per_layer_stat=True, verbose=True
                                            ) 

    logger.info("============> Vanilla Model info: {}, num params: {}, Macs: {}".format(
                                                vanilla_model, param_counter(vanilla_model), vanilla_macs))


    criterion = nn.CrossEntropyLoss()
    init_lr = args.lr

    if args.resume:
        logger.info('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.ckpt_path)
        model.load_state_dict(checkpoint['net'])

        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        if args.evaluate:
            validate(
                     test_loader=test_loader,
                     model=model, 
                     criterion=criterion, 
                     epoch=start_epoch,
                     args=args,
                     device=device)           
            exit()

    if args.mode in ("baseline", "pufferfish"):
        optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                                        momentum=args.momentum, 
                                        weight_decay=1e-4)

    vanilla_parameters = add_weight_decay(vanilla_model, 1e-4)
    weight_decay = 0.
    vanilla_optimizer = torch.optim.SGD(vanilla_parameters, args.lr,
                                       momentum=args.momentum, 
                                       weight_decay=weight_decay)
    #vanilla_optimizer = torch.optim.SGD(vanilla_model.parameters(), args.lr,
    #                                    momentum=args.momentum, 
    #                                    weight_decay=1e-4)
    
    if args.mode == "lowrank":
        est_rank_list, adjust_rank_scale, args.fr_warmup_epoch = rank_estimation(epoch=-1, net=vanilla_model,
                                                                        est_rank_tracker=est_rank_tracker,
                                                                        layers_to_factorize=layers_to_factorize,
                                                                        layer_stable_tracker=layer_stable_tracker,
                                                                        args=args)
        logger.info("##### est rank list: {}, len rank list: {}".format(
                                                            est_rank_list, 
                                                            len(est_rank_list)))
    else:
        est_rank_list, adjust_rank_scale, _ = rank_estimation(epoch=-1, net=vanilla_model,
                                                                        est_rank_tracker=est_rank_tracker,
                                                                        layers_to_factorize=layers_to_factorize,
                                                                        layer_stable_tracker=layer_stable_tracker,
                                                                        args=args)

    running_stats = {"Comp-Time": 0.0,
                    "Best-Val-Acc": 0.0}   

    for epoch in range(0, args.epochs):
        epoch_start = time.time()
        
        # adjusting lr schedule
        if epoch < milestone_epochs[0]:
            if args.mode in ("baseline", "pufferfish", "lowrank"):
                if epoch <= args.fr_warmup_epoch:
                    pass
                elif epoch in range(args.fr_warmup_epoch + 1, args.fr_warmup_epoch + args.lr_warmup_epochs):
                    factor = 1.0 + (args.scale_factor - 1.0) *min((epoch-args.fr_warmup_epoch)/args.lr_warmup_epochs, 1.0)
                    for group in optimizer.param_groups:
                        group['lr'] = init_lr * factor
                else:
                    for group in optimizer.param_groups:
                        group['lr'] = init_lr * args.scale_factor

            for group in vanilla_optimizer.param_groups:
                if epoch in range(args.lr_warmup_epochs):
                    factor = 1.0 + (args.scale_factor - 1.0) *min(epoch / args.lr_warmup_epochs, 1.0)
                    group['lr'] = init_lr * factor
                else:
                    group['lr'] = init_lr * args.scale_factor
        elif (epoch >= milestone_epochs[0] and epoch < milestone_epochs[1]):
            if args.mode in ("baseline", "pufferfish", "lowrank"):
                if epoch <= args.fr_warmup_epoch:
                    pass
                else:
                    for group in optimizer.param_groups:
                        group['lr'] = init_lr * args.scale_factor / 10.0

            for group in vanilla_optimizer.param_groups:
                group['lr'] = init_lr * args.scale_factor / 10.0
        elif epoch >= milestone_epochs[1]:
            if args.mode in ("baseline", "pufferfish", "lowrank"):
                if epoch <= args.fr_warmup_epoch:
                    pass
                else:
                    for group in optimizer.param_groups:
                        group['lr'] = init_lr * args.scale_factor / 100.0

            for group in vanilla_optimizer.param_groups:
                group['lr'] = init_lr * args.scale_factor / 100.0


        if args.mode in ("baseline", "pufferfish"):
            if epoch < args.fr_warmup_epoch:
                for group in vanilla_optimizer.param_groups:
                    logger.info("### Epoch: {}, Current effective lr: {}".format(epoch, group['lr']))
                    break
            elif epoch == args.fr_warmup_epoch:
                pass
            else:
                for group in optimizer.param_groups:
                    logger.info("### Epoch: {}, Current effective lr: {}".format(epoch, group['lr']))
                    break
        elif args.mode == "lowrank":
            if epoch < args.fr_warmup_epoch:
                for group in vanilla_optimizer.param_groups:
                    logger.info("### Epoch: {}, Current effective lr: {}".format(epoch, group['lr']))
                    break
            elif epoch == args.fr_warmup_epoch:
                pass
            else:
                for group in optimizer.param_groups:
                    logger.info("### Epoch: {}, Current effective lr: {}".format(epoch, group['lr']))
                    break                
        elif args.mode == "vanilla":
            for group in vanilla_optimizer.param_groups:
                logger.info("### Epoch: {}, Current effective lr: {}".format(epoch, group['lr']))
                break

        if args.full_rank_warmup and epoch in range(args.fr_warmup_epoch):
            logger.info("Epoch: {}, Warmuping ...".format(epoch))

            # support vanilla training
            rank_est_start = torch.cuda.Event(enable_timing=True)
            rank_est_end = torch.cuda.Event(enable_timing=True)
            rank_est_start.record()
            if args.mode == "lowrank":
                est_rank_list, args.fr_warmup_epoch = rank_estimation(epoch=epoch, net=vanilla_model, adjust_rank_scale=adjust_rank_scale, 
                                                                    est_rank_tracker=est_rank_tracker,
                                                                    layers_to_factorize=layers_to_factorize,
                                                                    layer_stable_tracker=layer_stable_tracker,
                                                                    args=args)
            else:
                est_rank_list, _ = rank_estimation(epoch=epoch, net=vanilla_model, adjust_rank_scale=adjust_rank_scale, 
                                                    est_rank_tracker=est_rank_tracker,
                                                    layers_to_factorize=layers_to_factorize,
                                                    layer_stable_tracker=layer_stable_tracker,
                                                    args=args)                
            rank_est_end.record()
            torch.cuda.synchronize()
            rank_est_dur = float(rank_est_start.elapsed_time(rank_est_end))/1000.0
            logger.info("#### Epoch: {}, Cost for Rank Est: {} ....".format(epoch, rank_est_dur))
            
            if args.mode == "lowrank":
                running_stats["Comp-Time"] += rank_est_dur

            epoch_time = train(train_loader, vanilla_model, criterion, vanilla_optimizer, epoch, fd=False,
                                    device=device)
        elif args.full_rank_warmup and epoch == args.fr_warmup_epoch:
            logger.info("Epoch: {}, swtiching to low rank model ...".format(epoch))
            if args.mode == "lowrank":
                est_rank_list, _ = rank_estimation(epoch=epoch, net=vanilla_model, adjust_rank_scale=adjust_rank_scale, 
                                                                        est_rank_tracker=est_rank_tracker,
                                                                        layers_to_factorize=layers_to_factorize,
                                                                        layer_stable_tracker=layer_stable_tracker,
                                                                        args=args)
                if args.arch == "resnet18":
                    model = LowrankResNet18Adapt(
                                                    rank_list=est_rank_list, 
                                                    num_classes=_num_classes, 
                                                    frob_decay=args.frob_decay,
                                                    extra_bns=args.extra_bns
                                                ).to(device)
                elif args.arch == "vgg19":
                    model = LowRankVGG19Adapt(
                                                rank_list=est_rank_list, 
                                                num_classes=_num_classes,
                                                frob_decay=args.frob_decay,
                                                extra_bns=args.extra_bns
                                            ).to(device)
                else:
                    raise NotImplementedError("Unsupported network architecture ...")

                decompose_start = torch.cuda.Event(enable_timing=True)
                decompose_end = torch.cuda.Event(enable_timing=True)
                decompose_start.record()
                model = decompose_weights(model=vanilla_model, 
                                  low_rank_model=model, 
                                  rank_list=est_rank_list,
                                  rank_ratio=None,
                                  args=args)
                decompose_end.record()
                torch.cuda.synchronize()
                decompose_dur = float(decompose_start.elapsed_time(decompose_end))/1000.0
                logger.info("#### Cost for decomposing the weights: {} ....".format(decompose_dur))

                logger.info("### The Adapt lowrank net: {}, {}".format(model, param_counter(model)))
                with torch.cuda.device(CUDA_DEVICE_COUNT):
                    lowrank_macs, lowrank_params = get_model_complexity_info(model, (3, 32, 32), as_strings=True,
                                                           print_per_layer_stat=True, verbose=True)
                logger.info("====> Adaptive Model info: num params: {}, Macs: {}".format(param_counter(model), lowrank_macs))
                running_stats["Comp-Time"] += decompose_dur
            else:
                decompose_start = torch.cuda.Event(enable_timing=True)
                decompose_end = torch.cuda.Event(enable_timing=True)
                decompose_start.record()
                model = decompose_weights(model=vanilla_model, 
                                  low_rank_model=model, 
                                  rank_list=None,
                                  rank_ratio=args.rank_ratio,
                                  args=args)
                decompose_end.record()
                torch.cuda.synchronize()
                decompose_dur = float(decompose_start.elapsed_time(decompose_end))/1000.0
                logger.info("#### Cost for decomposing the weights: {} ....".format(decompose_dur))
                running_stats["Comp-Time"] += decompose_dur

            if args.mode == "lowrank":
                # we will need to generate skip list here and add FD manually
                if args.arch == "resnet18":
                    skip_layer = set([s[0] + s[1] + ".weight" for s in itertools.product(
                                                ["layer2.1.conv1", "layer2.1.conv2",
                                                "layer3.0.conv1", "layer3.0.conv2", 
                                                "layer3.1.conv1", "layer3.1.conv2", 
                                                "layer4.0.conv1", "layer4.0.conv2", 
                                                "layer4.1.conv1", "layer4.1.conv2"], ["_u", "_v"])])   
                elif args.arch == "vgg19":
                    skip_layer =  set([s[0] + s[1] + ".weight" for s in itertools.product(
                                                ["conv{}".format(i) for i in range(5, 17)], ["_u", "_v"])])             
                else:
                    raise NotImplementedError("Unsupported network architecture ...")

                if args.frob_decay:
                    parameters = add_weight_decay(model, 1e-4, skip_list=skip_layer)
                else:
                    parameters = add_weight_decay(model, 1e-4)
            else:
                skip_layer = None
                parameters = add_weight_decay(model, 1e-4)
            weight_decay = 0.
            optimizer = torch.optim.SGD(parameters, args.lr,
                                            momentum=args.momentum, 
                                            weight_decay=weight_decay)

            for group in optimizer.param_groups:
                logger.info("### Epoch: {}, Current effective lr: {}".format(epoch, group['lr']))
                break

            epoch_time = train(train_loader, model, criterion, optimizer, epoch, 
                                fd=args.frob_decay, coef=args.weight_decay, fact_list=skip_layer, device=device)
        else:
            logger.info("Epoch: {}, {} training ...".format(epoch, args.mode))
            epoch_time = train(train_loader, model, criterion, optimizer, epoch, 
                                fd=args.frob_decay, coef=args.weight_decay, fact_list=skip_layer, device=device)

        running_stats["Comp-Time"] += epoch_time
        epoch_end = time.time()
        logger.info("####### Comp Time Cost for Epoch: {} is {}, os time: {}".format(
                                    epoch, epoch_time, epoch_end - epoch_start))

        # eval
        if args.full_rank_warmup and epoch in range(args.fr_warmup_epoch):
            best_acc = validate(
                     test_loader=test_loader,
                     model=vanilla_model, 
                     criterion=criterion, 
                     epoch=epoch,
                     args=args,
                     device=device)
        else:
            best_acc = validate(
                     test_loader=test_loader,
                     model=model, 
                     criterion=criterion, 
                     epoch=epoch,
                     args=args, 
                     device=device)
        running_stats["Best-Val-Acc"] = best_acc
    for k, v in running_stats.items():
        logger.info("{}: {}".format(k, v))


if __name__ == '__main__':
    main()