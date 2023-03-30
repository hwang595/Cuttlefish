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

from utils import logger, bool_string, decompose_weights, \
                    add_weight_decay, apply_fd, norm_calculator, param_counter

#import models
from lowrank_vgg import VGG19Benchmark
from resnet_cifar10 import *

from ptflops import get_model_complexity_info

best_acc = 0  # best test accuracy

CUDA_DEVICE_COUNT=0


def train(train_loader, model, criterion, optimizer, epoch, device=None):
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
    parser.add_argument('-re', '--resume', default=False, type=bool_string,
                        help='wether or not to resume from a checkpoint.')
    parser.add_argument('-eva', '--evaluate', type=bool_string, default=False,
                        help='wether or not to evaluate the model after loading the checkpoint.')
    parser.add_argument('-rr', '--rank-ratio', default=4.0, type=float,
                        metavar='N', help='the rank factor that is going to use in the low rank models')
    parser.add_argument('-cp', '--ckpt_path', type=str, default="./checkpoint/vgg19_best.pth",
                        help='path to the checkpoint to resume.')

    # for large-batch training
    parser.add_argument('--scale-factor', default=4, type=int,
                        help='the factor to scale the batch size.')
    parser.add_argument('--lr-warmup-epochs', type=int, default=5,
                        help='num of epochs to warmup the learning rate for large-batch training.')


    args = parser.parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda:{}".format(CUDA_DEVICE_COUNT) if torch.cuda.is_available() else "cpu")
    logger.info("Benchmarking over device: {}".format(device))
    logger.info("Args: {}".format(args))

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
                                              pin_memory=True,
                                              drop_last=True)
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
        model = ResNet18Benchmark(rank_ratio=args.rank_ratio, num_classes=_num_classes).to(device)
    elif args.arch == "vgg19":
        model = VGG19Benchmark(rank_ratio=args.rank_ratio, num_classes=_num_classes).to(device)
    else:
        raise NotImplementedError("Unsupported network architecture ...")


    with torch.cuda.device(CUDA_DEVICE_COUNT):
        lowrank_macs, lowrank_params = get_model_complexity_info(
                                            model, (3, 32, 32), as_strings=True,
                                           print_per_layer_stat=True, verbose=True
                                        )
        logger.info("============> Model info: {}, num params: {}, Macs: {}".format(model, param_counter(model), lowrank_macs))

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

    optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                                    momentum=args.momentum, 
                                    weight_decay=1e-4)


    for epoch in range(0, args.epochs):
        epoch_start = time.time()
        
        # adjusting lr schedule
        if epoch < milestone_epochs[0]:
            for group in optimizer.param_groups:
                if epoch in range(args.lr_warmup_epochs):
                    factor = 1.0 + (args.scale_factor - 1.0) *min(epoch / args.lr_warmup_epochs, 1.0)
                    group['lr'] = init_lr * factor
                else:
                    group['lr'] = init_lr * args.scale_factor
        elif (epoch >= milestone_epochs[0] and epoch < milestone_epochs[1]):
            for group in optimizer.param_groups:
                group['lr'] = init_lr * args.scale_factor / 10.0
        elif epoch >= milestone_epochs[1]:
            for group in optimizer.param_groups:
                group['lr'] = init_lr * args.scale_factor / 100.0


        for group in optimizer.param_groups:
            logger.info("### Epoch: {}, Current effective lr: {}".format(epoch, group['lr']))
            break                

        epoch_time = train(train_loader, model, criterion, optimizer, epoch, device=device)
        epoch_end = time.time()
        logger.info("####### Comp Time Cost for Epoch: {} is {}, os time: {}".format(
                                    epoch, epoch_time, epoch_end - epoch_start))


if __name__ == '__main__':
    main()