import sys
import os
import util
import random
import argparse
import logging

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from models.resnet import ResNet18
from models.vgg import VGG19
from torch.autograd import Variable
from torchvision import datasets, transforms

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

CUDA_DEVICE_COUNT=0
best_acc = 0  # best test accuracy

# def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
#     decay = []
#     no_decay = []
#     for name, param in model.named_parameters():
#         if not param.requires_grad:
#             continue
#         if len(param.shape) == 1 or name in skip_list:
#             no_decay.append(param)
#         else:
#             decay.append(param)
#     return [
#         {'params': no_decay, 'weight_decay': 0.},
#         {'params': decay, 'weight_decay': weight_decay}]

# def adjust_learning_rate(optimizer, epoch, args, milestone_epochs):
#     init_lr = args.lr
#     if epoch < milestone_epochs[0]:
#         for group in optimizer.param_groups:
#             if epoch in range(args.lr_warmup_epochs):
#                 factor = 1.0 + (args.scale_factor - 1.0) *min(epoch / args.lr_warmup_epochs, 1.0)
#                 group['lr'] = init_lr * factor
#             else:
#                 group['lr'] = init_lr * args.scale_factor
#     elif (epoch >= milestone_epochs[0] and epoch < milestone_epochs[1]):
#         for group in optimizer.param_groups:
#             group['lr'] = init_lr * args.scale_factor / 10.0
#     elif epoch >= milestone_epochs[1]:
#         for group in optimizer.param_groups:
#             group['lr'] = init_lr * args.scale_factor / 100.0


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


def save_state(model, best_acc):
    print('==> Saving model ...')
    state = {
            'best_acc': best_acc,
            'state_dict': model.state_dict(),
            }
    for key in state['state_dict'].keys():
        if 'module' in key:
            state['state_dict'][key.replace('module.', '')] = \
                    state['state_dict'].pop(key)
    torch.save(state, 'models/nin.pth.tar')

def train(epoch, criterion, optimizer, train_loader=None, device=None):
    model.train()
    epoch_timer = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

        iter_start.record()
        # process the weights including binarization
        bin_op.binarization()
        
        # forwarding
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        
        # backwarding
        loss = criterion(output, target)
        loss.backward()
        
        # restore weights
        bin_op.restore()
        bin_op.updateBinaryGradWeight()
        
        optimizer.step()

        iter_end.record()
        torch.cuda.synchronize()
        iter_comp_dur = float(iter_start.elapsed_time(iter_end))/1000.0
        epoch_timer += iter_comp_dur
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item(),
                optimizer.param_groups[0]['lr']))
    return epoch_timer


def validate(test_loader, model, criterion, epoch, args, device):
    global best_acc

    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    bin_op.binarization()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    bin_op.restore()
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


if __name__=='__main__':
    # prepare the options
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10',
                    help='which dataset to use.')
    parser.add_argument('--arch', action='store', default='nin',
                    help='the architecture for the network: nin')
    parser.add_argument('--lr', type=float, default=0.1,
                    help='the intial learning rate')
    parser.add_argument('--pretrained', action='store', default=None,
                    help='the path to the pretrained model')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
    parser.add_argument('--evaluate', action='store_true',
                    help='evaluate the model')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=42,
                    help='the random seed to use in the experiment for reproducibility')
    parser.add_argument('--test-batch-size', type=int, default=300, metavar='N',
                    help='input batch size for testing (default: 1000)')

    # for large-batch training
    parser.add_argument('--scale-factor', default=4, type=int,
                        help='the factor to scale the batch size.')
    parser.add_argument('--lr-warmup-epochs', type=int, default=5,
                        help='num of epochs to warmup the learning rate for large-batch training.')
    args = parser.parse_args()


    device = torch.device("cuda:{}".format(CUDA_DEVICE_COUNT) if torch.cuda.is_available() else "cpu")
    logger.info("Benchmarking over device: {}".format(device))
    logger.info("Args: {}".format(args))

    # set the seed
    seed(seed=args.seed)

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
    else:
        raise NotImplementedError("Unsupported Dataset ...")

    # load training and test set here:
    if args.dataset in ("cifar10", "cifar100"):
        training_set = data_obj(root='./{}_data'.format(args.dataset), train=True,
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
    else:
        raise NotImplementedError("Unsupported Dataset ...")

    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
                                             num_workers=4,
                                             shuffle=False,
                                             pin_memory=True)

    milestone_epochs = [int(0.5*args.epochs), int(0.75*args.epochs)]

    # define the model
    logger.info('==> building model {}'.format(args.arch))
    if args.arch == 'resnet18':
        model = ResNet18(num_classes=_num_classes)
    elif args.arch == "vgg19":
        model = VGG19(num_classes=_num_classes)
    else:
        raise Exception(args.arch+' is currently not supported')

    # initialize the model
    logger.info('==> Initializing model parameters ...')

    model.to(device)
    logger.info(model)


    optimizer = torch.optim.AdamW(model.parameters(), args.lr,
                                       weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    criterion = nn.CrossEntropyLoss()

    # define the binarization operator
    bin_op = util.BinOp(model)

    running_stats = {"Comp-Time": 0.0,
                    "Best-Val-Acc": 0.0} 
    # start training
    for epoch in range(0, args.epochs+1):
        epoch_time = train(epoch, criterion, optimizer, train_loader=train_loader, device=device)
        running_stats["Comp-Time"] += epoch_time
        best_acc = validate(test_loader, model, criterion, epoch, args, device)
        running_stats["Best-Val-Acc"] = best_acc
        lr_scheduler.step()

    for k, v in running_stats.items():
        logger.info("{}: {}".format(k, v))