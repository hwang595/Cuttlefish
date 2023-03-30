''''Writing everything into one script..'''
from __future__ import print_function
import math
import os
import pdb
import imp
import sys
import time
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from functools import reduce

from tqdm import tqdm
from tensorboardX import SummaryWriter

from funcs import *
from models.wide_resnet import WideResNet, WRN_50_2, compression
from models.darts import DARTS, Cutout, _data_transforms_cifar10 as darts_transforms
from models.MobileNetV2 import MobileNetV2

try:
    from frob import FactorizedConv, frobdecay, patch_module
    from make import compress_model, parameter_count
except ImportError:
    print("Failed to import factorization")

#os.mkdir('checkpoints/') if not os.path.isdir('checkpoints/') else None

parser = argparse.ArgumentParser(description='Student/teacher training')
parser.add_argument('dataset', type=str, choices=['cifar10', 'cifar100', 'imagenet'], help='Choose between Cifar10/100/imagenet.')
parser.add_argument('mode', choices=['student','teacher'], type=str, help='Learn a teacher or a student')
parser.add_argument('--imagenet_loc', default='/disk/scratch_ssd/imagenet',type=str, help='folder containing imagenet train and val folders')
parser.add_argument('--workers', default=2, type=int, help='No. of data loading workers. Make this high for imagenet')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--GPU', default=None, type=str, help='GPU to use')
parser.add_argument('--student_checkpoint', '-s', default='wrn_40_2_student_KT',type=str, help='checkpoint to save/load student')
parser.add_argument('--teacher_checkpoint', '-t', default='wrn_40_2_T',type=str, help='checkpoint to load in teacher')

#network stuff
parser.add_argument('--network', default='WideResNet', type=str, help='network to use')
parser.add_argument('--wrn_depth', default=40, type=int, help='depth for WRN')
parser.add_argument('--wrn_width', default=2, type=float, help='width for WRN')
parser.add_argument('--module', default=None, type=str, help='path to file containing custom Conv and maybe Block module definitions')
parser.add_argument('--blocktype', default='Basic',type=str, help='blocktype used if specify a --conv')
parser.add_argument('--conv', default=None, type=str, help='Conv type')
parser.add_argument('--AT_split', default=1, type=int, help='group splitting for AT loss')
parser.add_argument('--budget', default=None, type=float, help='budget of parameters to use for the network')

#learning stuff
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--lr_decay_ratio', default=0.2, type=float, help='learning rate decay')
parser.add_argument('--temperature', default=4, type=float, help='temp for KD')
parser.add_argument('--alpha', default=0.0, type=float, help='alpha for KD')
parser.add_argument('--aux_loss', default='AT', type=str, help='AT or SE loss')
parser.add_argument('--beta', default=1e3, type=float, help='beta for AT')
parser.add_argument('--epoch_step', default='[60,120,160]', type=str,
                    help='json list with epochs to drop lr on')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--print_freq', default=10, type=int, help="print stats frequency")
parser.add_argument('--batch_size', default=128, type=int,
                    help='minibatch size')
parser.add_argument('--weight_decay', default=0.0005, type=float)
parser.add_argument('--nocrswd', action='store_true', help='Disable compression ratio scaled weight decay.')
parser.add_argument('--clip_grad', default=None, type=float)
parser.add_argument('--rank-scale', default=0.0, type=float)
parser.add_argument('--target-ratio', default=0.0, type=float)
parser.add_argument('--wd2fd', action='store_true')
parser.add_argument('--spectral', action='store_true')

args = parser.parse_args()

if args.mode == 'teacher':
    logdir = args.teacher_checkpoint
elif args.mode == 'student':
    logdir = "runs/%s.%s"%(args.teacher_checkpoint, args.student_checkpoint)
append = 0
while os.path.isdir(logdir+".%i"%append):
    append += 1
if append > 0:
    logdir = logdir+".%i"%append
writer = SummaryWriter(logdir)
with open(os.path.join(logdir, 'args.json'), 'w') as f:
    json.dump(vars(args), f, indent=5)

def record_oom(train_func):
    def wrapper(*args):
        try:
            _ = train_func(*args)
            result = (True, "Success")
        except RuntimeError as e:
            result = (False, str(e))
        except AssertionError as e:
            result = (True, "Success")
        except Exception as e:
            # something else that's not a memory error going wrong
            result = (False, str(e))

        logfile = "oom_checks.json"
        if os.path.exists(logfile):
            with open(logfile, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        logs.append((sys.argv, result))
        with open(logfile, 'w') as f:
            f.write(json.dumps(logs))
        assert False, "recorded"
    return wrapper

def train_teacher(net, frobenius_decay=0.0, **kwargs):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    net.train()
    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
        if isinstance(net, DARTS):
            outputs, _, aux = net(inputs)
            outputs = torch.cat([outputs, aux], 0)
            targets = torch.cat([targets, targets], 0)
        else:
            outputs, _ = net(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        err1 = 100. - prec1
        err5 = 100. - prec5
        losses.update(loss.item(), inputs.size(0))
        top1.update(err1[0], inputs.size(0))
        top5.update(err5[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if frobenius_decay:
            for module in net.modules():
                if hasattr(module, 'normsqW'):
                    loss += 0.5 * frobenius_decay * module.normsqW
        frobdecay(net, **kwargs)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Error@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, batch_idx, len(trainloader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    writer.add_scalar('train_loss', losses.avg, epoch)
    writer.add_scalar('train_top1', top1.avg, epoch)
    writer.add_scalar('train_top5', top5.avg, epoch)

    train_losses.append(losses.avg)
    train_errors.append(top1.avg)

def train_student(net, teach):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    net.train()
    teach.eval()
    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if isinstance(net, DARTS):
            outputs, student_AMs, aux = net(inputs)
            if aux is not None:
                outputs_student = torch.cat([outputs, aux], 0)
                targets_plus_aux = torch.cat([targets, targets], 0)
            else:
                outputs_student = outputs
                targets_plus_aux = targets
            with torch.no_grad():
                outputs_teacher, teacher_AMs, _ = teach(inputs)
                if aux is not None:
                    outputs_teacher = torch.cat([outputs_teacher, outputs_teacher], 0)
        else:
            outputs_student, student_AMs = net(inputs)
            outputs = outputs_student
            targets_plus_aux = targets
            with torch.no_grad():
                outputs_teacher, teacher_AMs = teach(inputs)

        # If alpha is 0 then this loss is just a cross entropy.
        loss = distillation(outputs_student, outputs_teacher, targets_plus_aux, args.temperature, args.alpha)

        #Add an attention tranfer loss for each intermediate. Let's assume the default is three (as in the original
        #paper) and adjust the beta term accordingly.

        adjusted_beta = (args.beta*3)/len(student_AMs)
        for i in range(len(student_AMs)):
            loss += adjusted_beta * F.mse_loss(student_AMs[i], teacher_AMs[i])

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        err1 = 100. - prec1
        err5 = 100. - prec5
        losses.update(loss.item(), inputs.size(0))
        top1.update(err1[0], inputs.size(0))
        top5.update(err5[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if args.clip_grad is not None:
            max_grad = 0.
            for p in net.parameters():
                g = p.grad.max().item()
                if g > max_grad:
                    max_grad = g
            nn.utils.clip_grad_norm(net.parameters(), args.clip_grad)
            print("Max grad: ", max_grad)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Error@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, batch_idx, len(trainloader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    writer.add_scalar('train_loss', losses.avg, epoch)
    writer.add_scalar('train_top1', top1.avg, epoch)
    writer.add_scalar('train_top5', top5.avg, epoch)

    train_losses.append(losses.avg)
    train_errors.append(top1.avg)


def validate(net, checkpoint=None, best=0.0):

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    net.eval()
    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(valloader):

        inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs, targets = Variable(inputs), Variable(targets)
            if isinstance(net, DARTS):
                outputs, _, _ = net(inputs)
            else:
                outputs, _ = net(inputs)
            if isinstance(outputs,tuple):
                outputs = outputs[0]

            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            err1 = 100. - prec1
            err5 = 100. - prec5

            losses.update(loss.item(), inputs.size(0))
            top1.update(err1[0], inputs.size(0))
            top5.update(err5[0], inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.print_freq == 0:
                print('validate: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Error@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Error@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    batch_idx, len(valloader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    print(' * Error@1 {top1.avg:.3f} Error@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    writer.add_scalar('val_loss', losses.avg, epoch)
    writer.add_scalar('val_top1', top1.avg, epoch)
    writer.add_scalar('val_top5', top5.avg, epoch)

    val_losses.append(losses.avg)
    val_errors.append(top1.avg)

    if checkpoint:

        if isinstance(net, torch.nn.DataParallel):
            state_dict = net.module.state_dict()
        else:
            state_dict = net.state_dict()

        print('Saving..')
        state = {
            'net': state_dict,
            'epoch': epoch,
            'args': sys.argv,
            'width': args.wrn_width,
            'depth': args.wrn_depth,
            'conv': args.conv,
            'blocktype': args.blocktype,
            'module': args.module,
            'train_losses': train_losses,
            'train_errors': train_errors,
            'val_losses': val_losses,
            'val_errors': val_errors,
        }
        print('SAVED!')
        torch.save(state, '%s/checkpoint.t7' % checkpoint)
        if top1.avg > best:
            torch.save(state, '%s/best.t7' % checkpoint)

    return top1.avg

def set_for_budget(eval_network_size, conv_type, budget):
    assert False, "Deprecated this because I don't trust it 100%"
    # set bounds using knowledge of conv_type hyperparam domain
    if 'ACDC' == conv_type:
        bounds = (2, 128)
        post_process = lambda x: int(round(x))
    elif 'Hashed' == conv_type:
        bounds = (0.001,0.9)
        post_process = lambda x: x # do nothing
    elif 'SepHashed' == conv_type:
        bounds = (0.001,0.9)
        post_process = lambda x: x # do nothing
    elif 'Generic' == conv_type:
        bounds = (0.1,0.9)
        post_process = lambda x: x # do nothing
    elif 'TensorTrain' == conv_type:
        bounds = (0.1,0.9)
        post_process = lambda x: x # do nothing
    elif 'Tucker' == conv_type:
        bounds = (0.1,0.9)
        post_process = lambda x: x # do nothing
    elif 'CP' == conv_type:
        bounds = (0.1,0.9)
        post_process = lambda x: x # do nothing
    else:
        raise ValueError("Don't know: "+conv_type)
    def obj(h):
        return abs(budget-eval_network_size(h))

    from scipy.optimize import minimize_scalar
    minimizer = minimize_scalar(obj, bounds=bounds, method='bounded')
    return post_process(minimizer.x)

def n_params(net):
    return sum([reduce(lambda x,y:x*y, p.size()) for p in net.parameters()])

def darts_defaults(args):
    args.batch_size = 96
    args.lr = 0.025
    args.momentum = 0.9
    args.weight_decay = 3e-4
    args.epochs = 600
    return args

def imagenet_defaults(args):
    args.batch_size=256
    args.epochs = 90
    args.lr_decay_ratio = 0.1
    args.weight_decay = 1e-4
    args.epoch_step = '[30,60]'
    args.workers = 16
    return args

def mobilenetv2_defaults(args):
    args.batch_size=256
    args.epochs = 150
    args.lr = 0.05
    args.weight_decay = 4e-5
    args.workers = 16
    return args

def get_scheduler(optimizer, epoch_step, args):
    if args.network == 'WideResNet' or args.network == 'WRN_50_2':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=epoch_step,
                gamma=args.lr_decay_ratio)
    elif args.network == 'DARTS' or args.network == 'MobileNetV2':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    return scheduler

if __name__ == '__main__':
    if args.aux_loss == 'AT':
        aux_loss = at_loss
    elif args.aux_loss == 'SE':
        aux_loss = se_loss

    if args.network == 'DARTS':
        args = darts_defaults(args) # different training hyperparameters
    elif args.network == 'WRN_50_2':
        args = imagenet_defaults(args)
    elif args.network == 'MobileNetV2':
        args = mobilenetv2_defaults(args)

    print(vars(args))
    parallelise = None
#    if args.GPU is not None:
#        if args.GPU[0] != '[':
#            args.GPU = '[' + args.GPU + ']'
#        args.GPU = [i for i, _ in enumerate(json.loads(args.GPU))]
#        if len(args.GPU) > 1:
#            def parallelise(model):
#                model = torch.nn.DataParallel(model, device_ids=args.GPU)
#                model.grouped_parameters = model.module.grouped_parameters
#                return model
#        else:
#            os.environ["CUDA_VISIBLE_DEVICES"] = "%i"%args.GPU[0]

    val_losses = []
    train_losses = []
    val_errors = []
    train_errors = []

    start_epoch = 0
    epoch_step = json.loads(args.epoch_step)

    # Data and loaders
    print('==> Preparing data..')


    if args.dataset == 'cifar10':
        num_classes = 10
        if args.network == 'DARTS':
            transforms_train, transforms_validate = darts_transforms()
        else:
            transforms_train =  transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010)),
                    Cutout(16)])
            transforms_validate = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010)),])
        trainset = torchvision.datasets.CIFAR10(root='~/forwardonly/data/cifar',
                                                train=True, download=True, transform=transforms_train)
        valset = torchvision.datasets.CIFAR10(root='~/forwardonly/data/cifar',
                                               train=False, download=True, transform=transforms_validate)
    elif args.dataset == 'cifar100':
        num_classes = 100
        if args.network == 'DARTS':
            raise NotImplementedError("Could use transforms for CIFAR-10, but not ported yet.")
        transforms_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
        ])
        transforms_validate = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
        ])
        trainset = torchvision.datasets.CIFAR100(root='/disk/scratch/datasets/cifar100',
                                                train=True, download=True, transform=transforms_train)
        validateset = torchvision.datasets.CIFAR100(root='/disk/scratch/datasets/cifar100',
                                               train=False, download=True, transform=transforms_validate)

    elif args.dataset == 'imagenet':
        num_classes = 1000
        traindir = os.path.join(args.imagenet_loc, 'train')
        valdir = os.path.join(args.imagenet_loc, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_validate = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        trainset = torchvision.datasets.ImageFolder(traindir, transform_train)
        valset = torchvision.datasets.ImageFolder(valdir, transform_validate)


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.workers,
                                              pin_memory = True if args.dataset == 'imagenet' else False)
    valloader = torch.utils.data.DataLoader(valset, batch_size=min(100,args.batch_size), shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True if args.dataset == 'imagenet' else False)

    criterion = nn.CrossEntropyLoss()

    # a function for building networks
    def build_network(Conv, Block):
        if args.network == 'WideResNet':
            return WideResNet(args.wrn_depth, args.wrn_width, Conv, Block,
                    num_classes=num_classes, dropRate=0, s=args.AT_split,
                    spectral=args.spectral and not (args.rank_scale or args.target_ratio))
        elif args.network == 'WRN_50_2':
            return WRN_50_2(Conv)
        elif args.network == 'MobileNetV2':
            return MobileNetV2(Conv)
        elif args.network == 'DARTS':
            return DARTS(Conv, num_classes=num_classes)

    # if a budget is specified, figure out what we have to set the
    # hyperparameter to
    if args.budget is not None:
        def eval_network_size(hyperparam):
            net = build_network(*what_conv_block(args.conv+"_%s"%hyperparam, args.blocktype, args.module))
            return n_params(net)
        hyperparam = set_for_budget(eval_network_size, args.conv, args.budget)
        args.conv = args.conv + "_%s"%hyperparam
    # get the classes implementing the Conv and Blocks we're going to use in
    # the network
    Conv, Block = what_conv_block(args.conv, args.blocktype, args.module)

    def load_network(loc):
        net_checkpoint = torch.load(loc)
        start_epoch = net_checkpoint['epoch']
        SavedConv, SavedBlock = what_conv_block(net_checkpoint['conv'],
                net_checkpoint['blocktype'], net_checkpoint['module'])
        net = build_network(SavedConv, SavedBlock).cuda()
        torch.save(net.state_dict(), "checkpoints/darts.template.t7")
        net.load_state_dict(net_checkpoint['net'])
        return net, start_epoch

    if args.mode == 'teacher':

        if args.resume:
            print('Mode Teacher: Loading teacher and continuing training...')
            teach, start_epoch = load_network('checkpoints/%s.t7' % args.teacher_checkpoint)
        else:
            print('Mode Teacher: Making a teacher network from scratch and training it...')
            teach = build_network(Conv, Block)
        origpar = parameter_count(teach)
        if args.rank_scale or args.target_ratio:
            print('Original weight count:', origpar)
            def compress(model, rank_scale, spectral=False):
                get_denom = lambda conv: conv.out_channels*conv.kernel_size[0]*conv.kernel_size[1]
                blocks = [block for layer in list(model.children())[1:-3] for block in layer[0].layer]
                names = [['conv1.conv', 'conv2.conv'] for _ in blocks]
                denoms = [[get_denom(conv.conv) for conv in [block.conv1, block.conv2]] for block in blocks]
                for block, namelist, denomlist in zip(blocks, names, denoms):
                    if hasattr(block, 'convShortcut') and not block.convShortcut is None:
                        namelist.append('convShortcut')
                        denomlist.append(get_denom(block.convShortcut))
                for module, namelist, denomlist in zip(blocks, names, denoms):
                    for name, denom in zip(namelist, denomlist):
                        patch_module(module, name, FactorizedConv,
                                     rank_scale=rank_scale,
                                     init='spectral' if spectral else lambda X: nn.init.normal_(X, 0., math.sqrt(2. / denom)))
                return model 
            no_decay = ['.conv1', '.conv2', 'convShortcut'] if args.wd2fd else []
            skiplist = [] if args.wd2fd else ['.conv1', '.conv2', 'convShortCut']
            if args.target_ratio:
                if args.spectral:
                    _, rank_scale = compress_model(teach, compress, args.target_ratio)
                    compress(teach, rank_scale, spectral=True)
                else:
                    teach, _ = compress_model(teach, compress, args.target_ratio)
            else:
                compress(teach, args.rank_scale, spectral=args.spectral)
            newpar = parameter_count(teach)
            print('Compressed weight count:', newpar)
            print('Compression ratio:', newpar / origpar)
            if not args.wd2fd and not args.nocrswd:
                no_decay.extend(['.conv1', '.conv2', 'convShortcut']) 
                altwd = args.weight_decay * newpar / origpar
            else:
                altwd = 0.0
            parameters = [
                    {'params': [p for n, p in teach.named_parameters() 
                                if not any(nd in n for nd in no_decay)], 
                     'weight_decay': args.weight_decay},
                    {'params': [p for n, p in teach.named_parameters() 
                                if any(nd in n for nd in no_decay)], 
                     'weight_decay': altwd}
            ]
        else:
            skiplist = []
            if args.nocrswd:
                parameters = [{'params': list(teach.parameters()), 'weight_decay': args.weight_decay}]
                print(teach.compression_ratio())
            else:
                parameters = teach.grouped_parameters(args.weight_decay, wd2fd=args.wd2fd)

        use_cuda = torch.cuda.is_available()
        assert use_cuda, 'Error: No CUDA!'
        torch.cuda.set_device(int(args.GPU))
        teach = teach.cuda()

        if parallelise is not None:
            teach = parallelise(teach)

        optimizer = optim.SGD(parameters, lr=args.lr, momentum=args.momentum)
        scheduler = get_scheduler(optimizer, epoch_step, args)
        def schedule_drop_path(epoch, net):
            net.drop_path_prob = 0.2 * epoch / (start_epoch+args.epochs)

        # Decay the learning rate depending on the epoch
        for e in range(0,start_epoch):
            scheduler.step()

        best = float('inf')
        for epoch in tqdm(range(start_epoch, args.epochs)):
            if args.network == 'DARTS': schedule_drop_path(epoch, teach)
            print('Teacher Epoch %d:' % epoch)
            print('Learning rate is %s' % [v['lr'] for v in optimizer.param_groups][0])
            writer.add_scalar('learning_rate', [v['lr'] for v in optimizer.param_groups][0], epoch)
            train_teacher(teach, 
                          frobenius_decay=args.weight_decay if args.wd2fd and not args.conv == 'Conv' else 0.0, 
                          coef=args.weight_decay, 
                          skiplist=skiplist)
            err = validate(teach, args.teacher_checkpoint)
            best = min(best, err)
            scheduler.step()

    elif args.mode == 'student':
        print('Mode Student: First, load a teacher network and convert for (optional) attention transfer')
        teach, _ = load_network('checkpoints/%s.t7' % args.teacher_checkpoint)
        if parallelise is not None:
            teach = parallelise(teach)
        # Very important to explicitly say we require no gradients for the teacher network
        for param in teach.parameters():
            param.requires_grad = False
        validate(teach)
        val_losses, val_errors = [], [] # or we'd save the teacher's error as the first entry

        if args.resume:
            print('Mode Student: Loading student and continuing training...')
            student, start_epoch = load_network('checkpoints/%s.t7' % args.student_checkpoint)
        else:
            print('Mode Student: Making a student network from scratch and training it...')
            student = build_network(Conv, Block).cuda()
        if parallelise is not None:
            student = parallelise(student)

        parameters = student.grouped_parameters(args.weight_decay) if not args.nocrswd else student.parameters()
        optimizer = optim.SGD(parameters,
                lr=args.lr, momentum=args.momentum,
                weight_decay=args.weight_decay)
        scheduler = get_scheduler(optimizer, epoch_step, args)
        def schedule_drop_path(epoch, net):
            net.drop_path_prob = 0.2 * epoch / (start_epoch+args.epochs)

        # Decay the learning rate depending on the epoch
        for e in range(0, start_epoch):
            scheduler.step()

        for epoch in tqdm(range(start_epoch, args.epochs)):
            scheduler.step()
            if args.network == 'DARTS': schedule_drop_path(epoch, student)

            print('Student Epoch %d:' % epoch)
            print('Learning rate is %s' % [v['lr'] for v in optimizer.param_groups][0])
            writer.add_scalar('learning_rate', [v['lr'] for v in optimizer.param_groups][0], epoch)

            train_student(student, teach)
            validate(student, args.student_checkpoint)

    writer.flush()
    if args.rank_scale or args.target_ratio:
        compression = newpar / origpar
    else:
        teach.kwargs['ConvClass'] = Conv
        compression = teach.compression_ratio()
        origpar, newpar = int(round(origpar / compression)), origpar
    with open(os.path.join(logdir, 'results.json'), 'w') as f:
        json.dump({'final validation error': err.item(),
                   'best validation error': best.item(),
                   'original parameter count': origpar,
                   'compressed parameter count': newpar,
                   'compression ratio': compression},
                   f, indent=4)
