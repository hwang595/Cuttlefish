import os
from collections import defaultdict
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn.functional as F
from torch.autograd import Variable


def fix_tiny_imagenet(folder):

    if len(os.listdir(folder)) == 200:
        return folder

    new_folder = os.path.join(*folder.split('/')[:-1], 'val-fixed')
    try:
        os.mkdir(new_folder)
    except FileExistsError:
        return new_folder

    images, boxes = defaultdict(lambda: []), {}
    with open(os.path.join(folder, 'val_annotations.txt'), 'r') as f:
        for line in f:
            split = line.split()
            images[split[1]].append(split[0])
            boxes[split[0]] = '\t'.join(split[2:6])
    
    for i, (cls, imgs) in enumerate(images.items()):
        print('\rFixing Tiny-ImageNet Validation Set: Class', i, 'of', len(images), end='')
        clsdir = os.path.join(new_folder, cls)
        imgdir = os.path.join(clsdir, 'images')
        os.mkdir(clsdir)
        os.mkdir(imgdir)
        with open(os.path.join(clsdir, '_'.join([cls, 'boxes.txt'])), 'w') as f:
            for img in imgs:
                imgfile = img.replace('val', cls)
                os.popen(' '.join(['cp', 
                                   os.path.join(folder, 'images', img), 
                                   os.path.join(imgdir, imgfile)]))
                f.write('\t'.join([imgfile, boxes[img]]) + '\n')
    print('\rCompleted Fixing Tiny-ImageNet Validation Set:', len(images), 'Total Classes')
    return new_folder 


def get_transforms(dataset):
    transform_train = None
    transform_test = None
    if dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    if dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    if dataset == 'cinic-10':
        # cinic_directory = '/path/to/cinic/directory'
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cinic_mean, cinic_std)])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cinic_mean, cinic_std)])

    if dataset == 'tiny_imagenet':
        tiny_mean = [0.48024578664982126, 0.44807218089384643, 0.3975477478649648]
        tiny_std = [0.2769864069088257, 0.26906448510256, 0.282081906210584]
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(tiny_mean, tiny_std)])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(tiny_mean, tiny_std)])


    if dataset == 'svhn':
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

    assert transform_test is not None and transform_train is not None, 'Error, no dataset %s' % dataset
    return transform_train, transform_test


def get_dataloader(dataset, train_batch_size, test_batch_size, num_workers=2, root='../data', pin_memory=False):
    transform_train, transform_test = get_transforms(dataset)
    trainset, testset = None, None
    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)

    if dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)

    if dataset == 'cinic-10':
        trainset = torchvision.datasets.ImageFolder(root + '/cinic-10/trainval', transform=transform_train)
        testset = torchvision.datasets.ImageFolder(root + '/cinic-10/test', transform=transform_test)

    if dataset == 'tiny_imagenet':
        trainset = torchvision.datasets.ImageFolder(root + '/tiny_imagenet/train', transform=transform_train)
        testset = torchvision.datasets.ImageFolder(fix_tiny_imagenet(root + '/tiny_imagenet/val'), transform=transform_test)

    if dataset == 'svhn':
        trainset = torchvision.datasets.SVHN(root, split="train",
                                            download=True, transform=transform_train)
        testset = torchvision.datasets.SVHN(root, split="test",
                                            download=True, transform=transform_test)

    assert trainset is not None and testset is not None, 'Error, no dataset %s' % dataset
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True,
                                              num_workers=num_workers, pin_memory=pin_memory)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False,
                                             num_workers=num_workers, pin_memory=pin_memory)

    return trainloader, testloader
