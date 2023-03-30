# DARTS network definition
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.checkpoint import checkpoint

from collections import namedtuple

from .blocks import DepthwiseSep
from .wide_resnet import group_lowrank, compression

#############################
# Training utils start here # 
#############################


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10():
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if True: # always use cutout
    train_transform.transforms.append(Cutout(16))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform

#####################################
# End of training utils             #
#####################################
# Model definition code starts here #
#####################################

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])


OPS = {
  'none' : lambda C, stride, affine, conv: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine, conv: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, affine, conv: nn.MaxPool2d(3, stride=stride, padding=1),
  'skip_connect' : lambda C, stride, affine, conv: Identity() if stride == 1 else FactorizedReduce(C, C, conv, affine=affine),
  'sep_conv_3x3' : lambda C, stride, affine, conv: SepConv(C, C, 3, stride, 1, Conv=conv, affine=affine),
  'sep_conv_5x5' : lambda C, stride, affine, conv: SepConv(C, C, 5, stride, 2, Conv=conv, affine=affine),
  'sep_conv_7x7' : lambda C, stride, affine, conv: SepConv(C, C, 7, stride, 3, Conv=conv, affine=affine),
  'dil_conv_3x3' : lambda C, stride, affine, conv: DilConv(C, C, 3, stride, 2, 2, Conv=conv, affine=affine),
  'dil_conv_5x5' : lambda C, stride, affine, conv: DilConv(C, C, 5, stride, 4, 2, Conv=conv, affine=affine),
# this is never used so you can remove it without hitting any errors
#  'conv_7x1_1x7' : lambda C, stride, affine, conv: nn.Sequential(
#    nn.ReLU(inplace=False),
#    nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
#    nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
#    nn.BatchNorm2d(C, affine=affine)
#    ),
}


class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, ConvClass, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    #ConvClass = nn.Conv2d if ConvClass is DepthwiseSep else ConvClass
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      ConvClass(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    #return self.op(x)
    return checkpoint(self.op, x)

class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, Conv=DepthwiseSep, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      Conv(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, Conv=DepthwiseSep, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      Conv(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      Conv(C_in, C_out, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, ConvClass, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    #ConvClass = nn.Conv2d if ConvClass is DepthwiseSep else ConvClass
    #ConvClass = nn.Conv2d
    self.conv_1 = ConvClass(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = ConvClass(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    def factorized_reduce(x):
      x = self.relu(x)
      out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
      return self.bn(out)
    #out = checkpoint(cat_1, *[self.conv_1(x), self.conv_2(x[:,:,1:,1:])])
    #return factorized_reduce(x)
    return checkpoint(factorized_reduce, x)
    #return out


class Cell(nn.Module):
  def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev, Conv):
    super(Cell, self).__init__()
    self.Conv = Conv

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, Conv)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, Conv, 1, 1, 0)
    self.preprocess1 = ReLUConvBN(C_prev, C, Conv, 1, 1, 0)
    
    if reduction:
      op_names, indices = zip(*genotype.reduce)
      concat = genotype.reduce_concat
    else:
      op_names, indices = zip(*genotype.normal)
      concat = genotype.normal_concat
    self._compile(C, op_names, indices, concat, reduction)

  def _compile(self, C, op_names, indices, concat, reduction):
    assert len(op_names) == len(indices)
    self._steps = len(op_names) // 2
    self._concat = concat
    self.multiplier = len(concat)

    self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      stride = 2 if reduction and index < 2 else 1
      op = OPS[name](C, stride, True, self.Conv)
      self._ops += [op]
    self._indices = indices

  def forward(self, s0, s1, drop_prob):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    for i in range(self._steps):
      h1 = states[self._indices[2*i]]
      h2 = states[self._indices[2*i+1]]
      op1 = self._ops[2*i]
      op2 = self._ops[2*i+1]
      h1 = op1(h1)
      h2 = op2(h2)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_prob)
      s = h1 + h2
      states += [s]
    return torch.cat([states[i] for i in self._concat], dim=1)
    #return checkpoint(cat_1, *[states[i] for i in self._concat])


class AuxiliaryHeadCIFAR(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 8x8"""
    super(AuxiliaryHeadCIFAR, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), # image size = 2 x 2
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, 2, bias=False),
      nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
    mask = mask.to(x.device)
    x.div_(keep_prob)
    x.mul_(mask)
  return x


class DARTS(nn.Module):
  def __init__(self, ConvClass=DepthwiseSep, C=36, num_classes=10, layers=20, auxiliary=True,
          genotype=DARTS_V2, drop_path_prob=0.2):
    self.kwargs = dict(ConvClass=ConvClass, C=C, num_classes=num_classes,
            layers=layers, auxiliary=auxiliary, genotype=genotype,
            drop_path_prob=drop_path_prob)
    super(DARTS, self).__init__()
    self.drop_path_prob = drop_path_prob
    self._layers = layers
    self._auxiliary = auxiliary

    stem_multiplier = 3
    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
    
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, ConvClass)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr
      if i == 2*layers//3:
        C_to_auxiliary = C_prev

    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

  def compression_ratio(self):
    return compression(self.__class__, self.kwargs)

  def grouped_parameters(self, weight_decay):
    return group_lowrank(self.named_parameters(), weight_decay,
        self.compression_ratio())

  def forward(self, input):
    logits_aux = None
    s0 = s1 = self.stem(input)
    cell_AMs = []
    attention = lambda x: F.normalize(x.pow(2).mean(1).view(x.size(0), -1))
    layers = len(self.cells)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if i in [layers//3, 2*layers//3]:
        cell_AMs.append(attention(s0))
      if i == 2*self._layers//3:
        if self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(s1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits, cell_AMs, logits_aux


if __name__ == '__main__':
  darts = DARTS()
  X = torch.randn(10,3,32,32)
  print(darts(X))
