# network definition
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from collections import OrderedDict

# wildcard import for legacy reasons
if __name__ == '__main__':
    from blocks import *
else:
    from .blocks import *

def parse_options(convtype, blocktype):
    # legacy cmdline argument parsing
    if isinstance(convtype,str):
        conv = conv_function(convtype)
    else:
        raise NotImplementedError("Tuple convolution specification no longer supported.")

    if blocktype =='Basic':
        block = BasicBlock
    elif blocktype =='Bottle':
        block = BottleBlock
    elif blocktype =='Old':
        block = OldBlock
    else:
        block = None
    return conv, block

def group_lowrank(named_parameters, weight_decay, compression_ratio, wd2fd=False):
    lowrank_params, other_params = [], []
    for n,p in named_parameters:
        if 'A' in n or 'D' in n:
            lowrank_params.append(p)
        elif 'shuffle' in n:
            lowrank_params.append(p)
        elif 'hashed' in n:
            lowrank_params.append(p)
        elif 'weight_core' in n or 'weight_u' in n:
            lowrank_params.append(p)
        elif 'lowrank' in n:
            lowrank_params.append(p)
        else:
            other_params.append(p)
    return [{'params': lowrank_params,
                'weight_decay': 0. if wd2fd else compression_ratio*weight_decay},
            {'params': other_params,
                'weight_decay': weight_decay}] 

def compression(model_class, kwargs):
    # assume there is a kwarg "conv", which is the convolution we've chosen
    compressed_params = sum([p.numel() for p in
        model_class(**kwargs).parameters()])
    if 'genotype' in list(kwargs.keys()):
        # standard conv with DARTS is DepthwiseSep
        kwargs['ConvClass'] = DepthwiseSep
    else:
        # everything else it's Conv
        kwargs['ConvClass'] = Conv
    uncompressed_params = sum([p.numel() for p in
        model_class(**kwargs).parameters()])
    ratio = float(compressed_params)/float(uncompressed_params)
    print("Compression: %i to %i, ratio"%(uncompressed_params,
        compressed_params), ratio)
    return ratio


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, ConvClass, block, num_classes=10, dropRate=0.0, s = 1, spectral=False):
        super(WideResNet, self).__init__()
        self.kwargs = dict(depth=depth, widen_factor=widen_factor, ConvClass=ConvClass,
                block=block, num_classes=num_classes, dropRate=dropRate, s=s)
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        nChannels = [int(a) for a in nChannels]
        assert ((depth - 4) % 6 == 0) # why?
        n = (depth - 4) // 6

        assert n % s == 0, 'n mod s must be zero'

        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = torch.nn.ModuleList()
        for i in range(s):
            self.block1.append(NetworkBlock(int(n//s), nChannels[0] if i == 0 else nChannels[1],
                                            nChannels[1], block, 1, dropRate, ConvClass))
        # 2nd block
        self.block2 = torch.nn.ModuleList()
        for i in range(s):
            self.block2.append(NetworkBlock(int(n//s), nChannels[1] if i == 0 else nChannels[2],
                                            nChannels[2], block, 2 if i == 0 else 1, dropRate, ConvClass))
        # 3rd block
        self.block3 = torch.nn.ModuleList()
        for i in range(s):
            self.block3.append(NetworkBlock(int(n//s), nChannels[2] if i == 0 else nChannels[3],
                                            nChannels[3], block, 2 if i == 0 else 1, dropRate, ConvClass))
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        # normal is better than uniform initialisation
        # this should really be in `self.reset_parameters`
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                try:
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                except AttributeError:
                    if spectral:
                        weight = m.conv_weight()
                        weight.data.normal_(0, math.sqrt(2. / n))
                        m.tn_weight = m.TnConstructor(weight.data.squeeze(), rank_scale=m.rank_scale)
                        m.register_tnparams(m.tn_weight.cores, m.tn_weight.Us)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def compression_ratio(self):
        return compression(self.__class__, self.kwargs)

    def grouped_parameters(self, weight_decay, **kwargs):
        # iterate over parameters and separate those in ACDC layers
        return group_lowrank(self.named_parameters(), weight_decay,
                self.compression_ratio(), **kwargs)

    def forward(self, x):
        activation_maps = []
        out = self.conv1(x)
        #activations.append(out)
        attention = lambda x: F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

        for sub_block in self.block1:
            out = sub_block(out)
            activation_maps.append(attention(out))

        for sub_block in self.block2:
            out = sub_block(out)
            activation_maps.append(attention(out))

        for sub_block in self.block3:
            out = sub_block(out)
            activation_maps.append(attention(out))

        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out), activation_maps


class ResNet(nn.Module):

    def __init__(self, ConvClass, layers, block=Bottleneck, widen=1,
            num_classes=1000, expansion=4):
        self.kwargs = dict(layers=layers, expansion=expansion,
                ConvClass=ConvClass, widen=widen, num_classes=num_classes,
                block=block)
        self.expansion = expansion
        super(ResNet, self).__init__()
        self.Conv = ConvClass
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64*widen, layers[0])
        self.layer2 = self._make_layer(block, 128*widen, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256*widen, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512*widen, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d((7, 7), 1, 0)
        self.fc = nn.Linear(512*widen * self.expansion, num_classes)
        #self.fc = self.Conv(512*widen * self.expansion, num_classes, kernel_size=1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if hasattr(m, 'weight'):
                    w = m.weight 
                    nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * self.expansion:
            downsample = nn.Sequential(OrderedDict([
                ('conv', self.Conv(self.inplanes, planes * self.expansion,
                    kernel_size=1, stride=stride, padding=0, bias=False)),
                ('bn', nn.BatchNorm2d(planes * self.expansion))
            ]))

        layers = []
        layers.append(block(self.inplanes, planes, self.Conv, stride, downsample, self.expansion))
        self.inplanes = planes * self.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.Conv, expansion=self.expansion))

        return nn.Sequential(*layers)

    def compression_ratio(self):
        return compression(self.__class__, self.kwargs)

    def grouped_parameters(self, weight_decay):
        # iterate over parameters and separate those in other layer types
        return group_lowrank(self.named_parameters(), weight_decay,
                self.compression_ratio())

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        attention_maps = []
        attention = lambda x: F.normalize(x.pow(2).mean(1).view(x.size(0), -1))
        if self.train:
            x = self.layer1(x)
            #x = checkpoint(self.layer1, x)
            #x = checkpoint_sequential(self.layer1, 1, x)
        else:
            x = self.layer1(x)
        attention_maps.append(attention(x))
        if self.train:
            x = self.layer2(x)
            #x = checkpoint(self.layer2, x)
            #x = checkpoint_sequential(self.layer2, 1, x)
        else:
            x = self.layer2(x)
        attention_maps.append(attention(x))
        if self.train:
            x = self.layer3(x)
            #x = checkpoint(self.layer3, x)
            #x = checkpoint_sequential(self.layer3, 1, x)
        else:
            x = self.layer3(x)
        
        attention_maps.append(attention(x))
        if self.train:
            x = self.layer4(x)
            #x = checkpoint(self.layer4, x)
            #x = checkpoint_sequential(self.layer4, 1, x)
        else:
            x = self.layer4(x)

        attention_maps.append(attention(x))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        #x = x.view(x.size(0), -1)

        return x, attention_maps


def WRN_50_2(Conv, Block=None):
    assert Block is None
    return ResNet(Conv, [3, 4, 6, 3], widen=2, expansion=2)

def test():
    net = WideResNet(28, 10, conv_function("Shuffle_7"), BasicBlock)
    params = net.grouped_parameters(5e-4)
    params = [d['params'] for d in params]
    print("Low-rank:  ", sum([p.numel() for p in params[0]]))
    print("Full-rank: ", sum([p.numel() for p in params[1]]))
    print("FC:        ", sum([p.numel() for p in net.fc.parameters()]))

    net = WRN_50_2(conv_function("Shuffle_7"))
    params = net.grouped_parameters(5e-4)
    params = [d['params'] for d in params]
    print("Low-rank:  ", sum([p.numel() for p in params[0]]))
    print("Full-rank: ", sum([p.numel() for p in params[1]]))
    print("FC:        ", sum([p.numel() for p in net.fc.parameters()]))
    x = torch.randn(1,3,224,224).float()
    y, _ = net(Variable(x))
    print(y.size())

if __name__ == '__main__':
    test()
