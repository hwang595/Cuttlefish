# blocks and convolution definitions
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

if __name__ == 'blocks' or __name__ == '__main__':
    from hashed import HashedConv2d, HalfHashedSeparable, HashedSeparable
    from decomposed import TensorTrain, Tucker, CP
else:
    from .hashed import HashedConv2d, HalfHashedSeparable, HashedSeparable
    from .decomposed import TensorTrain, Tucker, CP

def HashedDecimate(in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=False):
    # Hashed Conv2d using 1/10 the original parameters
    original_params = out_channels*in_channels*kernel_size*kernel_size // groups
    budget = original_params//10
    return HashedConv2d(in_channels, out_channels, kernel_size, budget,
            stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias)

def SepHashedDecimate(in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=False):
    # Hashed Conv2d using 1/10 the original parameters
    assert groups == 1
    original_params = out_channels*in_channels*kernel_size*kernel_size
    budget = original_params//10
    conv = HalfHashedSeparable(in_channels, out_channels, kernel_size,
            budget, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
    n_params = sum([p.numel() for p in conv.parameters()])
    budget = budget + conv.hashed.bias.numel()
    assert n_params <= budget, f"{n_params} > {budget}"
    return conv


try:
    from pytorch_acdc.layers import FastStackedConvACDC
except ImportError:
    print("Failed to import ACDC")


def ACDC(in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=False):
    return FastStackedConvACDC(in_channels, out_channels, kernel_size, 12,
            stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias)

def OriginalACDC(in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=False):
    return FastStackedConvACDC(in_channels, out_channels, kernel_size, 12,
            stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias, original=True)


class GenericLowRank(nn.Module):
    """A generic low rank layer implemented with a linear bottleneck, using two
    Conv2ds in sequence. Preceded by a depthwise grouped convolution in keeping
    with the other low-rank layers here."""
    def __init__(self, in_channels, out_channels, kernel_size, rank, stride=1,
        padding=0, dilation=1, groups=1, bias=False):
        assert groups == 1
        super(GenericLowRank, self).__init__()
        if kernel_size > 1:
            self.grouped = nn.Conv2d(in_channels, in_channels, kernel_size,
                    stride=stride, padding=padding, dilation=dilation,
                    groups=in_channels, bias=False)
            self.lowrank_contract = nn.Conv2d(in_channels, rank, 1, bias=False)
            self.lowrank_expand = nn.Conv2d(rank, out_channels, 1, bias=bias)
        else:
            self.grouped = None
            self.lowrank_contract = nn.Conv2d(in_channels, rank, 1, stride=stride,
                    dilation=dilation, bias=False)
            self.lowrank_expand = nn.Conv2d(rank, out_channels, 1, bias=bias)

    def forward(self, x):
        if self.grouped is not None:
            x = self.grouped(x)
        x = self.lowrank_contract(x)
        return self.lowrank_expand(x)


class LowRank(nn.Module):
    """A generic low rank layer implemented with a linear bottleneck, using two
    Conv2ds in sequence. Preceded by a depthwise grouped convolution in keeping
    with the other low-rank layers here."""
    def __init__(self, in_channels, out_channels, kernel_size, rank, stride=1,
        padding=0, dilation=1, groups=1, bias=False):
        assert groups == 1
        assert out_channels%in_channels == 0
        self.upsample = out_channels//in_channels
        super(LowRank, self).__init__()
        if kernel_size > 1:
            self.grouped = nn.Conv2d(in_channels, in_channels, kernel_size,
                    stride=stride, padding=padding, dilation=dilation,
                    groups=in_channels, bias=False)
            self.lowrank = nn.Conv2d(self.upsample*in_channels, rank, 1,
                    bias=bias)
        else:
            self.grouped = None
            self.lowrank = nn.Conv2d(self.upsample*in_channels, rank, 1,
                    stride=stride, dilation=dilation, bias=bias)

    def forward(self, x):
        if self.grouped is not None:
            x = self.grouped(x)
        if self.upsample > 1:
            x = x.repeat(1,self.upsample,1,1)
        x = F.conv2d(x, self.lowrank.weight, None, self.lowrank.stride,
                self.lowrank.padding, self.lowrank.dilation,
                self.lowrank.groups)
        return F.conv2d(x, self.lowrank.weight.permute(1,0,2,3),
                self.lowrank.bias)



# from: https://github.com/kuangliu/pytorch-cifar/blob/master/models/shufflenet.py#L10-L19
class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        return x.view(N,g,C//g,H,W).permute(0,2,1,3,4).contiguous().view(N,C,H,W)


class LinearShuffleNet(nn.Module):
    """Linear version of the ShuffleNet block, minus the shortcut connection,
    as we assume relevant shortcuts already exist in the network having a
    substitution. When linear, this can be viewed as a low-rank tensor
    decomposition."""
    def __init__(self, in_channels, out_channels, kernel_size, shuffle_groups,
            stride=1, padding=0, dilation=1, groups=1, bias=False):
        assert groups == 1
        super(LinearShuffleNet, self).__init__()
        # why 4? https://github.com/jaxony/ShuffleNet/blob/master/model.py#L67
        bottleneck_channels = out_channels // 4
        self.shuffle_gconv1 = nn.Conv2d(in_channels, bottleneck_channels, 1,
                groups=shuffle_groups, bias=False)
        self.shuffle = ShuffleBlock(shuffle_groups)
        self.shuffle_dwconv = nn.Conv2d(bottleneck_channels, bottleneck_channels,
                kernel_size, stride=stride, padding=padding, dilation=dilation,
                groups=bottleneck_channels, bias=False)
        self.shuffle_gconv2 = nn.Conv2d(bottleneck_channels, out_channels, 1,
                groups=shuffle_groups, bias=bias)

    def forward(self, x):
        x = self.shuffle_gconv1(x)
        x = self.shuffle(x)
        x = self.shuffle_dwconv(x)
        return self.shuffle_gconv2(x)


def cant_be_shuffled(shuffle_groups, in_channels, out_channels):
    # utility function, true if we can't instance shufflenet block using this
    divides_in = in_channels%shuffle_groups == 0
    divides_out = out_channels%shuffle_groups == 0
    divides_bottleneck = (out_channels//4)%shuffle_groups == 0
    return not (divides_in and divides_out and divides_bottleneck)


class DepthwiseSep(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=True):

        super(DepthwiseSep, self).__init__()
        assert groups == 1
        if kernel_size > 1:
            self.grouped = nn.Conv2d(in_channels, in_channels, kernel_size,
                    stride=stride, padding=padding, dilation=dilation,
                    groups=in_channels, bias=False)
            self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        else:
            self.pointwise = nn.Conv2d(in_channels, out_channels, 1,
                    stride=stride, padding=padding, dilation=dilation,
                    bias=bias)

    def forward(self, x):
        if hasattr(self, 'grouped'):
            out = self.grouped(x)
        else:
            out = x
        return self.pointwise(out)


class Conv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=1,
            dilation=1, bias=False):
        super(Conv, self).__init__()
        # Dumb normal conv incorporated into a class
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                stride=stride, padding=padding, bias=bias, dilation=dilation)

    def forward(self, x):
        return self.conv(x)


def conv_function(convtype):
    
    # if convtype contains an underscore, it must have a hyperparam in it
    if "_" in convtype:
        convtype, hyperparam = convtype.split("_")
        if convtype == 'ACDC':
            # then hyperparam controls how many layers in each conv
            n_layers = int(round(float(hyperparam)))
            def conv(in_channels, out_channels, kernel_size, stride=1,
                    padding=0, dilation=1, groups=1, bias=False):
                return FastStackedConvACDC(in_channels, out_channels,
                        kernel_size, n_layers, stride=stride,
                        padding=padding, dilation=dilation, groups=groups,
                        bias=bias)
        elif convtype == 'Hashed':
            # then hyperparam controls relative budget for each layer
            budget_scale = float(hyperparam)
            def conv(in_channels, out_channels, kernel_size, stride=1,
                    padding=0, dilation=1, groups=1, bias=False):
                # Hashed Conv2d using 1/10 the original parameters
                original_params = out_channels*in_channels*kernel_size*kernel_size // groups
                budget = int(original_params*budget_scale)
                return HashedConv2d(in_channels, out_channels, kernel_size,
                        budget, stride=stride, padding=padding,
                        dilation=dilation, groups=groups, bias=bias)
        elif convtype == 'SepHashed':
            # then hyperparam controls relative budget for each layer
            budget_scale = float(hyperparam)
            def conv(in_channels, out_channels, kernel_size, stride=1,
                    padding=0, dilation=1, groups=1, bias=False):
                original_params = out_channels*in_channels // groups
                budget = int(original_params*budget_scale)
                if kernel_size > 1: # budget for a grouped convolution
                    budget += in_channels*kernel_size*kernel_size
                return HalfHashedSeparable(in_channels, out_channels, kernel_size,
                        budget, stride=stride, padding=padding,
                        dilation=dilation, groups=groups, bias=bias)
        elif convtype == 'Generic':
            rank_scale = float(hyperparam)
            def conv(in_channels, out_channels, kernel_size, stride=1,
                    padding=0, dilation=1, groups=1, bias=False):
                full_rank = max(in_channels,out_channels)
                rank = int(rank_scale*full_rank)
                return GenericLowRank(in_channels, out_channels, kernel_size,
                        rank, stride=stride, padding=padding,
                        dilation=dilation, groups=groups, bias=bias)
        elif convtype == 'LR':
            rank_scale = float(hyperparam)
            def conv(in_channels, out_channels, kernel_size, stride=1,
                    padding=0, dilation=1, groups=1, bias=False):
                full_rank = max(in_channels,out_channels)
                rank = int(rank_scale*full_rank)
                return LowRank(in_channels, out_channels, kernel_size,
                        rank, stride=stride, padding=padding,
                        dilation=dilation, groups=groups, bias=bias)
        elif convtype == 'TensorTrain':
            rank_scale = float(hyperparam)
            def conv(in_channels, out_channels, kernel_size, stride=1,
                    padding=0, dilation=1, groups=1, bias=False):
                return TensorTrain(in_channels, out_channels, kernel_size,
                        rank_scale, 3, stride=stride, padding=padding,
                        dilation=dilation, groups=groups, bias=bias)
        elif convtype == 'Tucker':
            rank_scale = float(hyperparam)
            def conv(in_channels, out_channels, kernel_size, stride=1,
                    padding=0, dilation=1, groups=1, bias=False):
                return Tucker(in_channels, out_channels, kernel_size,
                        rank_scale, 3, stride=stride, padding=padding,
                        dilation=dilation, groups=groups, bias=bias)
        elif convtype == 'CP':
            assert False, "Deprecated"
            rank_scale = float(hyperparam)
            def conv(in_channels, out_channels, kernel_size, stride=1,
                    padding=0, dilation=1, groups=1, bias=False):
                return CP(in_channels, out_channels, kernel_size,
                        rank_scale, stride=stride, padding=padding,
                        dilation=dilation, groups=groups, bias=bias)
        elif convtype == 'Shuffle':
            def conv(in_channels, out_channels, kernel_size, stride=1,
                    padding=0, dilation=1, groups=1, bias=False):
                shuffle_groups = int(hyperparam)
                while cant_be_shuffled(shuffle_groups, in_channels, out_channels):
                    shuffle_groups += -1
                return LinearShuffleNet(in_channels, out_channels, kernel_size,
                        shuffle_groups, stride=stride, padding=padding,
                        dilation=dilation, groups=groups, bias=bias)
    else:
        if convtype == 'Conv':
            conv = Conv
        elif convtype =='ACDC':
            conv = ACDC
        elif convtype =='OriginalACDC':
            conv = OriginalACDC
        elif convtype == 'HashedDecimate':
            conv = HashedDecimate
        elif convtype == 'SepHashedDecimate':
            conv = SepHashedDecimate
        elif convtype == 'Sep':
            conv = DepthwiseSep
        else:
            raise ValueError('Conv "%s" not recognised'%convtype)
    return conv


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, conv=Conv):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        #assert self.conv2.grouped.padding[0] == 1
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


# modified from torchvision
class Bottleneck(nn.Module):
    """Bottleneck architecture block for ResNet"""
    def __init__(self, inplanes, planes, ConvClass, stride=1, downsample=None, expansion=4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        pointwise = lambda i,o: ConvClass(i, o, kernel_size=1, padding=0,
                bias=False)
        self.conv1 = pointwise(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = ConvClass(planes, planes, kernel_size=3, stride=stride,
                padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = pointwise(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def add_residual(self, x, out):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x
        return out + residual

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        #out = checkpoint(self.add_residual, x, out)
        out = self.add_residual(x, out)
        out = self.relu(out)

        return out


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, conv = Conv):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, conv)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, conv):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, conv))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

if __name__ == '__main__':
    X = torch.randn(5,16,32,32)
    # sanity of generic low-rank layer
    generic = GenericLowRank(16, 32, 3, 2)
    for n,p in generic.named_parameters():
        print(n, p.size(), p.numel())
    out = generic(X)
    print(out.size())
    low = LowRank(16, 32, 3, 2)
    for n, p in low.named_parameters():
        print(n, p.size(), p.numel())
    out = low(X)
    print(out.size())
    assert False
    # check we don't initialise a grouped conv when not required
    layers_to_test = [LowRank(3,32,1,1), GenericLowRank(3,32,1,1),
            HalfHashedSeparable(3,32,1,10), TensorTrain(3,32,1,0.5,3),
            Tucker(3,32,1,0.5,3), CP(3,32,1,0.5,3), ACDC(3,32,1)]
    for layer in layers_to_test:
        assert getattr(layer, 'grouped', None) is None
    # and we *do* when it is required
    layers_to_test = [LowRank(3,32,3,1), GenericLowRank(3,32,3,1),
            HalfHashedSeparable(3,32,3,100), TensorTrain(3,32,3,0.5,3),
            Tucker(3,32,3,0.5,3), CP(3,32,3,0.5,3), ACDC(3,32,3)]
    for layer in layers_to_test:
        assert getattr(layer, 'grouped', None) is not None, layer
    # sanity of LinearShuffleNet
    X = torch.randn(5,16,32,32)
    shuffle = LinearShuffleNet(16,32,3,4)
    print(shuffle(X).size())
