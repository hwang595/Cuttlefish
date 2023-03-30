'''Count parameters or mult-adds in models.'''
from __future__ import print_function
import math
import torch
import argparse
from torch.autograd import Variable
from models.wide_resnet import WideResNet, WRN_50_2
from models.darts import DARTS
from models.MobileNetV2 import MobileNetV2

from funcs import what_conv_block


ignored_modules = []


def get_layer_info(layer):
    layer_str = str(layer)
    type_name = layer_str[:layer_str.find('(')].strip()
    return type_name

def get_layer_param(model):
    return sum([p.numel() for p in model.parameters()])

class OpCounter(object):
    def __init__(self):
        self.count_ops = 0
        self.count_params = 0

    def measure_layer(self, layer, x):
        delta_ops = 0
        delta_params = 0
        multi_add = 1
        type_name = get_layer_info(layer)
        x = x[0]

        ### ops_conv
        if type_name in ['Conv2d']:
            out = layer.old_forward(x)
            out_h = out.size(2)
            out_w = out.size(3)
            delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                    layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
            delta_params = get_layer_param(layer)

        ### ops_nonlinearity
        elif type_name in ['ReLU']:
            delta_ops = x.numel()
            delta_params = get_layer_param(layer)

        ### ops_pooling
        elif type_name in ['AvgPool2d','MaxPool2d']:
            in_w = x.size()[2]
            if type(layer.kernel_size) is tuple:
                k = layer.kernel_size[0]
            else:
                k = layer.kernel_size
            kernel_ops = k * k
            out_w = int((in_w + 2 * layer.padding - k) / layer.stride + 1)
            out_h = int((in_w + 2 * layer.padding - k) / layer.stride + 1)
            out = layer.old_forward(x)
            assert out_h == out.size(2)
            assert out_w == out.size(3)
            delta_ops = x.size()[0] * x.size()[1] * out_w * out_h * kernel_ops
            delta_params = get_layer_param(layer)

        ### ops_linear
        elif type_name in ['Linear']:
            weight_ops = layer.weight.numel() * multi_add
            bias_ops = layer.bias.numel()
            delta_ops = x.size()[0] * (weight_ops + bias_ops)
            delta_params = get_layer_param(layer)

        ### ops_nothing
        elif type_name in ['BatchNorm2d', 'Dropout2d', 'DropChannel', 'Dropout']:
            delta_params = get_layer_param(layer)

        ### sequential takes no extra time
        elif type_name in ['Sequential']:
            pass
        
        ### riffle shuffle
        elif type_name in ['Riffle']:
            # technically no floating point operations
            pass

        ### channel expansion
        elif type_name in ['ChannelExpand']:
            # assume concatentation doesn't take extra FLOPs
            pass

        ### channel contraction
        elif type_name in ['ChannelCollapse']:
            # do as many additions as we have channels
            delta_ops += x.size(1)

        ### ACDC Convolution
        elif type_name in ['FastStackedConvACDC']:
            out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                        layer.stride[0] + 1)
            out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                        layer.stride[1] + 1)       
            assert layer.groups == 1
            # pretend we're actually passing through the ACDC layers within
            N = max(layer.out_channels, layer.in_channels) # size of ACDC layers
            acdc_ops = 0
            for l in layer.layers:
                acdc_ops += 4*N + 5*N*math.log(N,2)
                delta_params += 2*N
            delta_ops += acdc_ops*out_h*out_w


        ### Grouped ACDC Convolution
        elif type_name in ['GroupedConvACDC']:
            assert False
            out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                        layer.stride[0] + 1)
            out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                        layer.stride[1] + 1)       
            # pretend we're actually passing through the ACDC layers within
            N = layer.kernel_size[0]
            acdc_ops = layer.groups*(4*N + 5*N*math.log(N,2))
            conv_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                       layer.kernel_size[1]  / layer.groups
            ops = min(acdc_ops, conv_ops)
            delta_ops += ops*out_h*out_w
            delta_params += 2*N

        ### HashedNet Convolution
        elif type_name in ['HashedConv2d']:
            # same number of ops as convolution
            out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                        layer.stride[0] + 1)
            out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                        layer.stride[1] + 1)
            delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                    layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
            delta_params = get_layer_param(layer)

        elif type_name in ['DepthwiseSep']:
            # wrapper for Conv2Ds, that are counted above
            pass

        elif type_name in ['TensorTrain', 'Tucker']:
            if hasattr(layer, 'grouped'):
                out = layer.grouped.old_forward(x)
                out_h = out.size(2)
                out_w = out.size(3)
            else:
                out = layer.old_forward(x)
                out_h = out.size(2)
                out_w = out.size(3)
            delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                    layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
            delta_params = sum([p.numel() for k,p in layer._parameters.items() if p is not None])

        elif type_name in ['LinearShuffleNet', 'ShuffleBlock']:
            # all operations implemented by internal conv2d, so this can be ignored
            pass

        elif type_name in ['GenericLowRank']:
            # all operations implemented by internal conv2d
            pass

        elif type_name in ['LowRank']:
            if hasattr(layer, 'grouped'):
                x = layer.grouped.old_forward(x)
            if layer.upsample > 1:
                x = x.repeat(1,layer.upsample,1,1)
            out = layer.lowrank.old_forward(x)
            out_h = out.size(2)
            out_w = out.size(3)
            delta_ops = 2 * layer.lowrank.in_channels * \
                layer.lowrank.out_channels * layer.lowrank.kernel_size[0] * \
                layer.lowrank.kernel_size[1] * out_h * out_w / layer.lowrank.groups \
                * multi_add
            delta_params = get_layer_param(layer.lowrank)

        #elif type_name in ['TensorTrain']:
        elif False:
            # number of cores
            d = 0
            while hasattr(layer, 'weight_core_%i'%d):
                core = getattr(layer, 'weight_core_%i'%d)
                d += 1
            d += 1
            # max dimension
            m = max(layer.tn_weight.torch().size())
            # maximal rank
            r = max(layer.tn_weight.ranks_tt)
            # max dim of kernel matrix
            maxMN = max(layer.in_channels, layer.out_channels)
            # complexity
            c = d*r*r*m*maxMN + d*r*r*r*m
            print(layer.in_channels*layer.out_channels, c)
            import ipdb
            ipdb.set_trace()
            # number of Us
            n_us = 0
            while hasattr(layer, 'weight_u_%i'%n_us):
                u = getattr(layer, 'weight_u_%i'%n_us)
                n_us += 1
            if type_name == 'TensorTrain':
                # From "Tensorizing Neural Networks"
                #   For the case of the TT-matrix-by-explicit-vector product c = Wb,
                #   the computational complexity is O(d r^2 m max(M,N)), where d is
                #   the number of cores of the TT-matrix W, m is the max_k m_k, r is
                #   the maximal rank and N = \prod_k=1^d n_k is the length of the
                #   vector b.
                #
                # Seems like, naively, the mult-adds can be estimated as those used
                # by an independent matrix multiply for each core, with the result
                # then summed. Reading this from Section 4.
                d = n_cores
                r = layer.rank
                N = x.size(1)


            # plus the ops of the grouped convolution? or does that get caught anyway?
            #assert False
            # this would double count the grouped
            #delta_params = get_layer_param(layer) 

        ### unknown layer type
        else:
            if type_name not in ignored_modules:
                ignored_modules.append(type_name)
            #raise TypeError('unknown layer type: %s' % type_name)

        self.count_ops += delta_ops
        self.count_params += delta_params
        return None

def measure_model(model, H, W):
    opcount = OpCounter()
    data = Variable(torch.zeros(1, 3, H, W))

    def modify_forward(model):
        for child in model.modules():
            def new_forward(m):
                def lambda_forward(*x):
                    opcount.measure_layer(m, x)
                    try:
                        return m.old_forward(*x)
                    except NotImplementedError as e:
                        print(m)
                        raise e
                return lambda_forward
            child.old_forward = child.forward
            child.forward = new_forward(child)

    # recursive function
    def restore_forward(model):
        for child in model.children():
            # leaf node
            if is_leaf(child) and hasattr(child, 'old_forward'):
                child.forward = child.old_forward
                child.old_forward = None
            else:
                restore_forward(child)

    modify_forward(model)
    model.forward(data)
    #restore_forward(model)

    return opcount.count_ops, opcount.count_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WRN parameter/flop usage')
    parser.add_argument('dataset', type=str, choices=['cifar10', 'cifar100', 'imagenet'], help='Choose between Cifar10/100/imagenet.')

    #network stuff
    parser.add_argument('--network', default='WideResNet', type=str, help='network to use')
    parser.add_argument('--wrn_depth', default=40, type=int, help='depth for WRN')
    parser.add_argument('--wrn_width', default=2, type=float, help='width for WRN')
    parser.add_argument('--module', default=None, type=str, help='path to file containing custom Conv and maybe Block module definitions')
    parser.add_argument('--blocktype', default='Basic',type=str, help='blocktype used if specify a --conv')
    parser.add_argument('--conv', default=None, type=str, help='Conv type')

    args = parser.parse_args()

    # Stuff happens from here:
    Conv, Block = what_conv_block(args.conv, args.blocktype, args.module)

    if args.dataset == 'cifar10':
        h,w = 32,32
        num_classes = 10
    elif args.dataset == 'cifar100':
        h,w = 32, 32
        num_classes = 100
    elif args.dataset == 'imagenet':
        h,w = 224, 224
        num_classes = 1000
    else:
        raise ValueError(args.dataset)

    # instance the model
    def build_network(Conv, Block):
        if args.network == 'WideResNet':
            return WideResNet(args.wrn_depth, args.wrn_width, Conv, Block,
                    num_classes=num_classes, dropRate=0)
        elif args.network == 'WRN_50_2':
            return WRN_50_2(Conv)
        elif args.network == 'DARTS':
            assert not args.conv == 'Conv', 'The base network here used' \
            ' separable convolutions, so you probably did not mean to set this' \
            ' option.'
            return DARTS(Conv, num_classes=num_classes, drop_path_prob=0., auxiliary=False)
        elif args.network == 'MobileNetV2':
            return MobileNetV2(Conv)
    model = build_network(Conv, Block)

    # count how many parameters are in it
    flops, params = measure_model(model, h, w)
    print("Mult-Adds: %.5E"%flops)
    print("Params: %.5E"%params)
    sanity = sum([p.numel() for p in model.parameters()])
    assert sanity == params, "Sanity check, parameters: %.5E =/= %.5E \n %s"%(sanity, params, str(ignored_modules))
    print(ignored_modules)
    #import time
    #for m in model.modules():
    #    time.sleep(0.2)
    #    print(get_layer_info(m), sum([p.numel() for p in m.parameters()]))
