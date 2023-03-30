import torch
import torch.nn as nn
import math


# wildcard import for legacy reasons
if __name__ == '__main__':
    import sys
    sys.path.append("..")
from models.blocks import *
from models.wide_resnet import compression, group_lowrank

# only used in the first convolution, which we do not substitute by convention
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

# only used for final fully connectec layers
def conv_1x1_bn(inp, oup, ConvClass):
    return nn.Sequential(
        ConvClass(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, ConvClass):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.Conv = ConvClass
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                self.Conv(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                self.Conv(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                self.Conv(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, ConvClass, block=None, n_class=1000,
            input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        self.kwargs = dict(ConvClass=ConvClass, block=block, n_class=n_class,
                input_size=input_size, width_mult=width_mult)
        block = InvertedResidual
        self.Conv = ConvClass
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t, ConvClass=self.Conv))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t, ConvClass=self.Conv))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel, self.Conv))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier_conv = self.Conv(self.last_channel, n_class, 1, 1, 0, bias=True)
        #self.classifier = \
            #nn.Dropout(0.2), remove dropout for training according to github
        #    nn.(self.last_channel, n_class),
        #)

        self._initialize_weights()

    def classifier(self, x):
        n, c = x.size()
        x = self.classifier_conv(x.view(n,c,1,1))
        n, c, _, _ = x.size()
        return x.view(n,c)

    def forward(self, x):
        #y_orig = self.features(x)
        attention_maps = []
        attention = lambda x: F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

        y = x 
        for block in self.features:
            y = block(y)
            if isinstance(block, InvertedResidual):
                if block.stride > 1:
                    attention_maps.append(attention(y))
        
        #error = torch.abs(y-y_orig).max() 
        #assert error < 1e-2, f"Error {error} above 0.01"
        x = y
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x, attention_maps

    def compression_ratio(self):
        return compression(self.__class__, self.kwargs)

    def grouped_parameters(self, weight_decay):
        return group_lowrank(self.named_parameters(), weight_decay,
                self.compression_ratio())

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                if hasattr(m, 'weight'):
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def save_reference():
    net = MobileNetV2()
    net.eval()
    x = torch.randn(1,3,224,224).float()
    y = net(x)
    print(y.size())
    torch.save(x, "reference_input_mobilenet.torch")
    torch.save(y, "reference_output_mobilenet.torch")
    torch.save(net.state_dict(), "reference_state_mobilenet.torch")

def match_keys(net, state):
    nstate = net.state_dict()
    old_keys = [k for k in state]
    for i, k in enumerate(nstate):
        p = state[old_keys[i]]
        if i == (len(old_keys)-2):
            n,m = p.size() 
            nstate[k] = p.view(n,m,1,1)
        else:
            nstate[k] = p
    return nstate

def test():
    import os
    net = MobileNetV2(Conv)
    if os.path.exists("reference_state_mobilenet.torch"):
        state = torch.load("reference_state_mobilenet.torch")
        state = match_keys(net, state)
        net.load_state_dict(state)
        net.eval()
        x = torch.load("reference_input_mobilenet.torch")
    else:
        x = torch.randn(1,3,224,224).float()
    y, _ = net(Variable(x))
    print(y.size())
    # check if these match the test weights
    if os.path.exists("reference_output_mobilenet.torch"):
        ref_output = torch.load("reference_output_mobilenet.torch")
        error = torch.abs(ref_output - y).max()
        print(f"Error: {error}, Max logit: {y.max()}/{ref_output.max()}, Min logit: {y.min()}/{ref_output.min()}")
    state = {
        'net': net.state_dict(),
        'epoch': 150,
        'args': None,
        'width': None,
        'depth': None,
        'conv': 'Conv',
        'blocktype': None,
        'module': None,
        'train_losses': None,
        'train_errors': None,
        'val_losses': None,
        'val_errors': [28.2],
    }
    torch.save(state, "mobilenetv2.tonylins.t7")

def test_compression():
    net = MobileNetV2(Conv)
    #net = MobileNetV2(conv_function('Hashed_0.1'))
    nparams = lambda x: sum([p.numel() for p in x.parameters()])
    for block in net.features:
        print(nparams(block))
    for x in block:
        print(x)
        print(nparams(x))
    #CompressedConv = conv_function("Hashed_0.1")
    for conv in ['Shuffle_%i'%i for i in [4,8,16,32]]+['Hashed_0.01']:
        print(conv)
        CompressedConv = conv_function(conv)
        net = MobileNetV2(CompressedConv)
        print("  ", net.compression_ratio())

if __name__ == '__main__':
    test()
    #test_compression()
