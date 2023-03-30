'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from .layers import BinConv2d


def uniform(w):
    if isinstance(w, torch.nn.BatchNorm2d):
        w.weight.data = torch.rand(w.weight.data.shape)
        w.bias.data = torch.zeros_like(w.bias.data)

def kaiming_normal(w):
    if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(w.weight)


def param_counter(model):
    param_counter = 0
    for p_index, (p_name, p) in enumerate(model.named_parameters()):
        param_counter += p.numel()
    return param_counter



class VGG19(nn.Module):
    '''
    FullRankVGG19 Model 
    '''
    def __init__(self, num_classes=10):
        super(VGG19, self).__init__()
        # based on the literature, we don't touch the first conv layer
        self.conv1 = nn.Conv2d(3, 64, 3, 1, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)

        self.block1 = nn.Sequential(
                            BinConv2d(64, 64, 3, 1, padding=1, bias=False),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=2, stride=2),
                            BinConv2d(64, 128, 3, 1, padding=1, bias=False),
                            nn.ReLU(inplace=True),
                            BinConv2d(128, 128, 3, 1, padding=1, bias=False),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=2, stride=2),
                        ) 
        
        self.block2 = nn.Sequential(
                            BinConv2d(128, 256, 3, 1, padding=1, bias=False),
                            nn.ReLU(inplace=True),
                            BinConv2d(256, 256, 3, 1, padding=1, bias=False),
                            nn.ReLU(inplace=True),
                            BinConv2d(256, 256, 3, 1, padding=1, bias=False),
                            nn.ReLU(inplace=True),
                            BinConv2d(256, 256, 3, 1, padding=1, bias=False),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=2, stride=2)
                        )
        
        self.block3 = nn.Sequential(
                            BinConv2d(256, 512, 3, 1, padding=1, bias=False),
                            nn.ReLU(inplace=True),
                            BinConv2d(512, 512, 3, 1, padding=1, bias=False),
                            nn.ReLU(inplace=True),
                            BinConv2d(512, 512, 3, 1, padding=1, bias=False),
                            nn.ReLU(inplace=True),
                            BinConv2d(512, 512, 3, 1, padding=1, bias=False),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=2, stride=2),
                            BinConv2d(512, 512, 3, 1, padding=1, bias=False),
                            nn.ReLU(inplace=True),
                            BinConv2d(512, 512, 3, 1, padding=1, bias=False),
                            nn.ReLU(inplace=True),
                            BinConv2d(512, 512, 3, 1, padding=1, bias=False),
                            nn.ReLU(inplace=True),
                            BinConv2d(512, 512, 3, 1, padding=1, bias=False),
                            nn.ReLU(inplace=True),
                            nn.AvgPool2d(2)
                        )

        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1.0)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.conv1(x)))
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



if __name__ == "__main__":
    #net = LowrankVGG19LTH()
    lr_layers = ["conv9", "classifier.1.", "classifier.2."]
    net = LowRankVGG19Dynamic(rank_list=[i for i in range(5, 16)], 
                            low_rank_layers=lr_layers)
    print("#### Model arch: {}, num params: {}".format(net, param_counter(net)))