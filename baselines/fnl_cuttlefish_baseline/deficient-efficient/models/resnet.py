'''This is a rewriting of the native resnet definition that comes with Pytorch, to allow it to use our blocks and
 convolutions for imagenet experiments. Annoyingly, the pre-trained models don't use pre-activation blocks.'''

import torch
import torch.nn as nn
import math
import torchvision.models.resnet
import torch.utils.model_zoo as model_zoo
from .blocks import *


__all__ = ['ResNet', 'resnet18', 'resnet34']#, 'resnet50', 'resnet101','resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class ResNet(nn.Module):

    def __init__(self, conv, block, n, num_classes=1000, s=1):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        nChannels =[64, 64, 128, 256, 512]

        self.layer1 = torch.nn.ModuleList()
        for i in range(s):
            self.layer1.append(NetworkBlock(int(n[0] // s), nChannels[0] if i == 0 else nChannels[1],
                                            nChannels[1], block, 1, conv=conv))


        self.layer2 = torch.nn.ModuleList()
        for i in range(s):
            self.layer2.append(NetworkBlock(int(n[1] // s), nChannels[1] if i == 0 else nChannels[2],
                                            nChannels[2], block, 2, conv=conv))

        self.layer3 = torch.nn.ModuleList()
        for i in range(s):
            self.layer3.append(NetworkBlock(int(n[2] // s), nChannels[2] if i == 0 else nChannels[3],
                                            nChannels[3], block, 2, conv=conv))

        self.layer4 = torch.nn.ModuleList()
        for i in range(s):
            self.layer4.append(NetworkBlock(int(n[3] // s), nChannels[3] if i == 0 else nChannels[4],
                                            nChannels[4], block, 2, conv=conv))

        # self.layer1 = self._make_layer(block, 64, layers[0])
        # self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        activations = []
        out = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        for sub_block in self.layer1:
            out = sub_block(out)
            activations.append(out)

        for sub_block in self.layer2:
            out = sub_block(out)
            activations.append(out)

        for sub_block in self.layer3:
            out = sub_block(out)
            activations.append(out)

        for sub_block in self.layer4:
            out = sub_block(out)
            activations.append(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out, activations



def resnet18(pretrained=False, conv=nnConv, block=OldBlock):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(conv,block, [2, 2, 2, 2])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, conv=nnConv, block=OldBlock):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(conv,block, [3, 4, 6, 3])
    if pretrained:
        old_model = torchvision.models.resnet.resnet34(pretrained=False)
        old_model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))


        new_state_dict = model.state_dict()
        old_state_dict = old_model.state_dict()

        # This assumes the sequence of each module in the network is the same in both cases.
        # Ridiculously, batch norm params are stored in a different sequence in the downloaded state dict, so we have to
        # load the old model definition, load in its downloaded state dict to change the order back, then transfer this!

        old_model = torchvision.models.resnet.resnet34(pretrained=False)
        old_model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))

        old_names = [v for v in old_state_dict]
        new_names = [v for v in new_state_dict]

        for i,j in enumerate(old_names):
            new_state_dict[new_names[i]] = old_state_dict[j]

        model.load_state_dict(new_state_dict)

    return model


def test2():
    net = resnet34()
    x = torch.randn(1, 3, 224, 224)
    y, _ = net(Variable(x))
    print(y.size())

if __name__ == '__main__':
    test2()


        # Haven't written the old bottleneck yet.

#
# def resnet50(pretrained=False, **kwargs):
#     """Constructs a ResNet-50 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
#     return model
#
#
# def resnet101(pretrained=False, **kwargs):
#     """Constructs a ResNet-101 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
#     return model
#
#
# def resnet152(pretrained=False, **kwargs):
#     """Constructs a ResNet-152 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
#     return model
