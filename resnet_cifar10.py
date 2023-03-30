'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


CONST_RANK_DENOMINATOR=4


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class LowrankBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, rank_ratio=None):
        super(LowrankBasicBlock, self).__init__()
        self.conv1_u = nn.Conv2d(in_planes, int(planes/rank_ratio), 
                        kernel_size=3, 
                        stride=stride, 
                        padding=1, 
                        bias=False)
        self.bn1_u = nn.BatchNorm2d(int(planes/rank_ratio))
        self.conv1_v = conv1x1(int(planes/rank_ratio), planes)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2_u = nn.Conv2d(planes, int(planes/rank_ratio), 
                                kernel_size=3, 
                                stride=1,
                                padding=1, 
                                bias=False)
        self.bn2_u = nn.BatchNorm2d(int(planes/rank_ratio))
        self.conv2_v = conv1x1(int(planes/rank_ratio), planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1_v(self.bn1_u(self.conv1_u(x)))))
        out = self.bn2(self.conv2_v(self.bn2_u(self.conv2_u(out))))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class LowrankBasicBlockAdapt(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, rank_list, frob_decay, extra_bns, stride=1):
        super(LowrankBasicBlockAdapt, self).__init__()
        self._frob_decay = frob_decay
        self._extra_bns = extra_bns
        
        self.conv1_u = nn.Conv2d(in_planes, rank_list[0], 
                        kernel_size=3, 
                        stride=stride, 
                        padding=1, 
                        bias=False)

        self.conv1_v = conv1x1(rank_list[0], planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2_u = nn.Conv2d(planes, rank_list[1], 
                                kernel_size=3, 
                                stride=1,
                                padding=1, 
                                bias=False)
        self.conv2_v = conv1x1(rank_list[1], planes)
        self.bn2 = nn.BatchNorm2d(planes)

        if not self._frob_decay and self._extra_bns:
            self.bn1_u = nn.BatchNorm2d(rank_list[0])
            self.bn2_u = nn.BatchNorm2d(rank_list[1])

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        # the methodology here is that if we enable FD, we need to switch of extra BN
        # otherwise, we enable extra BN
        if self._frob_decay or not self._extra_bns:
            out = F.relu(self.bn1(self.conv1_v(self.conv1_u(x))))
            out = self.bn2(self.conv2_v(self.conv2_u(out)))
        else:
            out = F.relu(self.bn1(self.conv1_v(self.bn1_u(self.conv1_u(x)))))
            out = self.bn2(self.conv2_v(self.bn2_u(self.conv2_u(out))))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BaselineBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BaselineBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        # if stride != 1 or in_planes != self.expansion*planes:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(self.expansion*planes)
        #     )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        #out += self.shortcut(x)
        out = F.relu(out)
        return out


class LowRankBottleneckConv1x1(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(LowRankBottleneckConv1x1, self).__init__()
        self.conv1_u = nn.Conv2d(in_planes, int(planes/CONST_RANK_DENOMINATOR), kernel_size=1, bias=False)
        self.conv1_v = conv1x1(int(planes/CONST_RANK_DENOMINATOR), planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2_u = nn.Conv2d(planes, int(planes/CONST_RANK_DENOMINATOR), kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2_v = conv1x1(int(planes/CONST_RANK_DENOMINATOR), planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3_u = nn.Conv2d(planes, int(self.expansion*planes/CONST_RANK_DENOMINATOR), kernel_size=1, bias=False)
        self.conv3_v = conv1x1(int(self.expansion*planes/CONST_RANK_DENOMINATOR), self.expansion*planes)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1_v(self.conv1_u(x))))
        out = F.relu(self.bn2(self.conv2_v(self.conv2_u(out))))
        out = self.bn3(self.conv3_v(self.conv3_u(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class LowRankBasicBlockResidual(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(LowRankBasicBlockResidual, self).__init__()
        #self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1_u = nn.Conv2d(in_planes, int(planes/CONST_RANK_DENOMINATOR), 
                            kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1_v = conv1x1(int(planes/CONST_RANK_DENOMINATOR), planes)
        self.conv1_res = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_u = nn.Conv2d(planes, int(planes/CONST_RANK_DENOMINATOR), kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_v = conv1x1(int(planes/CONST_RANK_DENOMINATOR), planes)
        self.conv2_res = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1_v(self.conv1_u(x))+self.conv1_res(x)))
        out = self.bn2(self.conv2_v(self.conv2_u(out))+self.conv2_res(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class LowRankBasicBlockLowRankResidual(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(LowRankBasicBlockLowRankResidual, self).__init__()
        #self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # first low rank conv layer
        self.conv1_u = nn.Conv2d(in_planes, int(planes/CONST_RANK_DENOMINATOR), 
                            kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1_v = conv1x1(int(planes/CONST_RANK_DENOMINATOR), planes)
        # residual part
        self.conv1_res_u = nn.Conv2d(in_planes, int(planes/CONST_RANK_DENOMINATOR), 
                            kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1_res_v = conv1x1(int(planes/CONST_RANK_DENOMINATOR), planes)
        self.bn1 = nn.BatchNorm2d(planes)

        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_u = nn.Conv2d(planes, int(planes/CONST_RANK_DENOMINATOR), kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_v = conv1x1(int(planes/CONST_RANK_DENOMINATOR), planes)
        self.conv2_res_u = nn.Conv2d(planes, int(planes/CONST_RANK_DENOMINATOR),
                             kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_res_v = conv1x1(int(planes/CONST_RANK_DENOMINATOR), planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1_v(self.conv1_u(x))+self.conv1_res_v(self.conv1_res_u(x))))
        out = self.bn2(self.conv2_v(self.conv2_u(out))+self.conv2_res_v(self.conv2_res_u(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BaselineResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(BaselineResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BaselineBasicBlock, int(64/CONST_RANK_DENOMINATOR), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(128/CONST_RANK_DENOMINATOR), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(256/CONST_RANK_DENOMINATOR), num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, int(512/CONST_RANK_DENOMINATOR), num_blocks[3], stride=2)
        self.linear = nn.Linear(int(512/CONST_RANK_DENOMINATOR)*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class BasicBlockLR(nn.Module):
    # method from the paper: https://arxiv.org/pdf/1511.06744.pdf
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockLR, self).__init__()
        self.conv1_u = nn.Conv2d(in_planes, planes, kernel_size=(3, 1), stride=stride, padding=1, bias=False)
        #self.conv1_v = nn.Conv2d(in_planes, planes, kernel_size=(1, 3), stride=stride, padding=1, bias=False)
        self.conv1_v = nn.Conv2d(planes, planes, kernel_size=(1, 3), stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2_u = nn.Conv2d(planes, planes, kernel_size=(3, 1), stride=1, padding=1, bias=False)
        self.conv2_v = nn.Conv2d(planes, planes, kernel_size=(1, 3), stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1_v(self.conv1_u(x))))
        out = self.bn2(self.conv2_v(self.conv2_u(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        #out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class HybridResNet(nn.Module):
    def __init__(self, fullrank_block, lowrank_block, num_blocks, rank_ratio, num_classes=10):
        super(HybridResNet, self).__init__()
        self.in_planes = 64
        self._block_counter = 0

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(fullrank_block, lowrank_block, 64, num_blocks[0], stride=1, rank_ratio=rank_ratio)
        self.layer2 = self._make_layer(fullrank_block, lowrank_block, 128, num_blocks[1], stride=2, rank_ratio=rank_ratio)
        self.layer3 = self._make_layer(fullrank_block, lowrank_block, 256, num_blocks[2], stride=2, rank_ratio=rank_ratio)
        self.layer4 = self._make_layer(fullrank_block, lowrank_block, 512, num_blocks[3], stride=2, rank_ratio=rank_ratio)
        assert fullrank_block.expansion == lowrank_block.expansion
        self.linear = nn.Linear(512*fullrank_block.expansion, num_classes)

    def _make_layer(self, fullrank_block, lowrank_block, planes, num_blocks, stride, rank_ratio):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            if self._block_counter < 4:
                layers.append(fullrank_block(self.in_planes, planes, stride))
                self.in_planes = planes * fullrank_block.expansion
            else:
                layers.append(lowrank_block(self.in_planes, planes, stride, rank_ratio=rank_ratio))
                self.in_planes = planes * lowrank_block.expansion
            self._block_counter += 1
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        #out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class PufferfishResNet(nn.Module):
    """
    hybrid resnet18 arch used in Pufferfish (wang et al, MLSys 2021)
    paper link: https://proceedings.mlsys.org/paper/2021/file/84d9ee44e457ddef7f2c4f25dc8fa865-Paper.pdf
    """
    def __init__(self, fullrank_block, lowrank_block, num_blocks, rank_ratio, num_classes=10):
        super(PufferfishResNet, self).__init__()
        self.in_planes = 64
        self._block_counter = 0

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(fullrank_block, lowrank_block, 64, num_blocks[0], stride=1, rank_ratio=rank_ratio)
        self.layer2 = self._make_layer(fullrank_block, lowrank_block, 128, num_blocks[1], stride=2, rank_ratio=rank_ratio)
        self.layer3 = self._make_layer(fullrank_block, lowrank_block, 256, num_blocks[2], stride=2, rank_ratio=rank_ratio)
        self.layer4 = self._make_layer(fullrank_block, lowrank_block, 512, num_blocks[3], stride=2, rank_ratio=rank_ratio)
        assert fullrank_block.expansion == lowrank_block.expansion
        self.linear = nn.Linear(512*fullrank_block.expansion, num_classes)

    def _make_layer(self, fullrank_block, lowrank_block, planes, num_blocks, stride, rank_ratio):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            if self._block_counter < 1:
                layers.append(fullrank_block(self.in_planes, planes, stride))
                self.in_planes = planes * fullrank_block.expansion
            else:
                layers.append(lowrank_block(self.in_planes, planes, stride, rank_ratio=rank_ratio))
                self.in_planes = planes * lowrank_block.expansion
            self._block_counter += 1
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        #out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class HybridResNetAdapt(nn.Module):
    def __init__(self, fullrank_block, lowrank_block, num_blocks, rank_list, num_classes=10, 
                                    frob_decay=False, extra_bns=False):
        super(HybridResNetAdapt, self).__init__()
        self.in_planes = 64
        self._block_counter = 0
        self._rank_counter = 0
        self._frob_decay = frob_decay
        self._extra_bns = extra_bns

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(fullrank_block, lowrank_block, 64, num_blocks[0], 
                                        stride=1, rank_list=rank_list)
        self.layer2 = self._make_layer(fullrank_block, lowrank_block, 128, num_blocks[1], 
                                        stride=2, rank_list=rank_list)
        self.layer3 = self._make_layer(fullrank_block, lowrank_block, 256, num_blocks[2], 
                                        stride=2, rank_list=rank_list)
        self.layer4 = self._make_layer(fullrank_block, lowrank_block, 512, num_blocks[3], 
                                        stride=2, rank_list=rank_list)
        assert fullrank_block.expansion == lowrank_block.expansion
        self.linear = nn.Linear(512*fullrank_block.expansion, num_classes)

    def _make_layer(self, fullrank_block, lowrank_block, planes, num_blocks, stride, rank_list):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            if self._block_counter < 2:
            #if self._block_counter < 4:
                layers.append(fullrank_block(self.in_planes, planes, stride))
                self.in_planes = planes * fullrank_block.expansion
            else:
                layers.append(lowrank_block(self.in_planes, planes, 
                                rank_list[self._rank_counter:self._rank_counter+2], 
                                frob_decay=self._frob_decay, 
                                extra_bns=self._extra_bns,
                                stride=stride
                            ))
                self.in_planes = planes * lowrank_block.expansion
                self._rank_counter += 2
            self._block_counter += 1
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        #out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class BenchmarkResNet(nn.Module):
    def __init__(self, fullrank_block, lowrank_block, num_blocks, rank_ratio, num_classes=10):
        super(BenchmarkResNet, self).__init__()
        self.in_planes = 64
        self._rank_ratio = rank_ratio

        self._block_iter_start = torch.cuda.Event(enable_timing=True)
        self._block_iter_end = torch.cuda.Event(enable_timing=True)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        if self._rank_ratio == 0:
            block = fullrank_block
        else:
            block = lowrank_block
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            if self._rank_ratio == 0:
                layers.append(block(self.in_planes, planes, stride))
            else:
                layers.append(block(self.in_planes, planes, stride, rank_ratio=self._rank_ratio))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) # we do not factorize the first conv layer

        self._block_iter_start.record()
        out = self.layer1(out)
        self._block_iter_end.record()
        torch.cuda.synchronize()
        iter_comp_dur = float(self._block_iter_start.elapsed_time(self._block_iter_end))
        print("@ Block1, Itertime: {}".format(iter_comp_dur))

        self._block_iter_start.record()
        out = self.layer2(out)
        self._block_iter_end.record()
        torch.cuda.synchronize()
        iter_comp_dur = float(self._block_iter_start.elapsed_time(self._block_iter_end))
        print("@ Block2, Itertime: {}".format(iter_comp_dur))

        self._block_iter_start.record()
        out = self.layer3(out)
        self._block_iter_end.record()
        torch.cuda.synchronize()
        iter_comp_dur = float(self._block_iter_start.elapsed_time(self._block_iter_end))
        print("@ Block3, Itertime: {}".format(iter_comp_dur))

        self._block_iter_start.record()
        out = self.layer4(out)
        self._block_iter_end.record()
        torch.cuda.synchronize()
        iter_comp_dur = float(self._block_iter_start.elapsed_time(self._block_iter_end))
        print("@ Block4, Itertime: {}".format(iter_comp_dur))
        
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class LowRankResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(LowRankResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block[0], 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block[1], 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block[2], 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block[3], 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block[0].expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes)

def BaselineResNet18():
    return BaselineResNet(BasicBlock, [2,2,2,2])

def LowrankResNet18(rank_ratio=None, num_classes=10):
    return HybridResNet(BasicBlock, LowrankBasicBlock, [2,2,2,2], rank_ratio=rank_ratio, num_classes=num_classes)

def ResNet18Benchmark(rank_ratio=None, num_classes=10):
    return BenchmarkResNet(BasicBlock, LowrankBasicBlock, [2,2,2,2], rank_ratio=rank_ratio, num_classes=num_classes)

def PufferfishResNet18(num_classes=10):
    # Pufferfish 1.0, used as a baseline to compare
    return PufferfishResNet(BasicBlock, LowrankBasicBlock, [2,2,2,2], rank_ratio=4, num_classes=num_classes)

def LowrankResNet18Adapt(rank_list, frob_decay, extra_bns, num_classes=10):
    return HybridResNetAdapt(BasicBlock, LowrankBasicBlockAdapt, [2,2,2,2], 
                                rank_list=rank_list, 
                                num_classes=num_classes, 
                                frob_decay=frob_decay,
                                extra_bns=extra_bns
                    )

def LowRankResResNet18():
    return ResNet(LowRankBasicBlockResidual, [2,2,2,2])

def LowRankResLowRankResNet18():
    return ResNet(LowRankBasicBlockLowRankResidual, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def LowRankResNet34():
    return ResNet(LowRankBasicBlockConv1x1, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def LowRankResNet50():
    return ResNet(LowRankBottleneckConv1x1, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())


if __name__ == "__main__":
    model = LowRankResResNet18()