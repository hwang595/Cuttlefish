'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

#from torch.cuda.amp import autocast


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



class FullRankVGG19(nn.Module):
    '''
    FullRankVGG19 Model 
    '''
    def __init__(self, num_classes=10):
        super(FullRankVGG19, self).__init__()
        # based on the literature, we don't touch the first conv layer
        self.conv1 = nn.Conv2d(3, 64, 3, 1, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)

        self.block1 = nn.Sequential(
                            nn.Conv2d(64, 64, 3, 1, padding=1, bias=False),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=2, stride=2),
                            nn.Conv2d(64, 128, 3, 1, padding=1, bias=False),
                            nn.BatchNorm2d(128),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(128, 128, 3, 1, padding=1, bias=False),
                            nn.BatchNorm2d(128),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=2, stride=2),
                        ) 
        
        self.block2 = nn.Sequential(
                            nn.Conv2d(128, 256, 3, 1, padding=1, bias=False),
                            nn.BatchNorm2d(256),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(256, 256, 3, 1, padding=1, bias=False),
                            nn.BatchNorm2d(256),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(256, 256, 3, 1, padding=1, bias=False),
                            nn.BatchNorm2d(256),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(256, 256, 3, 1, padding=1, bias=False),
                            nn.BatchNorm2d(256),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=2, stride=2)
                        )
        
        self.block3 = nn.Sequential(
                            nn.Conv2d(256, 512, 3, 1, padding=1, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(512, 512, 3, 1, padding=1, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(512, 512, 3, 1, padding=1, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(512, 512, 3, 1, padding=1, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=2, stride=2),
                            nn.Conv2d(512, 512, 3, 1, padding=1, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(512, 512, 3, 1, padding=1, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(512, 512, 3, 1, padding=1, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(512, 512, 3, 1, padding=1, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(inplace=True),
                            nn.AvgPool2d(2)
                            #nn.AvgPool2d(kernel_size=1, stride=1)
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


LR_FACOR = 4
class PufferfishVGG19(nn.Module):
    '''
    PufferfishVGG19 model from the mlsys21 paper: 
    https://proceedings.mlsys.org/paper/2021/hash/84d9ee44e457ddef7f2c4f25dc8fa865-Abstract.html.
    '''
    def __init__(self, num_classes=10):
        super(PufferfishVGG19, self).__init__()
        # based on the literature, we don't touch the first conv layer
        self.conv1 = nn.Conv2d(3, 64, 3, 1, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 64, 3, 1, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1, bias=False)
        self.batch_norm4 = nn.BatchNorm2d(128)
        
        self.conv5 = nn.Conv2d(128, 256, 3, 1, padding=1, bias=False)
        self.batch_norm5 = nn.BatchNorm2d(256)
        
        self.conv6 = nn.Conv2d(256, 256, 3, 1, padding=1, bias=False)
        self.batch_norm6 = nn.BatchNorm2d(256)
        
        #self.conv7_u = nn.Conv2d(256, int(256/LR_FACOR), 3, 1, padding=1, bias=False)
        #self.conv7_v = nn.Conv2d(int(256/LR_FACOR), 256, kernel_size=1, stride=1, bias=False)
        self.conv7 = nn.Conv2d(256, 256, 3, 1, padding=1, bias=False)
        self.batch_norm7 = nn.BatchNorm2d(256)
        
        #self.conv8_u = nn.Conv2d(256, int(256/LR_FACOR), 3, 1, padding=1, bias=False)
        #self.conv8_v = nn.Conv2d(int(256/LR_FACOR), 256, kernel_size=1, stride=1, bias=False)
        self.conv8 = nn.Conv2d(256, 256, 3, 1, padding=1, bias=False)
        self.batch_norm8 = nn.BatchNorm2d(256)
        
        #self.max_pooling3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #self.conv9_u = nn.Conv2d(256, int(512/LR_FACOR), 3, 1, padding=1, bias=False)
        #self.conv9_v = nn.Conv2d(int(512/LR_FACOR), 512, kernel_size=1, stride=1, bias=False)
        self.conv9 = nn.Conv2d(256, 512, 3, 1, padding=1, bias=False)
        self.batch_norm9 = nn.BatchNorm2d(512)
        
        self.conv10_u = nn.Conv2d(512, int(512/LR_FACOR), 3, 1, padding=1, bias=False)
        self.conv10_v = nn.Conv2d(int(512/LR_FACOR), 512, kernel_size=1, stride=1, bias=False)
        #self.conv10 = nn.Conv2d(512, 512, 3, 1, padding=1, bias=False)
        self.batch_norm10 = nn.BatchNorm2d(512)
        
        self.conv11_u = nn.Conv2d(512, int(512/LR_FACOR), 3, 1, padding=1, bias=False)
        self.conv11_v = nn.Conv2d(int(512/LR_FACOR), 512, kernel_size=1, stride=1, bias=False)
        #self.conv11 = nn.Conv2d(512, 512, 3, 1, padding=1, bias=False)
        self.batch_norm11 = nn.BatchNorm2d(512)
        
        self.conv12_u = nn.Conv2d(512, int(512/LR_FACOR), 3, 1, padding=1, bias=False)
        self.conv12_v = nn.Conv2d(int(512/LR_FACOR), 512, kernel_size=1, stride=1, bias=False)
        #self.conv12 = nn.Conv2d(512, 512, 3, 1, padding=1, bias=False)
        self.batch_norm12 = nn.BatchNorm2d(512)

        #self.max_pooling4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv13_u = nn.Conv2d(512, int(512/LR_FACOR), 3, 1, padding=1, bias=False)
        self.conv13_v = nn.Conv2d(int(512/LR_FACOR), 512, kernel_size=1, stride=1, bias=False)
        #self.conv13 = nn.Conv2d(512, 512, 3, 1, padding=1, bias=False)
        self.batch_norm13 = nn.BatchNorm2d(512)
        
        self.conv14_u = nn.Conv2d(512, int(512/LR_FACOR), 3, 1, padding=1, bias=False)
        self.conv14_v = nn.Conv2d(int(512/LR_FACOR), 512, kernel_size=1, stride=1, bias=False)
        #self.conv14 = nn.Conv2d(512, 512, 3, 1, padding=1, bias=False)
        self.batch_norm14 = nn.BatchNorm2d(512)
        
        self.conv15_u = nn.Conv2d(512, int(512/LR_FACOR), 3, 1, padding=1, bias=False)
        self.conv15_v = nn.Conv2d(int(512/LR_FACOR), 512, kernel_size=1, stride=1, bias=False)
        #self.conv15 = nn.Conv2d(512, 512, 3, 1, padding=1, bias=False)
        self.batch_norm15 = nn.BatchNorm2d(512)

        self.conv16_u = nn.Conv2d(512, int(512/LR_FACOR), 3, 1, padding=1, bias=False)
        self.conv16_v = nn.Conv2d(int(512/LR_FACOR), 512, kernel_size=1, stride=1, bias=False)
        #self.conv16 = nn.Conv2d(512, 512, 3, 1, padding=1, bias=False)
        self.batch_norm16 = nn.BatchNorm2d(512)
        self.max_pooling5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

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
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        
        #x = self.conv2_v(self.conv2_u(x))
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        #x = self.conv3_v(self.conv3_u(x))
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = F.relu(x)
        
        #x = self.conv4_v(self.conv4_u(x))
        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        #x = self.conv5_v(self.conv5_u(x))
        x = self.conv5(x)
        x = self.batch_norm5(x)
        x = F.relu(x)
        
        #x = self.conv6_v(self.conv6_u(x))
        x = self.conv6(x)
        x = self.batch_norm6(x)
        x = F.relu(x)
        
        #x = self.conv7_v(self.conv7_u(x))
        x = self.conv7(x)
        x = self.batch_norm7(x)
        x = F.relu(x)
        
        #x = self.conv8_v(self.conv8_u(x))
        x = self.conv8(x)
        x = self.batch_norm8(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        #x = self.conv9_v(self.conv9_u(x))
        x = self.conv9(x)
        x = self.batch_norm9(x)
        x = F.relu(x)
        
        x = self.conv10_v(self.conv10_u(x))
        #x = self.conv10(x)
        x = self.batch_norm10(x)
        x = F.relu(x)
        
        x = self.conv11_v(self.conv11_u(x))
        #x = self.conv11(x)
        x = self.batch_norm11(x)
        x = F.relu(x)
        
        x = self.conv12_v(self.conv12_u(x))
        #x = self.conv12(x)
        x = self.batch_norm12(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv13_v(self.conv13_u(x))
        #x = self.conv13(x)
        x = self.batch_norm13(x)
        x = F.relu(x)
        
        x = self.conv14_v(self.conv14_u(x))
        #x = self.conv14(x)
        x = self.batch_norm14(x)
        x = F.relu(x)
        
        x = self.conv15_v(self.conv15_u(x))
        #x = self.conv15(x)
        x = self.batch_norm15(x)
        x = F.relu(x)
        
        x = self.conv16_v(self.conv16_u(x))
        #x = self.conv16(x)
        x = self.batch_norm16(x)
        x = F.relu(x)
        #x = F.max_pool2d(x, 2, 2)
        x = nn.AvgPool2d(2)(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



class LowRankVGG19Adapt(nn.Module):
    def __init__(self, rank_list, num_classes=10, frob_decay=False, extra_bns=True):
        super(LowRankVGG19Adapt, self).__init__()
        self._frob_decay = frob_decay
        self._extra_bns = extra_bns

        # based on the literature, we don't touch the first conv layer
        # the rank list contains 7 rank numbers, for layer conv10 - conv 16
        self.conv1 = nn.Conv2d(3, 64, 3, 1, padding=1, bias=False) # 0
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, padding=1, bias=False) # 1
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1, bias=False) # 2
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1, bias=False) # 3
        self.batch_norm4 = nn.BatchNorm2d(128)

        #self.conv5 = nn.Conv2d(128, 256, 3, 1, padding=1, bias=False) # 4
        self.conv5_u = nn.Conv2d(128, rank_list[0], 3, 1, padding=1, bias=False)
        self.conv5_v = nn.Conv2d(rank_list[0], 256, kernel_size=1, stride=1, bias=False)
        self.batch_norm5 = nn.BatchNorm2d(256)
        
        #self.conv6 = nn.Conv2d(256, 256, 3, 1, padding=1, bias=False) # 5
        self.conv6_u = nn.Conv2d(256, rank_list[1], 3, 1, padding=1, bias=False)
        self.conv6_v = nn.Conv2d(rank_list[1], 256, kernel_size=1, stride=1, bias=False)
        self.batch_norm6 = nn.BatchNorm2d(256)
        
        #self.conv7 = nn.Conv2d(256, 256, 3, 1, padding=1, bias=False) # 6
        self.conv7_u = nn.Conv2d(256, rank_list[2], 3, 1, padding=1, bias=False) # 6
        self.conv7_v = nn.Conv2d(rank_list[2], 256, kernel_size=1, stride=1, bias=False)
        self.batch_norm7 = nn.BatchNorm2d(256)
        
        #self.conv8 = nn.Conv2d(256, 256, 3, 1, padding=1, bias=False) # 7
        self.conv8_u = nn.Conv2d(256, rank_list[3], 3, 1, padding=1, bias=False) # 7
        self.conv8_v = nn.Conv2d(rank_list[3], 256, kernel_size=1, stride=1, bias=False)
        self.batch_norm8 = nn.BatchNorm2d(256)
        
        #self.conv9 = nn.Conv2d(256, 512, 3, 1, padding=1, bias=False) # 8
        self.conv9_u = nn.Conv2d(256, rank_list[4], 3, 1, padding=1, bias=False) # 8
        self.conv9_v = nn.Conv2d(rank_list[4], 512, kernel_size=1, stride=1, bias=False)        
        self.batch_norm9 = nn.BatchNorm2d(512)
        
        self.conv10_u = nn.Conv2d(512, rank_list[5], 3, 1, padding=1, bias=False) # 9
        self.conv10_v = nn.Conv2d(rank_list[5], 512, kernel_size=1, stride=1, bias=False)
        self.batch_norm10 = nn.BatchNorm2d(512)
        
        self.conv11_u = nn.Conv2d(512, rank_list[6], 3, 1, padding=1, bias=False) # 10
        self.conv11_v = nn.Conv2d(rank_list[6], 512, kernel_size=1, stride=1, bias=False)
        self.batch_norm11 = nn.BatchNorm2d(512)
        
        self.conv12_u = nn.Conv2d(512, rank_list[7], 3, 1, padding=1, bias=False) # 11
        self.conv12_v = nn.Conv2d(rank_list[7], 512, kernel_size=1, stride=1, bias=False)
        self.batch_norm12 = nn.BatchNorm2d(512)
        
        self.conv13_u = nn.Conv2d(512, rank_list[8], 3, 1, padding=1, bias=False) # 12
        self.conv13_v = nn.Conv2d(rank_list[8], 512, kernel_size=1, stride=1, bias=False)
        self.batch_norm13 = nn.BatchNorm2d(512)
        
        self.conv14_u = nn.Conv2d(512, rank_list[9], 3, 1, padding=1, bias=False) # 13
        self.conv14_v = nn.Conv2d(rank_list[9], 512, kernel_size=1, stride=1, bias=False)
        self.batch_norm14 = nn.BatchNorm2d(512)
        
        self.conv15_u = nn.Conv2d(512, rank_list[10], 3, 1, padding=1, bias=False) # 14
        self.conv15_v = nn.Conv2d(rank_list[10], 512, kernel_size=1, stride=1, bias=False)
        self.batch_norm15 = nn.BatchNorm2d(512)

        self.conv16_u = nn.Conv2d(512, rank_list[11], 3, 1, padding=1, bias=False) # 15
        self.conv16_v = nn.Conv2d(rank_list[11], 512, kernel_size=1, stride=1, bias=False)
        self.batch_norm16 = nn.BatchNorm2d(512)

        if not self._frob_decay and self._extra_bns:
            self.batch_norm5_u = nn.BatchNorm2d(rank_list[0])
            self.batch_norm6_u = nn.BatchNorm2d(rank_list[1])
            self.batch_norm7_u = nn.BatchNorm2d(rank_list[2])
            self.batch_norm8_u = nn.BatchNorm2d(rank_list[3])
            self.batch_norm9_u = nn.BatchNorm2d(rank_list[4])
            self.batch_norm10_u = nn.BatchNorm2d(rank_list[5])
            self.batch_norm11_u = nn.BatchNorm2d(rank_list[6])
            self.batch_norm12_u = nn.BatchNorm2d(rank_list[7])
            self.batch_norm13_u = nn.BatchNorm2d(rank_list[8])
            self.batch_norm14_u = nn.BatchNorm2d(rank_list[9])
            self.batch_norm15_u = nn.BatchNorm2d(rank_list[10])
            self.batch_norm16_u = nn.BatchNorm2d(rank_list[11])

        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)

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
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = F.relu(x)
        
        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        if self._frob_decay or not self._extra_bns:
            x = F.relu(self.batch_norm5(self.conv5_v(self.conv5_u(x))))
            x = F.relu(self.batch_norm6(self.conv6_v(self.conv6_u(x))))
            x = F.relu(self.batch_norm7(self.conv7_v(self.conv7_u(x))))
            x = F.relu(self.batch_norm8(self.conv8_v(self.conv8_u(x))))
            x = F.max_pool2d(x, 2, 2)

            x = F.relu(self.batch_norm9(self.conv9_v(self.conv9_u(x))))
            x = F.relu(self.batch_norm10(self.conv10_v(self.conv10_u(x))))
            x = F.relu(self.batch_norm11(self.conv11_v(self.conv11_u(x))))
            x = F.relu(self.batch_norm12(self.conv12_v(self.conv12_u(x))))
            x = F.max_pool2d(x, 2, 2)

            x = F.relu(self.batch_norm13(self.conv13_v(self.conv13_u(x))))
            x = F.relu(self.batch_norm14(self.conv14_v(self.conv14_u(x))))
            x = F.relu(self.batch_norm15(self.conv15_v(self.conv15_u(x))))
            x = F.relu(self.batch_norm16(self.conv16_v(self.conv16_u(x))))
        else:
            x = F.relu(self.batch_norm5(self.conv5_v(self.batch_norm5_u(self.conv5_u(x)))))
            x = F.relu(self.batch_norm6(self.conv6_v(self.batch_norm6_u(self.conv6_u(x)))))
            x = F.relu(self.batch_norm7(self.conv7_v(self.batch_norm7_u(self.conv7_u(x)))))
            x = F.relu(self.batch_norm8(self.conv8_v(self.batch_norm8_u(self.conv8_u(x)))))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.batch_norm9(self.conv9_v(self.batch_norm9_u(self.conv9_u(x)))))
            x = F.relu(self.batch_norm10(self.conv10_v(self.batch_norm10_u(self.conv10_u(x)))))
            x = F.relu(self.batch_norm11(self.conv11_v(self.batch_norm11_u(self.conv11_u(x)))))
            x = F.relu(self.batch_norm12(self.conv12_v(self.batch_norm12_u(self.conv12_u(x)))))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.batch_norm13(self.conv13_v(self.batch_norm13_u(self.conv13_u(x)))))
            x = F.relu(self.batch_norm14(self.conv14_v(self.batch_norm14_u(self.conv14_u(x)))))
            x = F.relu(self.batch_norm15(self.conv15_v(self.batch_norm15_u(self.conv15_u(x)))))
            x = F.relu(self.batch_norm16(self.conv16_v(self.batch_norm16_u(self.conv16_u(x)))))
        x = nn.AvgPool2d(2)(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGG19Benchmark(nn.Module):
    '''
    VGG19Benchmark Model 
    '''
    def __init__(self, rank_ratio=0.0, num_classes=10):
        super(VGG19Benchmark, self).__init__()
        # based on the literature, we don't touch the first conv layer
        self.conv1 = nn.Conv2d(3, 64, 3, 1, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self._rank_ratio = rank_ratio

        self._block_iter_start = torch.cuda.Event(enable_timing=True)
        self._block_iter_end = torch.cuda.Event(enable_timing=True)

        if self._rank_ratio == 0.0:
            self.block1 = nn.Sequential(
                                nn.Conv2d(64, 64, 3, 1, padding=1, bias=False),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(kernel_size=2, stride=2),
                                nn.Conv2d(64, 128, 3, 1, padding=1, bias=False),
                                nn.BatchNorm2d(128),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(128, 128, 3, 1, padding=1, bias=False),
                                nn.BatchNorm2d(128),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(kernel_size=2, stride=2),
                            ) 
            
            self.block2 = nn.Sequential(
                                nn.Conv2d(128, 256, 3, 1, padding=1, bias=False),
                                nn.BatchNorm2d(256),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(256, 256, 3, 1, padding=1, bias=False),
                                nn.BatchNorm2d(256),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(256, 256, 3, 1, padding=1, bias=False),
                                nn.BatchNorm2d(256),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(256, 256, 3, 1, padding=1, bias=False),
                                nn.BatchNorm2d(256),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(kernel_size=2, stride=2)
                            )
            
            self.block3 = nn.Sequential(
                                nn.Conv2d(256, 512, 3, 1, padding=1, bias=False),
                                nn.BatchNorm2d(512),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(512, 512, 3, 1, padding=1, bias=False),
                                nn.BatchNorm2d(512),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(512, 512, 3, 1, padding=1, bias=False),
                                nn.BatchNorm2d(512),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(512, 512, 3, 1, padding=1, bias=False),
                                nn.BatchNorm2d(512),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(kernel_size=2, stride=2),
                                nn.Conv2d(512, 512, 3, 1, padding=1, bias=False),
                                nn.BatchNorm2d(512),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(512, 512, 3, 1, padding=1, bias=False),
                                nn.BatchNorm2d(512),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(512, 512, 3, 1, padding=1, bias=False),
                                nn.BatchNorm2d(512),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(512, 512, 3, 1, padding=1, bias=False),
                                nn.BatchNorm2d(512),
                                nn.ReLU(inplace=True),
                                nn.AvgPool2d(2)
                            )
        else:
            self.block1 = nn.Sequential(
                                nn.Conv2d(64, int(64/self._rank_ratio), 3, 1, padding=1, bias=False),
                                nn.Conv2d(int(64/self._rank_ratio), 64, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(kernel_size=2, stride=2),
                                nn.Conv2d(64, int(128/self._rank_ratio), 3, 1, padding=1, bias=False),
                                nn.Conv2d(int(128/self._rank_ratio), 128, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(128),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(128, int(128/self._rank_ratio), 3, 1, padding=1, bias=False),
                                nn.Conv2d(int(128/self._rank_ratio), 128, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(128),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(kernel_size=2, stride=2),
                            ) 
            
            self.block2 = nn.Sequential(
                                nn.Conv2d(128, int(256/self._rank_ratio), 3, 1, padding=1, bias=False),
                                nn.Conv2d(int(256/self._rank_ratio), 256, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(256),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(256, int(256/self._rank_ratio), 3, 1, padding=1, bias=False),
                                nn.Conv2d(int(256/self._rank_ratio), 256, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(256),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(256, int(256/self._rank_ratio), 3, 1, padding=1, bias=False),
                                nn.Conv2d(int(256/self._rank_ratio), 256, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(256),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(256, int(256/self._rank_ratio), 3, 1, padding=1, bias=False),
                                nn.Conv2d(int(256/self._rank_ratio), 256, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(256),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(kernel_size=2, stride=2)
                            )
            
            self.block3 = nn.Sequential(
                                nn.Conv2d(256, int(512/self._rank_ratio), 3, 1, padding=1, bias=False),
                                nn.Conv2d(int(512/self._rank_ratio), 512, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(512),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(512, int(512/self._rank_ratio), 3, 1, padding=1, bias=False),
                                nn.Conv2d(int(512/self._rank_ratio), 512, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(512),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(512, int(512/self._rank_ratio), 3, 1, padding=1, bias=False),
                                nn.Conv2d(int(512/self._rank_ratio), 512, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(512),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(512, int(512/self._rank_ratio), 3, 1, padding=1, bias=False),
                                nn.Conv2d(int(512/self._rank_ratio), 512, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(512),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(kernel_size=2, stride=2),
                                nn.Conv2d(512, int(512/self._rank_ratio), 3, 1, padding=1, bias=False),
                                nn.Conv2d(int(512/self._rank_ratio), 512, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(512),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(512, int(512/self._rank_ratio), 3, 1, padding=1, bias=False),
                                nn.Conv2d(int(512/self._rank_ratio), 512, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(512),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(512, int(512/self._rank_ratio), 3, 1, padding=1, bias=False),
                                nn.Conv2d(int(512/self._rank_ratio), 512, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(512),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(512, int(512/self._rank_ratio), 3, 1, padding=1, bias=False),
                                nn.Conv2d(int(512/self._rank_ratio), 512, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(512),
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
        
        self._block_iter_start.record()
        x = self.block1(x)
        self._block_iter_end.record()
        torch.cuda.synchronize()
        iter_comp_dur = float(self._block_iter_start.elapsed_time(self._block_iter_end))
        print("@ Block1, Itertime: {}".format(iter_comp_dur))

        self._block_iter_start.record()
        x = self.block2(x)
        self._block_iter_end.record()
        torch.cuda.synchronize()
        iter_comp_dur = float(self._block_iter_start.elapsed_time(self._block_iter_end))
        print("@ Block2, Itertime: {}".format(iter_comp_dur))

        self._block_iter_start.record()
        x = self.block3(x)
        self._block_iter_end.record()
        torch.cuda.synchronize()
        iter_comp_dur = float(self._block_iter_start.elapsed_time(self._block_iter_end))
        print("@ Block3, Itertime: {}".format(iter_comp_dur))

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    #net = LowrankVGG19LTH()
    lr_layers = ["conv9", "classifier.1.", "classifier.2."]
    net = LowRankVGG19Dynamic(rank_list=[i for i in range(5, 16)], 
                            low_rank_layers=lr_layers)
    print("#### Model arch: {}, num params: {}".format(net, param_counter(net)))