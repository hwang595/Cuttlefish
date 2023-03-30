# HashedNet Convolutional Layer: https://arxiv.org/abs/1504.04788
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F


class HashedConv2d(nn.Conv2d):
    """Conv2d with the weights of the convolutional filters parameterised using
    a budgeted subset of parameters and random indexes to place those
    parameters in the weight tensor."""
    def __init__(self, in_channels, out_channels, kernel_size, budget,
            stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(HashedConv2d, self).__init__(in_channels, out_channels,
                kernel_size, stride=stride, padding=padding, dilation=dilation,
                groups=groups, bias=True)
        # grab budgeted subset of the weights
        assert self.weight.numel() >= budget, \
                f"budget {budget} higher than {self.weight.numel()}"
        self.weight_size = self.weight.size()
        budgeted = self.weight.data.view(-1)[:budget]
        del self.weight
        # register non-budgeted weights
        self.register_parameter('hashed_weight', nn.Parameter(budgeted))
        # precompute random index matrix
        idxs = torch.randint(high=budget-1, size=self.weight_size).long()
        idxs = idxs.view(-1)
        # register indexes as a buffer
        self.register_buffer('idxs', idxs)
        #self.W = self.weight[self.idxs].cuda()

    def forward(self, x):
        # index to make weight matrix
        try:
            W = self.hashed_weight.index_select(0, self.idxs).view(self.weight_size)
        except RuntimeError:
            import ipdb
            ipdb.set_trace()
        # complete forward pass as normal
        return F.conv2d(x, W, self.bias, self.stride, self.padding,
                self.dilation, self.groups)


class HalfHashedSeparable(nn.Module):
    """A depthwise grouped convolution followed by a HashedNet 1x1 convolution.
    Grouped convolution could also be hashed, but it's not."""
    def __init__(self, in_channels, out_channels, kernel_size, budget,
            stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(HalfHashedSeparable, self).__init__()
        # has to have hashed in the name to get caught by alternative weight
        # decay setting, it is not actually hashed
        if kernel_size > 1:
            self.grouped = nn.Conv2d(in_channels, in_channels, kernel_size,
                    stride=stride, padding=padding, dilation=dilation,
                    groups=in_channels, bias=False)
            # we spent some of the budget on that grouped convolution
            assert self.grouped.weight.numel() == reduce(lambda x,y: x*y, self.grouped.weight.size())
            budget = budget - self.grouped.weight.numel()
            assert budget > 0, \
                    "budget exceeded by grouped convolution: %i too many"%(-budget)
            self.hashed = HashedConv2d(in_channels, out_channels, 1, budget,
                    bias=bias)
        else:
            self.grouped = None
            self.hashed = HashedConv2d(in_channels, out_channels, 1, budget,
                    stride=stride, padding=padding, dilation=dilation,
                    bias=bias)

    def forward(self, x):
        if self.grouped is not None:
            x = self.grouped(x)
        return self.hashed(x)


class HashedSeparable(nn.Module):
    """Separabled, where grouped and pointwise are both Hashed.."""
    def __init__(self, in_channels, out_channels, kernel_size, budget,
            stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(HashedSeparable, self).__init__()
        # has to have hashed in the name to get caught by alternative weight
        # decay setting, it is not actually hashed
        grouped_params = float(in_channels * kernel_size * kernel_size)
        pointwise_params = float(in_channels * out_channels)
        total_params = grouped_params + pointwise_params
        grouped_budget = int(budget*grouped_params/total_params)
        pointwise_budget = int(budget*pointwise_params/total_params)
        #print(total_params, grouped_budget, pointwise_budget)
        if kernel_size > 1:
            self.grouped = HashedConv2d(in_channels, in_channels, kernel_size,
                    grouped_budget, stride=stride, padding=padding,
                    dilation=dilation, groups=in_channels, bias=False)
            stride = 1
        else:
            self.grouped = None
            pointwise_budget = budget
        assert budget > 0, "budget must be greater than 0, was %i"%(-budget)
        self.hashed = HashedConv2d(in_channels, out_channels, 1,
                pointwise_budget, stride=stride, bias=bias)

    def forward(self, x):
        if self.grouped is not None:
            x = self.grouped(x)
        return self.hashed(x)


if __name__ == '__main__':
    from timeit import timeit
    setup = "from __main__ import HashedConv2d; import torch; X = torch.randn(128, 256, 28, 28).cuda(); conv = HashedConv2d(256, 512, 3, 1000, bias=False).cuda()"
    print("HashedConv2d: ", timeit("_ = conv(X)", setup=setup, number=100))
    setup = "import torch.nn as nn; import torch; X = torch.randn(128, 256, 28, 28).cuda(); conv = nn.Conv2d(256, 512, 3, bias=False).cuda()"
    print("Conv2d: ", timeit("_ = conv(X)", setup=setup, number=100))
    setup = "from __main__ import HalfHashedSeparable; import torch; X = torch.randn(128, 256, 28, 28).cuda(); conv = HalfHashedSeparable(256, 512, 3, 5000, bias=False).cuda()"
    print("HalfHashedSeparable: ", timeit("_ = conv(X)", setup=setup, number=100))
    setup = "from __main__ import HashedSeparable; import torch; X = torch.randn(128, 256, 28, 28).cuda(); conv = HashedSeparable(256, 512, 3, 5000, bias=False).cuda()"
    print("HashedSeparable: ", timeit("_ = conv(X)", setup=setup, number=100))
