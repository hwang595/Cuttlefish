# Substitute layer explicitly decomposing the tensors in convolutional layers
# All implemented using tntorch: https://github.com/rballester/tntorch
# All also use a separable design: the low-rank approximate pointwise
# convolution is preceded by a grouped convolution
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import tntorch as tn
torch.set_default_dtype(torch.float32)


def dimensionize(t, d, rank_scale):
    """Take a tensor, t, and reshape so that it has d dimensions, of roughly
    equal size."""
    # if not, we have to do some work
    N = t.numel()
    # do d-th root with log
    equal = math.exp((1./d)*math.log(N))
    # if this is an integer, our work here is done
    if abs(round(equal) - equal) < 1e-6:
        dims = [int(round(equal))]*d
    # if the tensor already has d dimensions
    elif t.ndimension() == d:
        dims = list(t.size())
    # oh no, then we want to build up a list of dimensions it *does* divide by
    else:
        dims = []
        for i in range(d-1):
            divisor = closest_divisor(N, int(round(equal)))
            dims.append(divisor)
            N = N//divisor
        dims.append(N)
    # rank between dimensions must be less than
    ranks = {}
    ranks['ranks_tt'] = [max(1,int(round(rank_scale*min(b,a)))) for b,a in zip(dims, dims[1:])]
    ranks['ranks_tucker'] = [max(1,int(round(rank_scale*d))) for d in dims]
    ranks['ranks_cp'] = max(1,int(round(rank_scale*min(dims))))
    return t.view(*dims), ranks

def closest_divisor(N, d):
    if N < d:
        return N
    while N%d != 0:
        d += 1
    return d


class TnTorchConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, rank_scale,
            TnConstructor, stride=1, padding=0, dilation=1, groups=1,
            bias=True):
        self.TnConstructor = TnConstructor
        assert groups == 1
        if kernel_size == 1:
            super(TnTorchConv2d, self).__init__(in_channels, out_channels, 1,
                    stride=stride, padding=padding, dilation=dilation, bias=bias)
        elif kernel_size > 1:
            super(TnTorchConv2d, self).__init__(in_channels, out_channels, 1, bias=bias)
            self.grouped = nn.Conv2d(in_channels, in_channels,
                    kernel_size, stride=stride, padding=padding, dilation=dilation,
                    groups=in_channels, bias=False)
        self.rank_scale = rank_scale
        self.tn_weight = self.TnConstructor(self.weight.data.squeeze(), rank_scale=self.rank_scale)
        # store the correct size for this weight
        self.weight_size = self.weight.size()
        # check the fit to the weight initialisation
        self.store_metrics(self.weight)
        # delete the original weight
        del self.weight
        # then register the cores of the Tensor Train as parameters
        self.register_tnparams(self.tn_weight.cores, self.tn_weight.Us)

    def register_tnparams(self, cores, Us):
        cores = [] if all([c is None for c in cores]) else cores
        Us = [] if all([u is None for u in Us]) else Us
        # tensor train or cp cores
        for i,core in enumerate(cores):
            core_name = 'weight_core_%i'%i
            if hasattr(self, core_name):
                delattr(self, core_name)
            core.requires_grad = True
            self.register_parameter(core_name, nn.Parameter(core))
            # replace Parameter in tn.Tensor object
            self.tn_weight.cores[i] = getattr(self, core_name)
        for i, u in enumerate(Us):
            u_name = 'weight_u_%i'%i
            if hasattr(self, u_name):
                delattr(self, u_name)
            u.requires_grad = True
            self.register_parameter(u_name, nn.Parameter(u))
            # replace Parameter in tn.Tensor object
            self.tn_weight.Us[i] = getattr(self, u_name)

    def conv_weight(self):
        weight = self.tn_weight.torch()
        n,d,_,_ = self.weight_size
        return weight.view(n,d,1,1)

    def reset_parameters(self):
        if hasattr(self, 'tn_weight'):
            # full rank weight tensor
            weight = self.conv_weight()
        else:
            weight = self.weight.data
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        weight.data.uniform_(-stdv, stdv)
        if hasattr(self, 'tn_weight'):
            self.tn_weight = self.TnConstructor(weight.data.squeeze(), rank_scale=self.rank_scale)
            # update cores
            self.register_tnparams(self.tn_weight.cores, self.tn_weight.Us)
        else:
            self.weight.data = weight
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        if hasattr(self, 'grouped'):
            out = self.grouped(x)
        else:
            out = x
        weight = self.conv_weight()
        self.normsqW = weight.pow(2).sum()
        return F.conv2d(out, weight, self.bias, self.stride, self.padding,
                self.dilation, self.groups)

    def store_metrics(self, full):
        t = self.tn_weight
        full = full.view(t.torch().size())
        self.compression = (full.numel(), t.numel(), full.numel() / t.numel())
        self.relative_error = tn.relative_error(full, t)
        self.rmse = tn.rmse(full, t)
        self.r_squared = tn.r_squared(full, t)

    def extra_repr(self):
        extra = []
        extra.append(self.tn_weight.__repr__())
        extra.append('Compression ratio: {}/{} = {:g}'.format(*self.compression))
        extra.append('Relative error: %f'%self.relative_error)
        extra.append('RMSE: %f'%self.rmse)
        extra.append('R^2: %f'%self.r_squared)
        return "\n".join(extra)


class TensorTrain(TnTorchConv2d):
    def __init__(self, in_channels, out_channels, kernel_size, rank_scale,
            dimensions, stride=1, padding=0, dilation=1, groups=1, bias=True):
        def TT(tensor, rank_scale):
            tensor, ranks = dimensionize(tensor, dimensions, rank_scale)
            return tn.Tensor(tensor, ranks_tt=ranks['ranks_tt'])
        super(TensorTrain, self).__init__(in_channels, out_channels,
                kernel_size, rank_scale, TT, stride=stride, padding=padding,
                dilation=dilation, groups=groups, bias=bias)


class Tucker(TnTorchConv2d):
    def __init__(self, in_channels, out_channels, kernel_size, rank_scale,
            dimensions, stride=1, padding=0, dilation=1, groups=1, bias=True):
        def tucker(tensor, rank_scale):
            tensor, ranks = dimensionize(tensor, dimensions, rank_scale)
            return tn.Tensor(tensor, ranks_tucker=ranks['ranks_tucker'])
        super(Tucker, self).__init__(in_channels, out_channels, kernel_size,
                rank_scale, tucker, stride=stride, padding=padding,
                dilation=dilation, groups=groups, bias=bias)


class CP(TnTorchConv2d):
    def __init__(self, in_channels, out_channels, kernel_size, rank_scale,
            dimensions, stride=1, padding=0, dilation=1, groups=1, bias=True):
        def cp(tensor, rank_scale):
            tensor, ranks = dimensionize(tensor, dimensions, rank_scale)
            return tn.Tensor(tensor, ranks_cp=ranks['ranks_cp'])
        super(CP, self).__init__(in_channels, out_channels, kernel_size,
                rank_scale, cp, stride=stride, padding=padding,
                dilation=dilation, groups=groups, bias=bias)


if __name__ == '__main__':
    for ConvClass in [TensorTrain, Tucker, CP]:
        X = torch.randn(5,16,32,32)
        tnlayer = ConvClass(16,16,3,0.5,2,bias=False)
        tnlayer.reset_parameters()
        print(tnlayer)
        tnlayer.zero_grad()
        y = tnlayer(X)
        l = y.sum()
        l.backward()
        for n,p in tnlayer.named_parameters():
            assert p.requires_grad, n
        assert torch.abs(tnlayer.weight_core_0.grad - tnlayer.tn_weight.cores[0].grad).max() < 1e-5
        # same output on the GPU
        tnlayer, X = tnlayer.cuda(), X.cuda()
        assert torch.abs(tnlayer(X).cpu() - y).max() < 1e-5

    for ConvClass in [TensorTrain, Tucker, CP]:
        X = torch.randn(5,16,32,32)
        tnlayer = ConvClass(16,16,3,0.5,4,bias=False)
        tnlayer.reset_parameters()
        print(tnlayer)
