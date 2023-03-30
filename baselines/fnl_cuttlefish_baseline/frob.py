# // Copyright (c) Microsoft Corporation.
# // Licensed under the MIT license.
import math
import pdb
import torch
from torch import nn
from torch.nn import functional as F


def frobgrad(matrices):

    assert type(matrices) == list
    assert 1 <= len(matrices) <= 3
    output = [None for _ in matrices]
    if len(matrices) == 1:
        output[0] = matrices[0].clone()
    else:
        if len(matrices) == 2:
            U, VT = matrices
            UM, MVT = matrices
        else:
            U, M, VT = matrices
            UM, MVT = torch.matmul(U, M), torch.matmul(M, VT)
            output[1] = torch.chain_matmul(U.T, U, M, VT, VT.T)
        output[0] = torch.chain_matmul(U, MVT, MVT.T)
        output[-1] = torch.chain_matmul(UM.T, UM, VT)
    return output


def batch_frobgrad(tensors):

    assert type(tensors) == list
    assert 1 <= len(tensors) <= 3
    output = [None for _ in tensors]
    if len(tensors) == 1:
        output[0] = tensors[0].clone()
    else:
        if len(tensors) == 2:
            U, VT = tensors
            UM, MVT = tensors
        else:
            U, M, VT = tensors
            UM, MVT = torch.bmm(U, M), torch.bmm(M, VT)
            output[1] = torch.bmm(torch.bmm(torch.bmm(U.transpose(1, 2), U), M),
                                  torch.bmm(VT, VT.transpose(1, 2)))
        output[0] = torch.bmm(U, torch.bmm(MVT, MVT.transpose(1, 2)))
        output[-1] = torch.bmm(torch.bmm(UM.transpose(1, 2), UM), VT)
    return output


def apply_frobdecay(name, module, skiplist=[]):

    if hasattr(module, 'frobgrad'):
        return not any(name[-len(entry):] == entry for entry in skiplist)
    return False


def frobdecay(model, coef=0.0, skiplist=[], local_rank=-1, world_size=1):
    
    for i, module in enumerate(module for name, module in model.named_modules() 
                               if apply_frobdecay(name, module, skiplist=skiplist)):
        if local_rank == -1 or local_rank == i % world_size:
            module.frobgrad(coef=coef)


def spectral_init(weight, rank):

    U, S, V = torch.svd(weight)
    sqrtS = torch.diag(torch.sqrt(S[:rank]))
    return torch.matmul(U[:,:rank], sqrtS), torch.matmul(V[:,:rank], sqrtS).T


def batch_spectral_init(U, VT):

    for i, (u, vt) in enumerate(zip(U, VT)):
        U[i], VT[i] = spectral_init(torch.matmul(u, vt), U.shape[-1])


def frobenius_norm(U, VT):

    m, r = U.shape
    r, n = VT.shape
    if m * n * r < r * r * (m + n):
        return torch.norm(torch.matmul(U, VT))
    return torch.sqrt(torch.trace(torch.chain_matmul(VT, VT.T, U.T, U)))


def non_orthogonality(U, VT):

    return [torch.norm(matrix - torch.diag(torch.diag(matrix))) / torch.norm(matrix)
            for matrix in [torch.matmul(U.T, U), torch.matmul(VT, VT.T)]]


class FactorizedLinear(nn.Module):

    def __init__(self, linear, rank_scale=1.0, init='spectral'):

        super(FactorizedLinear, self).__init__()
        self.shape = linear.weight.shape
        dim1, dim2 = self.shape
        self.rank = int(round(rank_scale * min(dim1, dim2))) #int(round(rank_scale * dim1))
        self.U = nn.Parameter(torch.zeros(dim1, self.rank))
        self.VT = nn.Parameter(torch.zeros(self.rank, dim2))

        if init == 'spectral':
            U, VT = spectral_init(linear.weight, self.rank)
            self.U.data[:,:U.shape[1]] = U
            self.VT.data[:VT.shape[0],:] = VT
        else:
            init(self.U.data)
            init(self.VT.data)

        self.bias = linear.bias
        delattr(linear, 'weight')
        delattr(linear, 'bias')

    def forward(self, x):

        return F.linear(F.linear(x, self.VT), self.U, bias=self.bias)

    def frobgrad(self, coef=0.0):

        if coef:
            Ugrad, VTgrad = frobgrad([self.U.data, self.VT.data])
            if self.U.grad is None:
                self.U.grad = coef * Ugrad
            else:
                self.U.grad += coef * Ugrad
            if self.VT.grad is None:
                self.VT.grad = coef * VTgrad
            else:
                self.VT.grad += coef * VTgrad


class FactorizedConv(nn.Module):

    def __init__(self, conv, rank_scale=1.0, init='spectral', square=False, one_conv='auto', square_init=lambda I: I):

        super(FactorizedConv, self).__init__()
        self.shape = conv.weight.shape
        a, b, c, d = self.shape
        dim1, dim2 = a * c, b * d
        self.rank = max(int(round(rank_scale * dim1)), 1) #int(round(rank_scale * min(dim1, dim2)))
        self.U = nn.Parameter(torch.zeros(dim1, self.rank))
        self.VT = nn.Parameter(torch.zeros(self.rank, dim2))

        if init == 'spectral':
            weight = conv.weight.data.reshape(dim1, dim2)
            U, VT = spectral_init(weight, self.rank)
            self.U.data[:,:U.shape[1]] = U
            self.VT.data[:VT.shape[0],:] = VT
        else:
            init(self.U.data)
            init(self.VT.data)

        self.M = nn.Parameter(square_init(torch.eye(self.rank))) if square else None 
        self.one_conv = rank_scale >= 1.0 if one_conv == 'auto' else one_conv

        self.kwargs = {}
        for name in ['bias', 'stride', 'padding', 'dilation', 'groups']:
            attr = getattr(conv, name)
            setattr(self, name, attr)
            self.kwargs[name] = attr
        delattr(conv, 'weight')
        delattr(conv, 'bias')

    def forward(self, x):

        out_channels, in_channels, ks1, ks2 = self.shape
        MVT = self.VT if self.M is None else torch.matmul(self.M, self.VT)

        if self.one_conv:
            return F.conv2d(x, 
                            torch.matmul(self.U, MVT).reshape(out_channels, ks1, in_channels, ks2).transpose(1, 2), # torch.matmul(self.U, MVT).reshape(self.shape), 
                            **self.kwargs)

        x = F.conv2d(x, 
                     MVT.T.reshape(in_channels, ks2, 1, self.rank).permute(3, 0, 2, 1), 
                     None,
                     stride=(1, self.stride[1]),
                     padding=(0, self.padding[1]),
                     dilation=(1, self.dilation[1]),
                     groups=self.groups).contiguous()

        return F.conv2d(x, 
                        self.U.reshape(out_channels, ks1, self.rank, 1).permute(0, 2, 1, 3),
                        self.bias,
                        stride=(self.stride[0], 1),
                        padding=(self.padding[0], 0),
                        dilation=(self.dilation[0], 1),
                        groups=self.groups).contiguous()

    def frobgrad(self, coef=0.0):

        if coef:
            if self.M is None:
                Ugrad, VTgrad = frobgrad([self.U.data, self.VT.data])
            else:
                Ugrad, Mgrad, VTgrad = frobgrad([self.U.data, self.M.data, self.VT.data])
                if self.M.grad is None:
                    self.M.grad = coef * Mgrad
                else:
                    self.M.grad += coef * Mgrad

            if self.U.grad is None:
                self.U.grad = coef * Ugrad
            else:
                self.U.grad += coef * Ugrad
            if self.VT.grad is None:
                self.VT.grad = coef * VTgrad
            else:
                self.VT.grad += coef * VTgrad


def patch_module(module, childname, replacement, **kwargs):

    child = module
    for name in childname.split('.'):
        child = getattr(child, name)
    setattr(module, childname.split('.')[0], replacement(child, **kwargs))


if __name__ == '__main__':

    print('Testing Two Matrices')
    odim, rank, idim = 256, 64, 128
    U, VT = torch.randn(odim, rank), torch.randn(rank, idim)
    dU, dVT = frobgrad([U, VT])
    dUtest = torch.chain_matmul(U, VT, VT.T)
    print(torch.norm(dU - dUtest) / torch.norm(dUtest))
    dVTtest = torch.chain_matmul(VT.T, U.T, U).T
    print(torch.norm(dVT - dVTtest)  / torch.norm(dVTtest))

    print('Testing Three Matrices')
    M = torch.randn(rank, rank)
    dU, dM, dVT = frobgrad([U, M, VT])
    dUtest = torch.chain_matmul(U, M, VT, VT.T, M.T)
    print(torch.norm(dU - dUtest) / torch.norm(dUtest))
    dMtest = torch.chain_matmul(U.T, U, M, VT, VT.T)
    print(torch.norm(dM - dMtest) / torch.norm(dMtest))
    dVTtest = torch.chain_matmul(VT.T, M.T, U.T, U, M).T
    print(torch.norm(dVT - dVTtest) / torch.norm(dVTtest))

    print('Testing Two Matrix Batches')
    num = 8
    U, VT = torch.randn(num, odim, rank), torch.randn(num, rank, idim)
    dU, dVT = batch_frobgrad([U, VT])
    sliced = [frobgrad([u, vt]) for u, vt in zip(U, VT)]
    dUtest = torch.stack([dudvt[0] for dudvt in sliced])
    print(torch.norm(dU - dUtest) / torch.norm(dUtest))
    dVTtest = torch.stack([dudvt[1] for dudvt in sliced])
    print(torch.norm(dVT - dVTtest) / torch.norm(dVTtest))

    print('Testing Three Matrix Batches')
    M = torch.randn(num, rank, rank)
    dU, dM, dVT = batch_frobgrad([U, M, VT])
    sliced = [frobgrad([u, m, vt]) for u, m, vt in zip(U, M, VT)]
    dUtest = torch.stack([dudmdvt[0] for dudmdvt in sliced])
    print(torch.norm(dU - dUtest) / torch.norm(dUtest))
    dMtest = torch.stack([dudmdvt[1] for dudmdvt in sliced])
    print(torch.norm(dM - dMtest) / torch.norm(dMtest))
    dVTtest = torch.stack([dudmdvt[2] for dudmdvt in sliced])
    print(torch.norm(dVT - dVTtest) / torch.norm(dVTtest))

    print('Testing Convolutions')
    batch_size, kernel_size, dim = 4, 3, 28
    errs = []
    for in_channels, out_channels in [(3, 64), (64, 64)]:
        for stride in range(1, 3):
            for padding in range(2):
                for rank_scale in [0.01, 0.5, 1.0, 4.0]:
                    for init in ['spectral', nn.init.kaiming_normal_]:
                        for square in [False, True]:
                            X = torch.randn(batch_size, in_channels, dim, dim)
                            conv = FactorizedConv(nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size),
                                                            stride=stride, padding=padding, bias=False),
                                                  rank_scale=rank_scale,
                                                  init=init,
                                                  square=square)
                            if square:
                                conv.M.data = torch.randn(conv.M.shape)
                            A = conv(X)
                            conv.one_conv = not conv.one_conv
                            B = conv(X)
                            err = (torch.norm(A - B) / torch.norm(B)).item()
                            print(round(err, 6), end='\t')
                            errs.append(err)
                print()
    print(sum(errs) / len(errs))

    m, n, = 10000, 5000
    for r in [400, 4000]:
        print('Testing Frobenius Norm Rank', r)
        U, VT = torch.randn(m, r), torch.randn(r, n)
        precise = torch.norm(torch.matmul(U.to(torch.float64), VT.to(torch.float64))).to(torch.float32)
        print('Direct:', abs(precise - torch.norm(torch.matmul(U, VT))) / precise)
        print('Custom:', abs(precise - frobenius_norm(U, VT)) / precise)
