import re
import torch
import torch.nn.functional as F
from torch.utils import model_zoo
from models.blocks import Conv
from models.wide_resnet import WRN_50_2

from collections import OrderedDict

def all_equal(iterable_1, iterable_2):
    return all([x == y for x,y in zip(iterable_1, iterable_2)])

# functional model definition from functional zoo: https://github.com/szagoruyko/functional-zoo/blob/master/imagenet-validation.py#L27-L47
def define_model(params):
    def conv2d(input, params, base, stride=1, pad=0):
        return F.conv2d(input, params[base + '.weight'],
                        params[base + '.bias'], stride, pad)

    def group(input, params, base, stride, n):
        o = input
        for i in range(0,n):
            b_base = ('%s.block%d.conv') % (base, i)
            x = o
            o = conv2d(x, params, b_base + '0')
            o = F.relu(o)
            o = conv2d(o, params, b_base + '1', stride=i==0 and stride or 1, pad=1)
            o = F.relu(o)
            o = conv2d(o, params, b_base + '2')
            if i == 0:
                o += conv2d(x, params, b_base + '_dim', stride=stride)
            else:
                o += x
            o = F.relu(o)
        return o
    
    # determine network size by parameters
    blocks = [sum([re.match('group%d.block\d+.conv0.weight'%j, k) is not None
                   for k in params.keys()]) for j in range(4)]

    def f(input, params):
        o = F.conv2d(input, params['conv0.weight'], params['conv0.bias'], 2, 3)
        o = F.relu(o)
        o = F.max_pool2d(o, 3, 2, 1)
        o_g0 = group(o, params, 'group0', 1, blocks[0])
        o_g1 = group(o_g0, params, 'group1', 2, blocks[1])
        o_g2 = group(o_g1, params, 'group2', 2, blocks[2])
        o_g3 = group(o_g2, params, 'group3', 2, blocks[3])
        o = F.avg_pool2d(o_g3, 7, 1, 0)
        o = o.view(o.size(0), -1)
        o = F.linear(o, params['z.fc.weight'], params['z.fc.bias'])
        return o

    return f

if __name__ == '__main__':
    # our model definition
    net = WRN_50_2(Conv)
    # load parameters from model zoo
    params = model_zoo.load_url('https://s3.amazonaws.com/modelzoo-networks/wide-resnet-50-2-export-5ae25d50.pth')
    # otherwise the ordering will be messed up
    params['z.fc.weight'] = params.pop('fc.weight')
    params['z.fc.bias'] = params.pop('fc.bias')
    params = sorted(params.items()) # list of tuples, in order
    # make state_dict from model_zoo parameters
    state_dict = OrderedDict()
    w_i, b_i = 0, 0
    for n,p in net.state_dict().items():
        if 'weight' in n and 'bn' not in n:
            while 'weight' not in params[w_i][0]:
                w_i += 1
            k, v = params[w_i]
            print(k, " == ", n)
            assert all_equal(v.shape, p.size()), f"{v.shape} =/= {p.size()}"
            state_dict[n] = v
            w_i += 1
        elif 'bias' in n:
            while 'bias' not in params[b_i][0]:
                b_i += 1
            k, v = params[b_i]
            print(k, " == ", n)
            assert all_equal(v.shape, p.size()), f"{v.shape} =/= {p.size()}"
            state_dict[n] = v
            b_i += 1
        else:
            state_dict[n] = p
    assert max(w_i, b_i) == len(params) # all params are matched

    # test if this is the same as the functional implementation
    params = OrderedDict(params)
    f = define_model(params)
    net.load_state_dict(state_dict)
    net.eval()
    X = torch.randn(2,3,224,224)
    func_out, net_out = f(X, params), net(X)[0]
    error = torch.abs(func_out - net_out)
    assert error.max() < 1e-3, "%f"%error.max()
    print("Output given random input is equal within %f"%error.max())

    # now save a new checkpoint file, with correct saved terms
    save_dict = {}
    save_dict['net'] = state_dict
    save_dict['epoch'] = 100
    save_dict['conv'] = 'Conv'
    save_dict['blocktype'] = None
    save_dict['module'] = None

    torch.save(save_dict, 'checkpoints/wrn_50_2.imagenet.modelzoo.t7')
