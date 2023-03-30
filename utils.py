import math
import logging

import torch
import numpy as np
from scipy.linalg import svdvals

RESNET18_FR_BLOCKS_IDX_MAP = {0:6,
                                1:18,
                                2:30,
                                3:48,
                                4:60} # num_fr_blocks:start layer index


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def rank_estimation(epoch, net, adjust_rank_scale=None,
                        est_rank_tracker=None, 
                        layers_to_factorize=None,
                        layer_stable_tracker=None,
                        args=None):
    est_rank_list = []
    ori_rank_list = []

    if epoch == -1:
        # at the very beginning, we calculate the rank adjustment ceof
        adjust_rank_scale = []

    if args.arch == "vgg19":
        if args.mode == "pufferfish":
            fullrank_layer_range = 54
        elif args.mode == "lowrank":
            fullrank_layer_range = 24
        else:
            fullrank_layer_range = 0

        num_layers_remain = len(layer_stable_tracker) - sum(layer_stable_tracker)
        # for this function we count the # of ranks s.t. \sum \sigma_i > frac * \sum \sigma_all 
        for item_index, (param_name, param) in enumerate(net.state_dict().items()):
            if len(param.size()) == 4 and item_index not in range(0, fullrank_layer_range):
                reshaped_param = param.reshape(param.shape[0], -1).detach()
                rank = min(reshaped_param.size()[0], reshaped_param.size()[1])
                ori_rank_list.append(rank)
                #u, s, v = torch.svd(reshaped_param)
                # see how fast this can be
                s = torch.from_numpy(svdvals(reshaped_param.data.cpu().numpy()))
            else:
                continue

            # stable rank
            # https://nickhar.wordpress.com/2012/02/29/lecture-15-low-rank-approximation-of-matrices/
            estimated_rank = int(torch.sum(s ** 2).item() / (torch.max(s).item() ** 2))

            est_rank_list.append(estimated_rank)

            if param_name in layers_to_factorize:
                layer_index_in_factorized_layers = layers_to_factorize.index(param_name)
                if epoch >= 0:
                    est_rank_tracker[layer_index_in_factorized_layers].append(estimated_rank)
                if not layer_stable_tracker[layer_index_in_factorized_layers]: # only look at layers that's not stable
                    layer_stable_tracker[layer_index_in_factorized_layers] = layer_rank_stable_detector(epoch, 
                                            layer_est_ranks=est_rank_tracker[layer_index_in_factorized_layers],
                                            num_layers_remain=num_layers_remain, arch=args.arch)

            if epoch == -1:
                adjust_rank_scale.append(rank/estimated_rank)

            logger.info("#### Epoch: {}, Param index: {}, Param name: {}, Ori rank: {}, Est rank: {}".format(epoch, item_index, 
                        param_name,
                        min(reshaped_param.size()[0], reshaped_param.size()[1]), 
                        estimated_rank))
    elif args.arch == "resnet18":
        if args.mode == "pufferfish":
            fullrank_layer_range = 18
        elif args.mode == "lowrank":
            fullrank_layer_range = 25
        else:
            fullrank_layer_range = 0

        num_layers_remain = len(layer_stable_tracker) - sum(layer_stable_tracker)
        for item_index, (param_name, param) in enumerate(net.state_dict().items()):
            #if len(param.size()) == 4 and item_index not in range(0, 18) and ".shortcut." not in param_name:
            #if len(param.size()) == 4 and item_index not in range(0, 13) and ".shortcut." not in param_name:
            #if len(param.size()) == 4 and item_index not in range(0, 25) and ".shortcut." not in param_name:
            #if len(param.size()) == 4 and item_index not in range(0, 48) and ".shortcut." not in param_name: # three full-rank blocks
            if len(param.size()) == 4 and item_index not in range(0, fullrank_layer_range) and ".shortcut." not in param_name: # three full-rank blocks
                # resize --> svd --> two layer
                # shape_d1, shape_d2, shape_d3, shape_d4 = param.size()
                # reshaped_param2 = param.view(shape_d1*shape_d2, shape_d3*shape_d4)
                # u1, s1, v1 = torch.svd(reshaped_param2)
                # estimated_rank1 = int(torch.sum(s1 ** 2).item() / (torch.max(s1).item() ** 2))

                reshaped_param = param.reshape(param.size()[0], -1)
                rank = min(reshaped_param.size()[0], reshaped_param.size()[1])
                ori_rank_list.append(rank)

                #u, s, v = torch.svd(reshaped_param)
                s = torch.from_numpy(svdvals(reshaped_param.data.cpu().numpy()))
                # vanilla stable rank
                estimated_rank = int(torch.sum(s ** 2).item() / (torch.max(s).item() ** 2))
                
                est_rank_list.append(estimated_rank)

                if param_name in layers_to_factorize:
                    layer_index_in_factorized_layers = layers_to_factorize.index(param_name)
                    if epoch >= 0:
                        est_rank_tracker[layer_index_in_factorized_layers].append(estimated_rank)
                    if not layer_stable_tracker[layer_index_in_factorized_layers]: # only look at layers that's not stable
                        layer_stable_tracker[layer_index_in_factorized_layers] = layer_rank_stable_detector(epoch, 
                                                layer_est_ranks=est_rank_tracker[layer_index_in_factorized_layers],
                                                num_layers_remain=num_layers_remain, arch=args.arch)

                if epoch == -1:
                    adjust_rank_scale.append(rank/estimated_rank)

                logger.info("#### Epoch: {}, Param index: {}, Param name: {}, Ori rank: {}, Est rank: {}".format(
                        epoch, item_index, 
                        param_name,
                        min(reshaped_param.size()[0], reshaped_param.size()[1]), 
                        estimated_rank
                        ))
    else:
        raise NotImplementedError("Unsupported model arch ...")

    if all(layer_stable_tracker):
        switch_epoch = epoch + 1
    else:
        switch_epoch = args.epochs + 1
    logger.info("@@@ Epoch: {}, Layer stable tracker: {}, switch epoch: {}".format(
                epoch, layer_stable_tracker, switch_epoch
        ))


    if epoch == -1:
        return est_rank_list, adjust_rank_scale, switch_epoch
    else:
        adjusted_rank = []
        for er, ars, ori_rank in zip(est_rank_list, adjust_rank_scale, ori_rank_list):
            if args.rank_est_metric == "scaled-stable-rank":
                if int(er * ars) > ori_rank:
                    adjusted_rank.append(ori_rank)
                else:
                    adjusted_rank.append(int(math.ceil(er * ars)))
            elif args.rank_est_metric == "vanilla-stable-rank":
                if int(er * ars) > ori_rank:
                    adjusted_rank.append(ori_rank)
                else:
                    adjusted_rank.append(er)
            else:
                raise NotImplementedError("Unsupported rank estimation metric.")
        return adjusted_rank, switch_epoch


# helper function because otherwise non-empty strings
# evaluate as True
def bool_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def layer_rank_stable_detector(epoch, layer_est_ranks, num_layers_remain, arch):
    if arch in ("resnet18", "vgg19"):
        __predefined_widow_size = 15
        __base_thresold = 0.1
        __threshold_scaler = 2.5
    else:
        __predefined_widow_size = 15
        __base_thresold = 0.1
        __threshold_scaler = 1.0        

    if num_layers_remain == 2:
        __grad_threshold = __base_thresold * __threshold_scaler
    elif num_layers_remain == 1:
        __grad_threshold = __base_thresold  * (__threshold_scaler * 2)
    else:
        __grad_threshold = __base_thresold

    if epoch < __predefined_widow_size:
        return False
    else:
        #smooth_est_ranks = moving_average(layer_est_ranks, w=__predefined_widow_size)
        smooth_est_ranks = exponential_moving_average(layer_est_ranks, points=__predefined_widow_size)
        grad_smooth_est_ranks = np.absolute(np.gradient(smooth_est_ranks))
        logger.info("############## grad_smooth_est_ranks: {}, np.mean(grad_smooth_est_ranks[-11:]): {:.4f}, layers remain: {}, threshold: {}".format(
                            grad_smooth_est_ranks,
                             np.mean(grad_smooth_est_ranks[-11:]), num_layers_remain, __grad_threshold))
        return np.mean(grad_smooth_est_ranks[-11:]) <= __grad_threshold


def decompose_weights(model, low_rank_model, rank_list, rank_ratio, args):
    # SVD version
    reconstructed_aggregator = []
    
    if args.arch == "vgg19":
        layer_counter = 0
        if args.mode == "pufferfish":
            fullrank_layer_range = 54
        else:
            fullrank_layer_range = 24

        for item_index, (param_name, param) in enumerate(model.state_dict().items()):
            if len(param.size()) == 4 and item_index not in range(0, fullrank_layer_range):
                # resize --> svd --> two layer
                param_reshaped = param.view(param.size()[0], -1)
                rank = min(param_reshaped.size()[0], param_reshaped.size()[1])
                u, s, v = torch.svd(param_reshaped)

                if args.mode == "lowrank":
                    sliced_rank = rank_list[layer_counter]
                elif args.mode in ("baseline", "pufferfish"):
                    sliced_rank = int(rank/rank_ratio)
                else:
                    raise NotImplementedError("Unsupported mode ...")

                #u_weight = u * torch.sqrt(s) # alternative implementation: u_weight_alt = torch.mm(u, torch.diag(torch.sqrt(s)))
                #v_weight = torch.sqrt(s) * v # alternative implementation: v_weight_alt = torch.mm(torch.diag(torch.sqrt(s)), v.t())
                # alternative implementation
                u_weight = torch.matmul(u, torch.diag(torch.sqrt(s)))
                v_weight = torch.matmul(torch.diag(torch.sqrt(s)), v.t()).t()
                
                #print("layer indeix: {}, dist u u_alt:{}, dist v v_alt: {}".format(item_index, torch.dist(u_weight, u_weight_alt), torch.dist(v_weight.t(), v_weight_alt)))
                #print("layer indeix: {}, dist u u_alt:{}, dist v v_alt: {}".format(item_index, torch.equal(u_weight, u_weight_alt), torch.equal(v_weight.t(), v_weight_alt)))
                u_weight_sliced, v_weight_sliced = u_weight[:, 0:sliced_rank], v_weight[:, 0:sliced_rank]

                u_weight_sliced_shape, v_weight_sliced_shape = u_weight_sliced.size(), v_weight_sliced.size()

                model_weight_v = u_weight_sliced.view(u_weight_sliced_shape[0],
                                                      u_weight_sliced_shape[1], 1, 1)
                
                model_weight_u = v_weight_sliced.t().view(v_weight_sliced_shape[1], 
                                                          param.size()[1], 
                                                          param.size()[2], 
                                                          param.size()[3])

                reconstructed_aggregator.append(model_weight_u)
                reconstructed_aggregator.append(model_weight_v)
                layer_counter += 1
            else:
                reconstructed_aggregator.append(param)
                
                
        model_counter = 0
        reload_state_dict = {}
        for item_index, (param_name, param) in enumerate(low_rank_model.state_dict().items()):
            # print("#### {}, {}, recons agg: {}, param: {}".format(item_index, param_name, 
            #                                                                         reconstructed_aggregator[model_counter].size(),
            #                                                                        param.size()))
            if "batch_norm" in param_name and "_u" in param_name:
                reload_state_dict[param_name] = param
            else:
                assert (reconstructed_aggregator[model_counter].size() == param.size())
                reload_state_dict[param_name] = reconstructed_aggregator[model_counter]
                model_counter += 1

    elif args.arch == "resnet18":
        layer_counter = 0

        if args.mode == "pufferfish":
            fullrank_layer_range = 18
        elif args.mode == "baseline":
            fullrank_layer_range = RESNET18_FR_BLOCKS_IDX_MAP[0]
        elif args.mode == "lowrank":
            fullrank_layer_range = 25
        else:
            raise NotImplementedError("Unsupported training mode ...")

        for item_index, (param_name, param) in enumerate(model.state_dict().items()):
            #if len(param.size()) == 4 and item_index not in range(0, 18) and ".shortcut." not in param_name:
            #if len(param.size()) == 4 and item_index not in range(0, 13) and ".shortcut." not in param_name:
            #if len(param.size()) == 4 and item_index not in range(0, 25) and ".shortcut." not in param_name: # two full-rank blocks
            #if len(param.size()) == 4 and item_index not in range(0, 48) and ".shortcut." not in param_name: # three full-rank blocks
            if len(param.size()) == 4 and item_index not in range(0, fullrank_layer_range) and ".shortcut." not in param_name: # three full-rank blocks
            #if len(param.size()) == 4 and item_index not in range(0, 60) and ".shortcut." not in param_name:
                # resize --> svd --> two layer
                param_reshaped = param.view(param.size()[0], -1)
                rank = min(param_reshaped.size()[0], param_reshaped.size()[1])
                u, s, v = torch.svd(param_reshaped)

                if args.mode == "lowrank":
                    sliced_rank = rank_list[layer_counter]
                elif args.mode in ("baseline", "pufferfish"):
                    sliced_rank = int(rank/rank_ratio)
                else:
                    raise NotImplementedError("Unsupported mode ...")
                    
                #u_weight = u * torch.sqrt(s)
                #v_weight = torch.sqrt(s) * v
                u_weight = torch.matmul(u, torch.diag(torch.sqrt(s)))
                #v_weight = torch.mm(torch.diag(torch.sqrt(s)), v.t()).t()
                v_weight = torch.matmul(torch.diag(torch.sqrt(s)), v.t()).t()

                u_weight_sliced, v_weight_sliced = u_weight[:, 0:sliced_rank], v_weight[:, 0:sliced_rank]

                u_weight_sliced_shape, v_weight_sliced_shape = u_weight_sliced.size(), v_weight_sliced.size()

                model_weight_v = u_weight_sliced.view(u_weight_sliced_shape[0],
                                                      u_weight_sliced_shape[1], 1, 1)
                
                model_weight_u = v_weight_sliced.t().view(v_weight_sliced_shape[1], 
                                                          param.size()[1], 
                                                          param.size()[2], 
                                                          param.size()[3])

                reconstructed_aggregator.append(model_weight_u)
                reconstructed_aggregator.append(model_weight_v)
                layer_counter += 1
            else:
                reconstructed_aggregator.append(param)
                
        model_counter = 0
        reload_state_dict = {}
        for item_index, (param_name, param) in enumerate(low_rank_model.state_dict().items()):
            #print("#### {}, {}, recons agg: {}， param: {}".format(item_index, param_name, 
            #                                                                        reconstructed_aggregator[model_counter].size(),
            #                                                                       param.size()))
            if ".bn1_u." in param_name or ".bn2_u." in param_name:
                reload_state_dict[param_name] = param
            else:
                assert (reconstructed_aggregator[model_counter].size() == param.size())
                reload_state_dict[param_name] = reconstructed_aggregator[model_counter]
                model_counter += 1
    else:
        raise NotImplementedError("Unsupported model arch ...")
    
    low_rank_model.load_state_dict(reload_state_dict)
    return low_rank_model


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def apply_fd(model, weight_decay=1e-5, factor_list=()):
    v_treated_flag =False
    for param_name, param in model.named_parameters():
        if param_name in factor_list:
            if "_u.weight" in param_name:
                v_name = param_name.rstrip(".weight").rstrip("_u") + "_v.weight"
                v_weight, v_weight_shape = model.state_dict()[v_name], model.state_dict()[v_name].size() # size: (#out, r, 1, 1)
                u_weight, u_weight_shape = param, param.size() # size (r, #in, k, k)
                u_weight = u_weight.data.reshape(u_weight_shape[0], u_weight_shape[1]*u_weight_shape[2]*u_weight_shape[3])
                v_weight = v_weight.data.reshape(v_weight_shape[0], v_weight_shape[1])
                vu_res = torch.matmul(v_weight, u_weight)
                frob_grad_u = torch.matmul(v_weight.T, vu_res
                                                ).reshape(u_weight_shape) # size (r, #in * k * k)
                param.grad += weight_decay * frob_grad_u

                #v_name = param_name.rstrip(".weight").rstrip("_u") + "_v.weight"
                #v_weight, v_weight_shape = model.state_dict()[v_name], model.state_dict()[v_name].size() # size: (#out, r, 1, 1)
                #u_weight, u_weight_shape = param, param.size() # size (r, #in, k, k)
                #u_weight = u_weight.reshape(u_weight_shape[0], u_weight_shape[1]*u_weight_shape[2]*u_weight_shape[3])

                #v_weight = v_weight.reshape(v_weight_shape[0], v_weight_shape[1])
                #frob_grad_u = torch.chain_matmul(v_weight.T, v_weight, u_weight 
                #                                ).reshape(u_weight_shape) # size (r, #in * k * k)
                #param.grad += weight_decay * frob_grad_u
            elif "_v.weight" in param_name:
                #u_name = param_name.rstrip(".weight").rstrip("_v") + "_u.weight"
                #u_weight, u_weight_shape = model.state_dict()[u_name], model.state_dict()[u_name].size() # size: (r, #in, k, k)
                #u_weight = u_weight.reshape(u_weight_shape[0], u_weight_shape[1]*u_weight_shape[2]*u_weight_shape[3])
                #v_weight, v_weight_shape = param, param.size() # size (#out, r, 1, 1)
                #v_weight = v_weight.reshape(v_weight_shape[0], v_weight_shape[1])
                frob_grad_v = torch.matmul(vu_res, u_weight.T).reshape(v_weight_shape) # size (#out, r * 1 * 1)
                #frob_grad_v = torch.matmul(v_weight, torch.matmul(u_weight, u_weight.T)
                #                                ).reshape(v_weight_shape) # size (#out, r * 1 * 1)
                param.grad += weight_decay * frob_grad_v


# we track the norm of the model weights:
def norm_calculator(model):
    model_norm = 0
    for param_index, param in enumerate(model.parameters()):
        model_norm += torch.norm(param) ** 2
    return torch.sqrt(model_norm).item()


def param_counter(model):
    num_params = 0
    for param_index, (param_name, param) in enumerate(model.named_parameters()):
        num_params += param.numel()
    return num_params


def exponential_moving_average(signal, points, smoothing=2):
    """
    from: https://leofinance.io/@chasmic-cosm/calculating-the-exponential-moving-average-in-python

    Calculate the N-point exponential moving average of a signal

    Inputs:
        signal: numpy array -   A sequence of price points in time
        points:      int    -   The size of the moving average
        smoothing: float    -   The smoothing factor

    Outputs:
        ma:     numpy array -   The moving average at each point in the signal
    """

    weight = smoothing / (points + 1)
    ema = np.zeros(len(signal))
    ema[0] = signal[0]

    for i in range(1, len(signal)):
        ema[i] = (signal[i] * weight) + (ema[i - 1] * (1 - weight))

    return ema