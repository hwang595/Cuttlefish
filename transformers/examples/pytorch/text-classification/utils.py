import torch

def param_counter(model):
    param_count = 0
    for param_index, (param_name, param) in enumerate(model.named_parameters()):
        if param.requires_grad:
            param_count += param.numel()
    return param_count
