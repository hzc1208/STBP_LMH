import torch
import torch.nn as nn
import math
from modules import *


def isActivation(name):
    if 'relu' in name.lower() or 'qcfs' in name.lower():
        return True
    return False


def sineInc(n, N):
    return (1.0 + math.sin(math.pi * (float(n) / N - 0.5))) / 2


def replace_QCFS_by_Lneuron(model, use_TEBN, use_SEW, T):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_QCFS_by_Lneuron(module, use_TEBN, use_SEW, T)
        if isActivation(module.__class__.__name__.lower()):
            model._modules[name] = LearnableNeuron(scale=module.up.item(), use_TEBN=use_TEBN, use_SEW=use_SEW, T=T)
    return model


def replace_QCFS_by_Oneuron(model, use_TEBN, use_SEW, T, time_slice):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_QCFS_by_Oneuron(module, use_TEBN, use_SEW, T, time_slice)
        if isActivation(module.__class__.__name__.lower()):
            model._modules[name] = OnlineNeuron(use_TEBN=use_TEBN, use_SEW=use_SEW, T=T, time_slice=time_slice)
    return model


def replace_Lneuron_by_QCFS(model, T):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_Lneuron_by_QCFS(module, T)
        if module.__class__.__name__ == 'PRNeuron':
            thresh = module.v_threshold.item()
            model._modules[name] = QCFS(up=thresh, t=T)
    return model


def replace_activation_by_QCFS(model, T, thresh):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_activation_by_QCFS(module, T, thresh)
        if isActivation(module.__class__.__name__.lower()):
            model._modules[name] = QCFS(up=thresh, t=T)
    return model


def replace_maxpool2d_by_avgpool2d(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_maxpool2d_by_avgpool2d(module)
        if module.__class__.__name__ == 'MaxPool2d':
            model._modules[name] = nn.AvgPool2d(kernel_size=module.kernel_size,
                                                stride=module.stride,
                                                padding=module.padding)
    return model
    
    
def replace_batchnorm2d_by_TEBN(model, T):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_batchnorm2d_by_TEBN(module, T)
        if module.__class__.__name__ == 'BatchNorm2d':
            model._modules[name] = TEBN(num_features=module.num_features, T=T)
    return model
    
    
def replace_Conv2d_by_PConv(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_Conv2d_by_PConv(module)
        if module.__class__.__name__ == 'Conv2d':
            if module.kernel_size[0] == 3:
                model._modules[name] = conv3x3(module.in_channels, module.out_channels, module.stride[0])
            elif module.kernel_size[0] == 1:
                model._modules[name] = conv1x1(module.in_channels, module.out_channels, module.stride[0])
        elif module.__class__.__name__ == 'Linear':
            model._modules[name] = conv1x1(module.in_features, module.out_features, 1, True)
             
    return model


def set_flat_width(model, tot_flat_width, current_times, total_times):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            set_flat_width(module, tot_flat_width, current_times, total_times)
        if hasattr(module, 'setFlatWidth'):
            module.setFlatWidth(sineInc(current_times, total_times) * tot_flat_width)
    return model    

    
def print_sparse_rate(model, message):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            print_sparse_rate(module, message)
        if hasattr(module, 'getSparsity'):
            zero_cnt, numel = module.getSparsity()
            message[0] += zero_cnt
            message[1] += numel   


def print_message(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            print_message(module)
        if 'Neuron' in module.__class__.__name__:
            print(f'v_threshold = {module.v_threshold}, alpha_1 = {module.alpha_1.sigmoid()-0.5}, beta_1 = {module.beta_1.sigmoid()-0.5}, alpha_2 = {module.alpha_2.sigmoid()+0.5}, beta_2 = {module.beta_2.sigmoid()+0.5}')
                
        #if module.__class__.__name__ == 'TEBN':
            #print(f'TEBN = {module.p.detach().cpu().numpy()}')


def reset_net(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            reset_net(module)
        if 'Neuron' in module.__class__.__name__:
            module.reset()
    return model


def error(info):
    print(info)
    exit(1)
