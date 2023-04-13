import pdb

import torch
from torch import nn
from .models import resnet


def generate_model(model_name="resnet", model_depth=50, resnet_shortcut="B", no_cuda=False, pretrain_path=None):
    assert model_name in [
        'resnet'
    ]

    if model_name == 'resnet':
        assert model_depth in [10, 18, 34, 50, 101, 152, 200]
        
        if model_depth == 10:
            model = resnet.resnet10(
                shortcut_type=resnet_shortcut,
                no_cuda=no_cuda)
        elif model_depth == 18:
            model = resnet.resnet18(
                shortcut_type=resnet_shortcut,
                no_cuda=no_cuda)
        elif model_depth == 34:
            model = resnet.resnet34(
                shortcut_type=resnet_shortcut,
                no_cuda=no_cuda)
        elif model_depth == 50:
            model = resnet.resnet50(
                shortcut_type=resnet_shortcut,
                no_cuda=no_cuda)
        elif model_depth == 101:
            model = resnet.resnet101(
                shortcut_type=resnet_shortcut,
                no_cuda=no_cuda)
        elif model_depth == 152:
            model = resnet.resnet152(
                shortcut_type=resnet_shortcut,
                no_cuda=no_cuda)
        elif model_depth == 200:
            model = resnet.resnet200(
                shortcut_type=resnet_shortcut,
                no_cuda=no_cuda)

    """
    if not no_cuda:
        if len(opt.gpu_id) > 1:
            model = model.cuda() 
            model = nn.DataParallel(model, device_ids=opt.gpu_id)
            net_dict = model.state_dict() 
        else:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"]=str(opt.gpu_id[0])
            model = model.cuda() 
            model = nn.DataParallel(model, device_ids=None)
            net_dict = model.state_dict()
    else:
        net_dict = model.state_dict()
    
    # load pretrain
    if opt.phase != 'test' and opt.pretrain_path:
        print ('loading pretrained model {}'.format(opt.pretrain_path))
        pretrain = torch.load(opt.pretrain_path)
        pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
         
        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)

        new_parameters = [] 
        for pname, p in model.named_parameters():
            for layer_name in opt.new_layer_names:
                if pname.find(layer_name) >= 0:
                    new_parameters.append(p)
                    break

        new_parameters_id = list(map(id, new_parameters))
        base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
        parameters = {'base_parameters': base_parameters, 
                      'new_parameters': new_parameters}

        return model, parameters
    """

    # load pretrain
    if pretrain_path:
        print('loading pretrained model {}'.format(pretrain_path))
        net_dict = model.state_dict()
        pretrain = torch.load(pretrain_path)
        pretrain_dict = {k[7:]: v for k, v in pretrain['state_dict'].items() if k[7:] in net_dict.keys()}

        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)

        return model

    return model #, model.parameters()
