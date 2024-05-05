from utils import *
from NetworkFunction import *
import argparse
from dataprocess import PreProcess_Cifar10, PreProcess_Cifar100, PreProcess_TinyImageNet, PreProcess_ImageNet #, build_dvscifar
from Models.ResNet import *
from Models.VGG import *
from Models.SEWResNet import *
import torch
import random
import os
import numpy as np


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='CIFAR100', help='Dataset name')
    parser.add_argument('--datadir', type=str, default='/home/user/main/', help='Directory where the dataset is saved')
    parser.add_argument('--savedir', type=str, default='/home/user/main/', help='Directory where the model is saved')
    parser.add_argument('--load_model_name', type=str, default='None', help='The name of the loaded ANN model')
    parser.add_argument('--trainann_epochs', type=int, default=200, help='Training Epochs of ANNs')
    parser.add_argument('--trainsnn_epochs', type=int, default=50, help='Training Epochs of SNNs')
    parser.add_argument('--net_arch', type=str, default='vgg16', help='Network Architecture')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--batchsize', type=int, default=100, help='Batch size')
    parser.add_argument('--L', type=int, default=4, help='Quantization level of QCFS')
    parser.add_argument('--sim_len', type=int, default=32, help='Simulation length of SNNs')
    parser.add_argument('--time_step', type=int, default=4, help='Training Time-steps for SNNs')
    parser.add_argument('--time_slice', type=int, default=2, help='Online Training Time-slice for SNNs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--lr2', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--direct_training', action='store_true', default=False)
    parser.add_argument('--direct_inference', action='store_true', default=False)
    parser.add_argument('--train_dir', type=str, default='/home/data/ImageNet2012/ILSVRC2012_train', help='Directory where the ImageNet train dataset is saved')
    parser.add_argument('--test_dir', type=str, default='/home/data/ImageNet2012/ILSVRC2012_val', help='Directory where the ImageNet test dataset is saved')    
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dev', type=str, default='0')
    parser.add_argument('--use_TET', action='store_true', default=False)
    parser.add_argument('--use_TEBN', action='store_true', default=False)
    parser.add_argument('--pruning', action='store_true', default=False)
    parser.add_argument('--flat_width', type=float, default=0.2, help='pruning degree')
    parser.add_argument('--online_training', action='store_true', default=False)
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.dev
    
    torch.backends.cudnn.benchmark = True
    _seed_ = args.seed
    random.seed(_seed_)
    os.environ['PYTHONHASHSEED'] = str(_seed_)
    torch.manual_seed(_seed_)
    torch.cuda.manual_seed(_seed_)
    torch.cuda.manual_seed_all(_seed_)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(_seed_)
    
    use_SEW, dvs_data = False, False

    if args.dataset == 'CIFAR10':
        train, test = PreProcess_Cifar10(args.datadir, args.batchsize)
        cls = 10
    elif args.dataset == 'CIFAR100':
        train, test = PreProcess_Cifar100(args.datadir, args.batchsize)
        cls = 100
    elif args.dataset == 'TinyImageNet':
        train, test = PreProcess_TinyImageNet(args.datadir, args.batchsize)
        cls = 200
    elif args.dataset == 'ImageNet':
        train, test = PreProcess_ImageNet(args.batchsize, train_dir=args.train_dir, test_dir=args.test_dir)
        cls = 1000
    elif args.dataset == 'DVSCIFAR':
        dvs_data = True
        train, test = build_dvscifar('/home/data/public/cifar10dvs/cifar10_dvs/', args.batchsize)
        cls = 10
    else:
        error('unable to find dataset ' + args.dataset)
    
    
    if args.net_arch == 'resnet20':
        model = resnet20(num_classes=cls)
    elif args.net_arch == 'resnet18':
        model = resnet18(num_classes=cls)
    elif args.net_arch == 'resnet19':
        model = resnet19(num_classes=cls)
    elif args.net_arch == 'wideresnet19':
        model = wideresnet19(num_classes=cls)
    elif args.net_arch == 'resnet34':
        model = resnet34(num_classes=cls)
    elif args.net_arch == 'vgg16':
        model = vgg16(num_classes=cls)
    elif args.net_arch == 'vgg13':
        model = vgg13(num_classes=cls)
    elif args.net_arch == 'vggdvs':
        model = vggdvs(num_classes=cls)
    elif args.net_arch == 'cifarnet':
        model = CIFAR10Net(num_classes=cls)
    elif args.net_arch == 'spiking-resnet18':
        use_SEW = True
        model = spiking_resnet18(num_classes=cls, zero_init_residual=True, connect_f='ADD')
    elif args.net_arch == 'sew-resnet34':
        use_SEW = True
        model = sew_resnet34(num_classes=cls, zero_init_residual=True, connect_f='ADD')
    else:
        error('unable to find model ' + args.arch)
    
    model = replace_maxpool2d_by_avgpool2d(model)
    
    if args.use_TEBN is True:
        model = replace_batchnorm2d_by_TEBN(model, args.time_step)
        if args.time_step != args.sim_len:
            error('under TEBN framework, args.time_step = args.sim_len')
            
    if dvs_data is True:
        args.use_TEBN = True
        print(f'args.use_TEBN = {args.use_TEBN}')
            
    if args.pruning is True:
        model = replace_Conv2d_by_PConv(model)
        model = set_flat_width(model, args.flat_width, 1, 1)
    
    print(model)

    if args.direct_training is True:
        model = replace_activation_by_QCFS(model, args.L, 8.)
    else:
        model = replace_activation_by_QCFS(model, args.L, 1.)
    
    if args.direct_inference is not True:
        if args.load_model_name != 'None':
            print(f'=== Load Pretrained ANNs ===')
            model.load_state_dict(torch.load(args.load_model_name + '.pth'))  
        if args.direct_training is True:
            print(f'=== Start Training ANNs ===')
            save_name = args.savedir + args.dataset + '_' + args.net_arch + '_L' + str(args.L) + '_QuantizedANN.pth'
            model = train_ann(train, test, model, epochs=args.trainann_epochs, lr=args.lr, wd=args.wd, device=args.device, save_name=save_name, T=args.L)
        
        '''
        new_acc = eval_snn(test, model, sim_len=args.sim_len, device=args.device, use_TEBN=args.use_TEBN, use_SEW=use_SEW, dvs_data=dvs_data)
        print(new_acc)
        
        t = 1
        while t < args.sim_len:
            print(f'time step {t}, Accuracy = {(new_acc[t-1]):.4f}')
            t *= 2
        print(f'time step {args.sim_len}, Accuracy = {(new_acc[args.sim_len-1]):.4f}')
        '''
        
        print(f'=== Start Training SNNs ===')
              
        save_name = args.savedir + args.dataset + '_' + args.net_arch + '_T' + str(args.time_step) + '_' + '_TET_' + str(args.use_TET) + '_TEBN_' + str(args.use_TEBN) + '_pruning_' + str(args.pruning) + '_online_' + str(args.online_training) + '_lr' + str(args.lr2) + '_epoch' + str(args.trainsnn_epochs) + '_SNN.pth'
        
        if args.online_training is True:
            model = replace_QCFS_by_Oneuron(model, args.use_TEBN, use_SEW, args.time_step, args.time_slice)
            train_snn_online(train, test, model, epochs=args.trainsnn_epochs, lr=args.lr2, wd=args.wd, device=args.device, save_name=save_name, time_step=args.time_step, time_slice=args.time_slice, use_TEBN=args.use_TEBN, use_TET=args.use_TET)
        else:
            model = replace_QCFS_by_Lneuron(model, args.use_TEBN, use_SEW, args.time_step)
            train_snn(train, test, model, epochs=args.trainsnn_epochs, lr=args.lr2, wd=args.wd, device=args.device, save_name=save_name, time_step=args.time_step, use_TET=args.use_TET, use_TEBN=args.use_TEBN, pruning=args.pruning, tot_flat_width=args.flat_width, use_SEW=use_SEW, dvs_data=dvs_data)
        
        new_acc = eval_snn(test, model, sim_len=args.sim_len, device=args.device, use_TEBN=args.use_TEBN, use_SEW=use_SEW, dvs_data=dvs_data)
        print(new_acc)
        t = 1
        while t < args.sim_len:
            print(f'time step {t}, Accuracy = {(new_acc[t-1]):.4f}')
            t *= 2
        print(f'time step {args.sim_len}, Accuracy = {(new_acc[args.sim_len-1]):.4f}')
        
    else:
        if args.online_training is True:
            model = replace_QCFS_by_Oneuron(model, args.use_TEBN, use_SEW, args.time_step, args.time_slice)
        else:
            model = replace_QCFS_by_Lneuron(model, args.use_TEBN, use_SEW, args.time_step)
        print(f'=== Load Pretrained SNNs ===')
        model.load_state_dict(torch.load(args.load_model_name + '.pth'))   
        print_message(model)
        new_acc = eval_snn(test, model, sim_len=args.sim_len, device=args.device, use_TEBN=args.use_TEBN, use_SEW=use_SEW, dvs_data=dvs_data)
        print(new_acc)
        t = 1
        while t < args.sim_len:
            print(f'time step {t}, Accuracy = {(new_acc[t-1]):.4f}')
            t *= 2
        print(f'time step {args.sim_len}, Accuracy = {(new_acc[args.sim_len-1]):.4f}')