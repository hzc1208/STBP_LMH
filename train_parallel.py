from utils import *
import argparse
from Models.ResNet import *
from Models.SEWResNet import *
import torch
from torch.utils.data import DataLoader
import random
import os
import numpy as np
from torchvision import datasets, transforms
import logging
from dataprocess import PreProcess_Cifar100, ImageNetPolicy
from torch.cuda import amp
from timm.data import Mixup


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def init_distributed(distributed_init_mode):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        return False, 0, 1, 0

    torch.cuda.set_device(local_rank)
    backend = 'nccl'
    print('Distributed init rank {}'.format(rank))
    torch.distributed.init_process_group(backend=backend, init_method=distributed_init_mode,
                                         world_size=world_size, rank=rank)
    return True, rank, world_size, local_rank


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= nprocs
    return rt


def load_Cifar100(batchsize):
    train_data, test_data = PreProcess_Cifar100('/home/haozc/2023-AAAI/datasets', batchsize)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)

    train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=False, num_workers=2, sampler=train_sampler,
                                  pin_memory=True, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=2, sampler=test_sampler,
                                 pin_memory=True, drop_last=False)

    return train_dataloader, test_dataloader, train_sampler, test_sampler


def load_ImageNet_dataset(batchsize, train_dir, test_dir):
    trans_t = transforms.Compose([transforms.RandomResizedCrop(224),
                                  transforms.RandomHorizontalFlip(),
                                  ImageNetPolicy(),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                  ])
    trans = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])

    train_data = datasets.ImageFolder(root=train_dir, transform=trans_t)
    test_data = datasets.ImageFolder(root=test_dir, transform=trans)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)

    train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=False, num_workers=16,
                                  sampler=train_sampler, pin_memory=True, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=4, sampler=test_sampler,
                                 pin_memory=True, drop_last=False)

    return train_dataloader, test_dataloader, train_sampler, test_sampler


def train_one_epoch(model, loss_fn, optimizer, train_dataloader, time_step, epoch, opt_mode, use_TET, local_rank,
                    scaler=None, mixup=None):
    epoch_loss, lenth = 0, 0
    model.train()
    for img, label in train_dataloader:
        img = img.cuda(local_rank, non_blocking=True)
        label = label.cuda(local_rank, non_blocking=True)
        lenth += len(img)

        optimizer.zero_grad()

        if mixup is not None:
            img, label = mixup(img, label)

        if scaler is not None:
            with amp.autocast():
                spikes = model(img)
                if use_TET:
                    loss = torch.stack([loss_fn(spikes[t], label) for t in range(time_step)]).mean(dim=0)
                else:
                    spikes = spikes.mean(dim=0)
                    loss = loss_fn(spikes, label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            spikes = model(img)
            if use_TET:
                loss = torch.stack([loss_fn(spikes[t], label) for t in range(time_step)]).mean(dim=0)
            else:
                spikes = spikes.mean(dim=0)
                loss = loss_fn(spikes, label)
            loss.backward()
            optimizer.step()

        vis_loss = reduce_mean(loss, torch.distributed.get_world_size())
        epoch_loss += vis_loss.item()

    return epoch_loss / lenth


def eval_one_epoch(model, test_dataloader, sim_len):
    tot = torch.zeros(sim_len).cuda()
    model.eval()
    lenth = 0
    with torch.no_grad():
        for img, label in test_dataloader:
            spikes = 0
            img = img.to(torch.device('cuda'), non_blocking=True)
            label = label.to(torch.device('cuda'), non_blocking=True)
            lenth += len(img)
            out = model(img)
            for t in range(sim_len):
                spikes += out[t]
                tot[t] += (label == spikes.max(1)[1]).sum().item()
            # reset_net(model)

    return tot / lenth


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--savedir', type=str, default='/home/user/',
                        help='Directory where the model is saved')
    parser.add_argument('--trainsnn_epochs', type=int, default=320, help='Training Epochs of SNNs')
    parser.add_argument('--net_arch', type=str, default='ms-resnet34', help='Network Architecture')
    parser.add_argument('--batchsize', type=int, default=64, help='Batch size')
    parser.add_argument('--time_step', type=int, default=4, help='Training Time-steps for SNNs')
    parser.add_argument('--lr2', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--direct_inference', action='store_true', default=False)
    parser.add_argument('--train_dir', type=str, default='/home/data/public/ImageNet/train',
                        help='Directory where the ImageNet train dataset is saved')
    parser.add_argument('--test_dir', type=str, default='/home/data/public/ImageNet/val',
                        help='Directory where the ImageNet test dataset is saved')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dev', type=str, default='0')
    parser.add_argument('--opt_mode', type=str, default='sigmoid', help='Optimization mode for hyperparameter')
    parser.add_argument('--use_TET', action='store_true', default=False)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--distributed_init_mode', type=str, default='env://')
    parser.add_argument("--sync_bn", action="store_true", help="Use sync batch norm")
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--amp', action='store_true', help='Use AMP training')
    parser.add_argument('--warm-up', type=str, nargs='+', default=[], help='--warm-up <epochs> <start-factor>')
    parser.add_argument('--mixup', action='store_true', help='Mixup')

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

    log_dir = args.savedir + 'ImageNet-checkpoints'
    identifier = 'ImageNet_' + args.net_arch + '_T' + str(args.time_step) + '_' + args.opt_mode + '_TET_' + str(
        args.use_TET) + '_lr' + str(args.lr2) + '_epoch' + str(args.trainsnn_epochs)
    save_name_suffix = log_dir + '/' + identifier

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = get_logger(os.path.join(log_dir, '%s.log' % (identifier)))

    distributed, rank, world_size, local_rank = init_distributed(args.distributed_init_mode)
    train_dataloader, test_dataloader, train_sampler, test_sampler = load_ImageNet_dataset(args.batchsize,
                                                                                           args.train_dir,
                                                                                           args.test_dir)
    # train_dataloader, test_dataloader, train_sampler, test_sampler = load_Cifar100(args.batchsize)

    if args.net_arch == 'ms-resnet34':
        model = ms_resnet34(num_classes=1000, zero_init_residual=False, T=args.time_step, connect_f='ADD')
    elif args.net_arch == 'spiking-resnet34':
        model = spiking_resnet34(num_classes=1000, zero_init_residual=False, T=args.time_step, connect_f='ADD')
    else:
        error('unable to find model ' + args.net_arch)

    if args.opt_mode == 'linear':
        a1, b1, a2, b2 = 0., 0., 1., 1.
    else:
        a1, b1, a2, b2 = 0., 0., 0., 0.

    model = replace_ReLU_by_Lneuron(model, False, args.time_step, True)

    if local_rank == 0:
        print(model)

    model.cuda()
    if distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    mixup = None
    if args.mixup:
        mixup = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None, prob=1.0,
                      switch_prob=0.5, mode='batch', label_smoothing=0.1, num_classes=1000)

    if args.amp:
        scaler = amp.GradScaler()
    else:
        scaler = None

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr2, momentum=0.9, weight_decay=args.wd, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.trainsnn_epochs)
    loss_fn = nn.CrossEntropyLoss()

    if len(args.warm_up) != 0:
        assert len(args.warm_up) == 2 and scheduler is not None
        scheduler = torch.optim.lr_scheduler.ChainedScheduler([
            torch.optim.lr_scheduler.LinearLR(optimizer=optimizer,
                                              start_factor=float(args.warm_up[1]),
                                              total_iters=int(args.warm_up[0])),
            scheduler, ])

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                      find_unused_parameters=False, broadcast_buffers=False)

    if args.resume:
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        model.module.load_state_dict(checkpoint['model'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        max_acc1 = checkpoint['max_acc1']
        scheduler.load_state_dict(checkpoint['scheduler'])
        print(max_acc1, start_epoch)
    else:
        start_epoch = 0
        max_acc1 = 0

    if args.direct_inference is not True:
        best_acc = max_acc1

        for epoch in range(start_epoch, args.trainsnn_epochs):
            train_sampler.set_epoch(epoch)
            epoch_loss = train_one_epoch(model, loss_fn, optimizer, train_dataloader, args.time_step, epoch,
                                         args.opt_mode, args.use_TET, local_rank, scaler, mixup)
            scheduler.step()

            checkpoint = {
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'max_acc1': max_acc1
            }

            # print(f"finish epoch {epoch}")
            if local_rank == 0:
                acc = eval_one_epoch(model, test_dataloader, args.time_step)
                if best_acc < acc[-1].item():
                    best_acc = acc[-1].item()
                    max_acc1 = acc[-1].item()
                    checkpoint['max_acc1'] = acc[-1].item()
                    torch.save(model.module.state_dict(), save_name_suffix + '_SNN.pth')
                torch.save(checkpoint, save_name_suffix + '_checkpoint.pth')

                logger.info(f"SNNs training Epoch {epoch}: Val_loss: {epoch_loss}")
                logger.info(f"SNNs training Epoch {epoch}: Test Acc: {acc} Best Acc: {best_acc}")

            torch.distributed.barrier()

    else:
        if local_rank == 0:
            print(f'=== Load Pretrained SNNs ===')
            model.load_state_dict(torch.load(args.load_model_name + '.pth'))
            print_message(model, args.opt_mode)
            new_acc = eval_one_epoch(model, test_dataloader, args.time_step)
            print(new_acc)
            t = 1
            while t < args.sim_len:
                print(f'time step {t}, Accuracy = {(new_acc[t - 1]):.4f}')
                t *= 2
            print(f'time step {args.sim_len}, Accuracy = {(new_acc[args.sim_len - 1]):.4f}')

        torch.distributed.barrier()