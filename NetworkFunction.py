from torch import nn
import torch
from tqdm import tqdm
from utils import *
import numpy as np
import time


def train_ann(train_dataloader, test_dataloader, model, epochs, lr, wd, device, save_name, T):
    model = model.cuda(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    best_acc = 0
    
    for epoch in range(epochs):
        epoch_loss = 0
        lenth = 0
        model.train()
        for img, label in tqdm(train_dataloader):
            img = img.cuda(device)
            label = label.cuda(device)
            
            optimizer.zero_grad()
            out = model(img)
            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            lenth += len(img)
        
        acc = eval_ann(test_dataloader, model, device)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_name)
        
        print(f"ANNs training Epoch {epoch}: Val_loss: {epoch_loss/lenth} Acc: {acc} BestAcc: {best_acc}")
        scheduler.step()  
        
        if epoch == 0 or (epoch+1) % 10 == 0:
            model = replace_QCFS_by_Lneuron(model, False, False, T)
            print_message(model)
            new_acc = eval_snn(test_dataloader, model, sim_len=16, device=device, use_TEBN=False, use_SEW=False, dvs_data=False)
            print(f"Epoch {epoch} SNNs acc. = {new_acc}")
            model = replace_Lneuron_by_QCFS(model, T=T)
            model = model.cuda(device)
          
    return model
    

def train_snn(train_dataloader, test_dataloader, model, epochs, lr, wd, device, save_name, time_step, use_TET, use_TEBN, pruning, tot_flat_width, use_SEW, dvs_data):
    model = model.cuda(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd) #, nesterov=True
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    best_acc = 0
    
    if pruning is True:
        current_times, total_times = 0, epochs*len(train_dataloader)
        print(current_times, total_times)
    
    for epoch in range(epochs):
        epoch_loss = 0
        lenth = 0
        model.train()
        for img, label in tqdm(train_dataloader):
            spikes = []
            img = img.cuda(device)
            label = label.cuda(device)
            lenth += len(img)
            
            if pruning is True:
                current_times += 1
                set_flat_width(model, tot_flat_width, current_times, total_times)
            
            optimizer.zero_grad()
            if dvs_data:
                img = img.transpose(0, 1).flatten(0, 1).contiguous()
                out = model(img)
                out_shape = [time_step, int(out.shape[0]/time_step)]
                out_shape.extend(out.shape[1:])                
                spikes = out.view(out_shape)
            elif use_SEW:
                spikes = model(img)
            elif use_TEBN:
                img = img.unsqueeze(0).repeat(time_step, 1, 1, 1, 1).flatten(0, 1).contiguous()
                out = model(img)
                out_shape = [time_step, int(out.shape[0]/time_step)]
                out_shape.extend(out.shape[1:])                
                spikes = out.view(out_shape)
            else:
                for t in range(time_step):
                    out = model(img)
                    spikes.append(out)
                reset_net(model)
                spikes = torch.stack(spikes)
                           
            if use_TET:
                loss = torch.stack([loss_fn(spikes[t], label) for t in range(time_step)]).mean(dim=0)
            else:
                spikes = spikes.mean(dim=0)
                loss = loss_fn(spikes, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()


        if epoch == 0 or (epoch+1) % 10 == 0:
            print_message(model)
        acc = eval_snn(test_dataloader, model, sim_len=time_step, device=device, use_TEBN=use_TEBN, use_SEW=use_SEW, dvs_data=dvs_data)
        
        if acc[-1].item() > best_acc:
            best_acc = acc[-1].item()
            torch.save(model.state_dict(), save_name)
        
        print(f"SNNs training Epoch {epoch}: Val_loss: {epoch_loss/lenth} Acc: {acc} BestAcc: {best_acc}")
        model = model.cuda(device)
        
        if pruning is True:
            message = [0., 0.]
            print_sparse_rate(model, message)
            print(f"SNNs training Epoch {epoch}: SparseRate: {message[0] / message[1] * 100:.2f}% Acc: {acc}")
        
        scheduler.step()
          
    return model


def train_snn_online(train_dataloader, test_dataloader, model, epochs, lr, wd, device, save_name, time_step, time_slice, use_TEBN, use_TET):
    model = model.cuda(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    best_acc = 0
    for epoch in range(epochs):
        epoch_loss = 0
        lenth = 0
        model.train()
        for img, label in tqdm(train_dataloader):
            img = img.cuda(device)
            label = label.cuda(device)
            lenth += len(img)
            
            for t in range(0, time_step, time_slice):
                optimizer.zero_grad()
                if use_TEBN:
                    if t == 0:
                        img = img.unsqueeze(0).repeat(time_slice, 1, 1, 1, 1).flatten(0, 1).contiguous()
                    out = model(img)
                    out_shape = [time_slice, int(out.shape[0]/time_slice)]
                    out_shape.extend(out.shape[1:])
                    spikes = out.view(out_shape)
                else:
                    spikes = []
                    for t in range(time_slice):
                        out = model(img)
                        spikes.append(out)
                    spikes = torch.stack(spikes)

                if use_TET:
                    loss = torch.stack([loss_fn(spikes[t], label) for t in range(time_slice)]).mean(dim=0)
                else:
                    spikes = spikes.mean(dim=0)
                    loss = loss_fn(spikes, label)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            reset_net(model)
                       

        if epoch == 0 or (epoch+1) % 5 == 0:
            print_message(model)
        acc = eval_snn(test_dataloader, model, sim_len=time_step, device=device, use_TEBN=use_TEBN, use_SEW=False, dvs_data=False)
        
        if acc[-1].item() > best_acc:
            best_acc = acc[-1].item()
            torch.save(model.state_dict(), save_name)
        
        print(f"SNNs training Epoch {epoch}: Val_loss: {epoch_loss/lenth} Acc: {acc} BestAcc: {best_acc}")
        model = model.cuda(device)

        scheduler.step()
          
    return model

    
def eval_ann(test_dataloader, model, device):
    tot = 0
    model.eval()
    model.cuda(device)
    lenth = 0
    with torch.no_grad():
        for img, label in tqdm(test_dataloader):
            img = img.cuda(device)
            label = label.cuda(device)
            out = model(img)
            tot += (label==out.max(1)[1]).sum().item()
            lenth += len(img)
    return tot / lenth


def eval_snn(test_dataloader, model, sim_len, device, use_TEBN, use_SEW, dvs_data):
    tot = torch.zeros(sim_len).cuda(device)
    model = model.cuda(device)
    model.eval()
    lenth = 0
    with torch.no_grad():
        for img, label in tqdm(test_dataloader):
            spikes = 0
            img = img.cuda(device)
            label = label.cuda(device)
            lenth += len(img)
            
            if dvs_data:
                img = img.transpose(0, 1).flatten(0, 1).contiguous()
                out = model(img)
                out_shape = [sim_len, int(out.shape[0]/sim_len)]
                out_shape.extend(out.shape[1:])                
                out = out.view(out_shape)
                for t in range(sim_len):
                    spikes += out[t]
                    tot[t] += (label==spikes.max(1)[1]).sum().item()
                reset_net(model)
            elif use_SEW:
                out = model(img)
                for t in range(sim_len):
                    spikes += out[t]
                    tot[t] += (label==spikes.max(1)[1]).sum().item()
                reset_net(model)
            elif use_TEBN:
                img = img.unsqueeze(0).repeat(sim_len, 1, 1, 1, 1).flatten(0, 1).contiguous()
                out = model(img)
                out_shape = [sim_len, int(out.shape[0]/sim_len)]
                out_shape.extend(out.shape[1:])                
                out = out.view(out_shape)
                for t in range(sim_len):
                    spikes += out[t]
                    tot[t] += (label==spikes.max(1)[1]).sum().item()
                reset_net(model)
            else:
                for t in range(sim_len):
                    out = model(img)
                    spikes += out
                    tot[t] += (label==spikes.max(1)[1]).sum().item()
                reset_net(model)
      
    return tot/lenth