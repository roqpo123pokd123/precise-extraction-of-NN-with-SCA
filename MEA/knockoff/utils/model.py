#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import argparse
import os.path as osp
import os
import time
from datetime import datetime
from collections import defaultdict as dd

import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as torch_models

import knockoff.config as cfg
import knockoff.utils.utils as knockoff_utils

from PIL import Image
import torchvision

from .resnet_128 import ResNet50_128
from .resnet_64 import ResNet50_64
from .resnet_32 import ResNet50_32
from .wresnet import WideResNet
from functools import partial

from tqdm import tqdm

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"

model_dict = {
    "res50_128": ResNet50_128,
    "res50_64": ResNet50_64,
    "res50_32": ResNet50_32,

    "wres28-10": partial(WideResNet, 28, widen_factor=10),
    "wres28-5": partial(WideResNet, 28, widen_factor=5),
    "wres28-1": partial(WideResNet, 28, widen_factor=1),
    
    
    
}

def get_net(model_name, n_output_classes, pretrained, device=None, **kwargs):
    print('=> loading model {} with arguments: {}'.format(model_name, kwargs))
    valid_models = [x for x in torch_models.__dict__.keys() if not x.startswith('__')]
    print(model_name)
    if model_name not in ["res50_32","res50_64","res50_128","res50_224", "wres28-10","wres28-5","wres28-1"]:
        raise ValueError('Model not found. Valid arguments = {}...'.format(valid_models))
    #model = torch_models.__dict__[model_name](**kwargs)
    # Edit last FC layer to include n_output_classes

       
    if model_name == 'res50_224':
        print('pretrained: ', pretrained)
        model = torch_models.__dict__["resnet50"](pretrained=pretrained, **kwargs)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_output_classes)

    elif model_name == 'wres50-2':
        print('pretrained: ', pretrained)
        model = torch_models.__dict__["wide_resnet50_2"](pretrained=pretrained, **kwargs)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_output_classes)

    else:
        model = model_dict[model_name](num_classes=n_output_classes)
        
        print('pretrained: ', pretrained)
        if pretrained == True:
            model = model_dict[model_name](num_classes=1000)
            if model_name == 'res50_128':
                pretrained_model_path = "./models/pretrained/128/res50_128_pretrained.pt"
            elif model_name == 'res50_224':
                pretrained_model_path = "./models/pretrained/64/res50_224_pretrained.pt"
            elif model_name == 'res50_64':
                pretrained_model_path = "./models/pretrained/64/res50_64_pretrained.pt"
            elif model_name == 'res50_32':
                pretrained_model_path = "./models/pretrained/32/res50_32_pretrained.pt"


            print("load pretrained weights from ", pretrained_model_path)
            if device  == None: 
                model.load_state_dict(torch.load(pretrained_model_path))
            else: 
                model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
        
        if model_name in ["wres28-10","wres28-5","wres28-1"]:
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, n_output_classes)
        else:
            num_ftrs = model.linear.in_features
            model.linear = nn.Linear(num_ftrs, n_output_classes)

   
    if model == None:
        assert("no model")
        
    return model


def soft_cross_entropy(pred, soft_targets, weights=None):
    if weights is not None:
        return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1) * weights, 1))
    else:
        return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1), 1))


def train_step(model, train_loader, criterion, optimizer, epoch, device, log_interval=10, writer=None):
    model.train()
    train_loss = 0.
    correct = 0
    total = 0
    train_loss_batch = 0
    epoch_size = len(train_loader.dataset)
    t_start = time.time()

    for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if writer is not None:
            pass

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        if len(targets.size()) == 2:
            # Labels could be a posterior probability distribution. Use argmax as a proxy.
            target_probs, target_labels = targets.max(1)
        else:
            target_labels = targets
        correct += predicted.eq(target_labels).sum().item()

        prog = total / epoch_size
        exact_epoch = epoch + prog - 1
        acc = 100. * correct / total
        train_loss_batch = train_loss / total
        
        if writer is not None:
            writer.add_scalar('Loss/train', loss.item(), exact_epoch)
            writer.add_scalar('Accuracy/train', acc, exact_epoch)

    t_end = time.time()
    t_epoch = int(t_end - t_start)
    acc = 100. * correct / total

    return train_loss_batch, acc

def sur_test_step(model, model_dimension, test_loader, criterion, device, epoch=0., silent=False, writer=None):
    model.eval()
    test_loss = 0.
    correct = 0
    total = 0
    t_start = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            if inputs.shape[3] != model_dimension:
                inputs = torchvision.transforms.functional.resize(inputs, model_dimension, interpolation=Image.BILINEAR)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            nclasses = outputs.size(1)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    t_end = time.time()
    t_epoch = int(t_end - t_start)

    acc = 100. * correct / total
    test_loss /= total
    """    
    if not silent:
        print('[Test]  Epoch: {}\tLoss: {:.6f}\tAcc: {:.1f}% ({}/{})'.format(epoch, test_loss, acc,
                                                                             correct, total))
    """
    if writer is not None:
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', acc, epoch)

    return test_loss, acc


def test_step(model, test_loader, criterion, device, epoch=0., silent=False, writer=None):
    model.eval()
    test_loss = 0.
    correct = 0
    total = 0
    t_start = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            nclasses = outputs.size(1)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    t_end = time.time()
    t_epoch = int(t_end - t_start)

    acc = 100. * correct / total
    test_loss /= total
    
    if writer is not None:
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', acc, epoch)

    return test_loss, acc
                                                                                                                                                                                    
def sur_fid_test_step_case2(victim_model, sur_model, sur_dim, test_loader, sur_loader, device, silent=False, writer=None):
    # Compute fidelity for each test case
    
    victim_model = victim_model
    sur_model = sur_model
    victim_model.eval()
    sur_model.eval()

    correct_vic = 0
    correct_orig = 0
    total = 0
    t_start = time.time()
    
    with torch.no_grad():
        for (inputs, targets) ,(sur_inputs,sur_targets)in tqdm(zip(test_loader,sur_loader), total=len(test_loader), desc=str(sur_dim)):
            inputs, targets = inputs.to(device), targets.to(device)
            sur_inputs, sur_targets = sur_inputs.to(device), sur_targets.to(device)
            v_pred = victim_model(inputs)
            _, v_pred_class = v_pred.max(1)
                                                                                        
            if inputs.shape[3] != sur_dim:
                inputs = torchvision.transforms.functional.resize(inputs, sur_dim, interpolation=Image.BILINEAR)
            sur_pred_vic = sur_model(inputs)
            sur_pred_orig = sur_model(sur_inputs)

            _, sur_pred_class_vic = sur_pred_vic.max(1)
            _, sur_pred_class_orig = sur_pred_orig.max(1)
            correct_vic += sur_pred_class_vic.eq(v_pred_class).sum().item()
            correct_orig += sur_pred_class_orig.eq(v_pred_class).sum().item()
    t_end = time.time()
    fid1 = correct_vic / len(test_loader.dataset)
    fid2 = correct_orig / len(test_loader.dataset)


    print("$$$$$$$$$$Fidelity:  ",fid1, fid2)
    if writer is not None:
        writer.add_scalar('Fid/test', fid1)

    return fid1

def train_model(model, trainset, out_path, batch_size=64, criterion_train=None, criterion_test=None, testset=None,origtest_set=None,
                device=None, num_workers=10, lr=0.1, momentum=0.5, lr_step=30, lr_gamma=0.1, resume=None,
                epochs=100, log_interval=100, weighted_loss=False, checkpoint_suffix='', optimizer=None, scheduler=None,
                writer=None, **kwargs):
    if device is None:
        device = torch.device('cuda')
    if not osp.exists(out_path):
        knockoff_utils.create_dir(out_path)
    run_id = str(datetime.now())

    # Data loaders
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    if testset is not None:
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        test_loader = None

    if weighted_loss:
        if not isinstance(trainset.samples[0][1], int):
            print('Labels in trainset is of type: {}. Expected: {}.'.format(type(trainset.samples[0][1]), int))

        class_to_count = dd(int)
        for _, y in trainset.samples:
            class_to_count[y] += 1
        class_sample_count = [class_to_count[c] for c, cname in enumerate(trainset.classes)]
        print('=> counts per class: ', class_sample_count)
        weight = np.min(class_sample_count) / torch.Tensor(class_sample_count)
        weight = weight.to(device)
        print('=> using weights: ', weight)
    else:
        weight = None

    # Optimizer
    if criterion_train is None:
        criterion_train = nn.CrossEntropyLoss(reduction='mean', weight=weight)
    if criterion_test is None:
        criterion_test = nn.CrossEntropyLoss(reduction='mean', weight=weight)
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)
    if scheduler is None:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
    start_epoch = 1
    best_train_acc, train_acc = -1., -1.
    best_test_acc, best_origtest_acc, origtest_acc, test_acc, test_loss = -1., -1., -1., -1., -1.

    # Resume if required
    if resume is not None:
        model_path = resume
        if osp.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            start_epoch = checkpoint['epoch']
            best_test_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

    # Initialize logging
    log_path = osp.join(out_path, 'train{}.log.tsv'.format(checkpoint_suffix))
    if not osp.exists(log_path):
        with open(log_path, 'w') as wf:
            columns = ['run_id', 'epoch', 'split', 'loss', 'accuracy', 'best_accuracy']
            wf.write('\t'.join(columns) + '\n')

    model_out_path = osp.join(out_path, 'checkpoint{}.pth.tar'.format(checkpoint_suffix))
    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_acc = train_step(model, train_loader, criterion_train, optimizer, epoch, device,
                                           log_interval=log_interval)
        scheduler.step(epoch)
        
        best_train_acc = max(best_train_acc, train_acc)



        if test_loader is not None:
            test_loss, test_acc = test_step(model, test_loader, criterion_test, device, epoch=epoch)
            best_test_acc = max(best_test_acc, test_acc)

            print('[Train] Epoch: {} Loss: {:.6f} Accuracy: {:.3f} \t[TEST] Loss: {:.6f} Accuracy: {:.3f}'.format(epoch, train_loss, train_acc, test_loss, test_acc))



        # Checkpoint
        #if test_acc >= best_test_acc:
        state = {
            'epoch': epoch,
            'arch': model.__class__,
            'state_dict': model.state_dict(),
            'best_acc': test_acc,
            'optimizer': optimizer.state_dict(),
            'created_on': str(datetime.now()),
        }
        torch.save(state, model_out_path)

        # Log
        with open(log_path, 'a') as af:
            train_cols = [run_id, epoch, 'train', train_loss, train_acc, best_train_acc]
            af.write('\t'.join([str(c) for c in train_cols]) + '\n')
            test_cols = [run_id, epoch, 'test', test_loss, test_acc, best_test_acc]
            af.write('\t'.join([str(c) for c in test_cols]) + '\n')

    return model




def sur_train_model(model, model_dimension, trainset, out_path, victim_acc, batch_size=64, criterion_train=None, criterion_test=None, testset=None,origtest_set=None,
                device=None, num_workers=10, lr=0.1, momentum=0.5, lr_step=30, lr_gamma=0.1, resume=None,
                epochs=100, log_interval=100, weighted_loss=False, checkpoint_suffix='', optimizer=None, scheduler=None,
                writer=None, **kwargs):
    if device is None:
        device = torch.device('cuda')
    if not osp.exists(out_path):
        knockoff_utils.create_dir(out_path)
    run_id = str(datetime.now())

    # Data loaders
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    if testset is not None:
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        test_loader = None
    
    if origtest_set is not None:
        origtest_loader = DataLoader(origtest_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        origtest_loader = None


    if weighted_loss:
        if not isinstance(trainset.samples[0][1], int):
            print('Labels in trainset is of type: {}. Expected: {}.'.format(type(trainset.samples[0][1]), int))

        class_to_count = dd(int)
        for _, y in trainset.samples:
            class_to_count[y] += 1
        class_sample_count = [class_to_count[c] for c, cname in enumerate(trainset.classes)]
        print('=> counts per class: ', class_sample_count)
        weight = np.min(class_sample_count) / torch.Tensor(class_sample_count)
        weight = weight.to(device)
        print('=> using weights: ', weight)
    else:
        weight = None

    # Optimizer
    if criterion_train is None:
        criterion_train = nn.CrossEntropyLoss(reduction='mean', weight=weight)
    if criterion_test is None:
        criterion_test = nn.CrossEntropyLoss(reduction='mean', weight=weight)
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)
    if scheduler is None:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
    start_epoch = 1
    best_train_acc, train_acc = -1., -1.
    #best_test_acc, test_acc, test_loss = -1., -1., -1.
    best_test_acc, best_origtest_acc, origtest_acc, test_acc, test_loss = -1., -1., -1., -1., -1.


    # Resume if required
    if resume is not None:
        model_path = resume
        if osp.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            start_epoch = checkpoint['epoch']
            best_test_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

    # Initialize logging
    log_path = osp.join(out_path, 'train{}.log.tsv'.format(checkpoint_suffix))
    if not osp.exists(log_path):
        with open(log_path, 'w') as wf:
            columns = ['run_id', 'epoch', 'split', 'loss', 'accuracy', 'best_accuracy']
            wf.write('\t'.join(columns) + '\n')

    model_out_path = osp.join(out_path, 'checkpoint{}.pth.tar'.format(checkpoint_suffix))
    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_acc = train_step(model, train_loader, criterion_train, optimizer, epoch, device,
                                           log_interval=log_interval)
        scheduler.step(epoch)
        besttrain_acc = max(best_train_acc, train_acc)
        

        if epoch % 5 == 0:  
            if test_loader is not None:
                #print("===============Victim test")
                test_loss, test_acc = sur_test_step(model, model_dimension, test_loader, criterion_test, device, epoch=epoch)
                best_test_acc = max(best_test_acc, test_acc)

            if origtest_loader is not None:
                #print("==============Original test")
                origtest_loss, origtest_acc = sur_test_step(model, model_dimension, origtest_loader, criterion_test, device, epoch=epoch)
                best_origtest_acc = max(best_origtest_acc, origtest_acc)


       
                print('[Train] Epoch: {} Loss: {:.6f} Accuracy: {:.3f} \t[Test] Victim Acc: {:.3f} ({:.3f}x) Original Acc: {:.3f}% ({:.3f}x)'.format(epoch, train_loss, train_acc, test_acc, test_acc / victim_acc, origtest_acc, origtest_acc / victim_acc))
                print("[Best] vic acc : {:.2f} orig acc : {:.2f}".format(best_test_acc,best_origtest_acc))
                
                


        # Checkpoint
        #if test_acc >= best_test_acc: 
        state = {
            'epoch': epoch,
            'arch': model.__class__,
            'state_dict': model.state_dict(),
            'best_acc': test_acc,
            'optimizer': optimizer.state_dict(),
            'created_on': str(datetime.now()),
        }
        torch.save(state, model_out_path)

        # Log
        with open(log_path, 'a') as af:
            train_cols = [run_id, epoch, 'train', train_loss, train_acc, best_train_acc]
            af.write('\t'.join([str(c) for c in train_cols]) + '\n')
            test_cols = [run_id, epoch, 'test', test_loss, test_acc, best_test_acc]
            af.write('\t'.join([str(c) for c in test_cols]) + '\n')
    
    return model





def sur_train_model_at(model, model_dimension, trainset, out_path, victim_acc, batch_size=64, criterion_train=None, criterion_test=None, testset=None,origtest_set=None,
                device=None, num_workers=10, lr=0.1, momentum=0.5, lr_step=30, lr_gamma=0.1, resume=None,
                epochs=100, log_interval=100, weighted_loss=False, checkpoint_suffix='', optimizer=None, scheduler=None,
                writer=None, **kwargs):
    if device is None:
        device = torch.device('cuda')
    if not osp.exists(out_path):
        knockoff_utils.create_dir(out_path)
    run_id = str(datetime.now())

    # Data loaders
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    if testset is not None:
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        test_loader = None
    
    if origtest_set is not None:
        origtest_loader = DataLoader(origtest_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        origtest_loader = None


    if weighted_loss:
        if not isinstance(trainset.samples[0][1], int):
            print('Labels in trainset is of type: {}. Expected: {}.'.format(type(trainset.samples[0][1]), int))

        class_to_count = dd(int)
        for _, y in trainset.samples:
            class_to_count[y] += 1
        class_sample_count = [class_to_count[c] for c, cname in enumerate(trainset.classes)]
        print('=> counts per class: ', class_sample_count)
        weight = np.min(class_sample_count) / torch.Tensor(class_sample_count)
        weight = weight.to(device)
        print('=> using weights: ', weight)
    else:
        weight = None

    # Optimizer
    if criterion_train is None:
        criterion_train = nn.CrossEntropyLoss(reduction='mean', weight=weight)
    if criterion_test is None:
        criterion_test = nn.CrossEntropyLoss(reduction='mean', weight=weight)
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)
    if scheduler is None:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
    start_epoch = 1
    best_train_acc, train_acc = -1., -1.
    #best_test_acc, test_acc, test_loss = -1., -1., -1.
    best_test_acc, best_origtest_acc, origtest_acc, test_acc, test_loss = -1., -1., -1., -1., -1.


    # Resume if required
    if resume is not None:
        model_path = resume
        if osp.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            start_epoch = checkpoint['epoch']
            best_test_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

    # Initialize logging
    log_path = osp.join(out_path, 'train{}.at.log.tsv'.format(checkpoint_suffix))
    if not osp.exists(log_path):
        with open(log_path, 'w') as wf:
            columns = ['run_id', 'epoch', 'split', 'loss', 'accuracy', 'best_accuracy']
            wf.write('\t'.join(columns) + '\n')

    model_out_path = osp.join(out_path, 'checkpoint{}.at.pth.tar'.format(checkpoint_suffix))
    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_acc = train_step(model, train_loader, criterion_train, optimizer, epoch, device,
                                           log_interval=log_interval)
        scheduler.step(epoch)
        besttrain_acc = max(best_train_acc, train_acc)
        

        if epoch % 5 == 0:  
            if test_loader is not None:
                #print("===============Victim test")
                test_loss, test_acc = sur_test_step(model, model_dimension, test_loader, criterion_test, device, epoch=epoch)
                best_test_acc = max(best_test_acc, test_acc)

            if origtest_loader is not None:
                #print("==============Original test")
                origtest_loss, origtest_acc = sur_test_step(model, model_dimension, origtest_loader, criterion_test, device, epoch=epoch)
                best_origtest_acc = max(best_origtest_acc, origtest_acc)


       
                print('[Train] Epoch: {} Loss: {:.6f} Accuracy: {:.3f} \t[Test] Victim Acc: {:.3f} ({:.3f}x) Original Acc: {:.3f}% ({:.3f}x)'.format(epoch, train_loss, train_acc, test_acc, test_acc / victim_acc, origtest_acc, origtest_acc / victim_acc))
                print("[Best] vic acc : {:.2f} orig acc : {:.2f}".format(best_test_acc,best_origtest_acc))
                
                


        # Checkpoint
        #if test_acc >= best_test_acc: 
        state = {
            'epoch': epoch,
            'arch': model.__class__,
            'state_dict': model.state_dict(),
            'best_acc': test_acc,
            'optimizer': optimizer.state_dict(),
            'created_on': str(datetime.now()),
        }
        torch.save(state, model_out_path)

        # Log
        with open(log_path, 'a') as af:
            train_cols = [run_id, epoch, 'train', train_loss, train_acc, best_train_acc]
            af.write('\t'.join([str(c) for c in train_cols]) + '\n')
            test_cols = [run_id, epoch, 'test', test_loss, test_acc, best_test_acc]
            af.write('\t'.join([str(c) for c in test_cols]) + '\n')
    
    return model
