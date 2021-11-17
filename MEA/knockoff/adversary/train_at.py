#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import argparse
import json
import os
import os.path as osp
import pickle
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, TensorDataset
from torch import optim
import torch.nn.functional as F
from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS, default_loader
from tqdm import tqdm
import torch.nn as nn 
import knockoff.config as cfg
import knockoff.utils.model as model_utils
from knockoff import datasets
#import knockoff.models.zoo as zoo
from torch.utils.data import DataLoader 
import time 
from knockoff.victim.blackbox import Blackbox

torch.multiprocessing.set_sharing_strategy('file_system')

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


class TransferSetImagePaths(ImageFolder):
    """TransferSet Dataset, for when images are stored as *paths*"""

    def __init__(self, samples, transform=None, target_transform=None):
        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.transform = transform
        self.target_transform = target_transform


class TransferSetImages(Dataset):
    def __init__(self, samples, transform=None, target_transform=None):
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform

        self.data = [self.samples[i][0] for i in range(len(self.samples))]
        self.targets = [self.samples[i][1] for i in range(len(self.samples))]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


def samples_to_transferset(samples, budget=None, transform=None, target_transform=None):
    # Images are either stored as paths, or numpy arrays
    sample_x = samples[0][0]
    assert budget <= len(samples), 'Required {} samples > Found {} samples'.format(budget, len(samples))

    if isinstance(sample_x, str):
        return TransferSetImagePaths(samples[:budget], transform=transform, target_transform=target_transform)
    elif isinstance(sample_x, np.ndarray):
        return TransferSetImages(samples[:budget], transform=transform, target_transform=target_transform)
    else:
        raise ValueError('type(x_i) ({}) not recognized. Supported types = (str, np.ndarray)'.format(type(sample_x)))

def initial_transferset(samples, budget=None, transform=None, target_transform=None):
    # Images are either stored as paths, or numpy arrays
    sample_x = samples[0][0]
    assert budget <= len(samples), 'Required {} samples > Found {} samples'.format(budget, len(samples))

    transfer_idx = []
    rest_idx = []

    for i in range(len(samples)):
        if i < budget:
            transfer_idx.append(i)
        else:
            rest_idx.append(i)
    
    if isinstance(sample_x, str):
        return TransferSetImagePaths(samples[:budget], transform=transform, target_transform=target_transform), transfer_idx, rest_idx
    elif isinstance(sample_x, np.ndarray):
        return TransferSetImages(samples[:budget], transform=transform, target_transform=target_transform), transfer_idx, rest_idx
    else:
        raise ValueError('type(x_i) ({}) not recognized. Supported types = (str, np.ndarray)'.format(type(sample_x)))


def get_idx_to_transferset(samples, idx=None, transform=None, target_transform=None):
    # Images are either stored as paths, or numpy arrays
    sample_x = samples[0][0]

    if isinstance(sample_x, str):
        return TransferSetImagePaths([samples[i] for i in idx], transform=transform, target_transform=target_transform)
    elif isinstance(sample_x, np.ndarray):
        return TransferSetImages([samples[i] for i in idx], transform=transform, target_transform=target_transform)
    else:
        raise ValueError('type(x_i) ({}) not recognized. Supported types = (str, np.ndarray)'.format(type(sample_x)))

    
def uncertainty_transferset(model, samples, cur_idx, rest_idx, device, budget=None, transform=None, target_transform=None):
    # Uncertainty sampling on unused data
    
    restset = get_idx_to_transferset(samples,rest_idx,transform)
    rest_loader = DataLoader(restset, batch_size=32, num_workers=16, pin_memory=False,shuffle=False)
    print("=> Sampling")
    cnt = 0
    data_xy = []
    with torch.no_grad():
        for x,y_vic in tqdm(rest_loader,desc="Sampling"):
            x = x.to(device)

            y_sur = model(x)
            y_sur = F.softmax(y_sur,dim=-1)
            y_sur, _ = torch.max(y_sur,dim=1)
            
            y_sur = y_sur.cpu()
            
            for idx, i in enumerate(x):
                
                data_xy.append((cnt,y_sur[idx]))
                cnt += 1

    print("=> Sorting")
    print(len(data_xy))
    data_xy.sort(key = lambda x:x[1])
    
    for idx, y_sur in tqdm(data_xy[:budget],desc="Transfer idx"):
        cur_idx.append(rest_idx[idx])

    rest_idx = []
    for i in tqdm(range(len(samples)),desc="Rest idx"):
        if i not in cur_idx:
            rest_idx.append(i)
    
    
    return cur_idx, rest_idx
    
def get_optimizer(parameters, optimizer_type, lr=0.01, momentum=0.5, wd=0): #, **kwargs):
    assert optimizer_type in ['sgd', 'sgdm', 'adam', 'adagrad']
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(parameters, lr)
    elif optimizer_type == 'sgdm':
        if wd != 0: 
            optimizer = optim.SGD(parameters, lr, momentum=momentum, weight_decay=wd)
        else: 
            optimizer = optim.SGD(parameters, lr, momentum=momentum)


    elif optimizer_type == 'adagrad':
        optimizer = optim.Adagrad(parameters)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(parameters)
    else:
        raise ValueError('Unrecognized optimizer type')
    return optimizer


def main():
    parser = argparse.ArgumentParser(description='Train a model')
    # Required arguments
    parser.add_argument('--model_dir', metavar='DIR', type=str, help='Directory containing transferset.pickle')
    parser.add_argument('--model_arch', metavar='MODEL_ARCH', type=str, help='Model name')
    parser.add_argument('--testdataset', metavar='DS_NAME', type=str, help='Name of test')
    parser.add_argument('--origdataset', metavar='DS_NAME', type=str, help='Name of orig test')
    parser.add_argument('--budgets', metavar='B', type=str,
                        help='Comma separated values of budgets. Knockoffs will be trained for each budget.')
    # Optional arguments
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('-r', '--rounds', type=int, default=10,
                        help='actitheif number of rounds')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--wd', type=float, default=0,
                        help='SGD wd (default: None)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--lr-step', type=int, default=60, metavar='N',
                        help='Step sizes for LR')
    parser.add_argument('--lr-gamma', type=float, default=0.1, metavar='N',
                        help='LR Decay Rate')
    parser.add_argument('-w', '--num_workers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    parser.add_argument('--pretrained', type=str, help='Use pretrained network', default=None)
    parser.add_argument('--weighted-loss', action='store_true', help='Use a weighted loss', default=False)
    # Attacker's defense
    parser.add_argument('--argmaxed', action='store_true', help='Only consider argmax labels', default=False)
    parser.add_argument('--optimizer_choice', type=str, help='Optimizer', default='sgdm', choices=('sgd', 'sgdm', 'adam', 'adagrad'))
    
    parser.add_argument('--victim_model', type=str, help='Victim model architecture')
    parser.add_argument('--victim_model_dir', type=str, help='victim model dir')
   

    args = parser.parse_args()
    params = vars(args)

    torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model_dir = params['model_dir']

    # ----------- Set up transferset
    transferset_path = osp.join(model_dir, 'transferset.pickle')
    print("transferset path: ", transferset_path)
    with open(transferset_path, 'rb') as rf:
        transferset_samples = pickle.load(rf)
    num_classes = transferset_samples[0][1].size(0)
    print('=> found transfer set with {} samples, {} classes'.format(len(transferset_samples), num_classes))

    # ----------- Clean up transfer (if necessary)
    if params['argmaxed']:
        new_transferset_samples = []
        print('=> Using argmax labels (instead of posterior probabilities)')
        for i in range(len(transferset_samples)):
            x_i, y_i = transferset_samples[i]
            argmax_k = y_i.argmax()
            y_i_1hot = torch.zeros_like(y_i)
            y_i_1hot[argmax_k] = 1.
            new_transferset_samples.append((x_i, y_i_1hot))
        transferset_samples = new_transferset_samples


    # ----------- Set up vic testset
    dataset_name = params['testdataset']
    valid_datasets = datasets.__dict__.keys()
    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    transform = datasets.modelfamily_to_transforms[modelfamily]['test']

    if dataset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    dataset = datasets.__dict__[dataset_name]
    testset = dataset(train=False, transform=transform)
    dataiter = iter(DataLoader(testset))
    test_img, test_label = dataiter.next()
    print("test image shape: ", test_img.shape)
    if len(testset.classes) != num_classes:
        raise ValueError('# Transfer classes ({}) != # Testset classes ({})'.format(num_classes, len(testset.classes)))

       
    # ----------- Set up orig testset
    dataset_name = params['origdataset']
    valid_datasets = datasets.__dict__.keys()
    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    #if dataset_name not in valid_datasets:
        #raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    dataset = datasets.__dict__[dataset_name]
    origtest_set = dataset(train=False, transform=transform)
    dataiter = iter(DataLoader(origtest_set))
    origtest_img, origtest_label = dataiter.next()
    print("original test image shape: ", origtest_img.shape)



    # ----------- Initialize blackbox
    blackbox_dir = params['victim_model_dir']
    num_class = datasets.num_class[dataset_name]
    victim_model_name = params['victim_model']
    blackbox = Blackbox.from_modeldir(blackbox_dir, model_name=victim_model_name, num_class=num_class, device=device)

    ###test victim 
    test_dataloader = DataLoader(testset, batch_size=32, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    _, victim_acc = model_utils.test_step(blackbox, test_dataloader, criterion=criterion, device=device)
    print("Victim accuracy:", victim_acc)
 



    if len(origtest_set.classes) != num_classes:
        raise ValueError('# Transfer classes ({}) != # Testset classes ({})'.format(num_classes, len(origtest_set.classes)))

    # ----------- Set up model
    model_name = params['model_arch']
    pretrained = params['pretrained']

    pretrained_bool = False 
    if pretrained == 'imagenet':
        pretrained_bool = True
    print("pretrain: ", pretrained_bool)
    model = model_utils.get_net(model_name, n_output_classes=num_classes, pretrained=pretrained_bool)
    model = model.to(device)
    model_dimension = datasets.sir_model_dimension[model_name]
    print("Surrogate model")
    print(model)
    # ----------- Train
    budgets = [int(b) for b in params['budgets'].split(',')]
    rounds = params['rounds']

    for b in budgets:
        np.random.seed(cfg.DEFAULT_SEED)
        torch.manual_seed(cfg.DEFAULT_SEED)
        torch.cuda.manual_seed(cfg.DEFAULT_SEED)
        print(transform)
        transferset, transfer_idx, rest_idx = initial_transferset(transferset_samples, budget=int(b/rounds), transform=transform)
        


        for i in range(rounds):


            print('=> Training at budget = {} / round = {}'.format(len(transferset),i))
            pretrained_bool = False 
            if pretrained == 'imagenet':
                pretrained_bool = True
            print("pretrain: ", pretrained_bool)
            model = model_utils.get_net(model_name, n_output_classes=num_classes, pretrained=pretrained_bool)
            model = model.to(device)
            model_dimension = datasets.sir_model_dimension[model_name]
            
            optimizer = get_optimizer(model.parameters(), params['optimizer_choice'], lr=params['lr'], wd=params['wd'], momentum=params['momentum'])#, **params)
            print(params)
        
            checkpoint_suffix = '.{}'.format(b)
            criterion_train = model_utils.soft_cross_entropy
            sec = time.perf_counter()
            model_utils.sur_train_model_at(model=model, 
                                        model_dimension=model_dimension, 
                                        trainset=transferset, 
                                        out_path=model_dir, 
                                        testset=testset, 
                                        origtest_set=origtest_set,
                                        criterion_train=criterion_train,
                                        checkpoint_suffix=checkpoint_suffix, 
                                        device=device, 
                                        victim_acc=victim_acc, 
                                        optimizer=optimizer, **params)
            retrain_sec = time.perf_counter()
            print("##############retrain time###########:", retrain_sec - sec)

            transfer_idx, rest_idx = uncertainty_transferset(model,transferset_samples,transfer_idx,rest_idx, device,budget=int(b/rounds), transform=transform)

            transferset = get_idx_to_transferset(transferset_samples,transfer_idx,transform=transform)

            
    # Store arguments
    params['created_on'] = str(datetime.now())
    params_out_path = osp.join(model_dir, 'params_train.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)


if __name__ == '__main__':
    main()
