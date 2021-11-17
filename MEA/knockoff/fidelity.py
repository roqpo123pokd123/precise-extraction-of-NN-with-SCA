from torch.utils.data import DataLoader

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import argparse
import os.path as osp
import os
import pickle
import json
from datetime import datetime

import numpy as np
import argparse

from tqdm import tqdm
from PIL import Image
import  torchvision.transforms as transforms
from skimage.metrics import structural_similarity as compare_ssim
from numpy import linalg as LA
import knockoff.utils.model as model_utils
from knockoff import datasets 

global main_path
main_path = '/home/hangjung/'



def load_victim_model(victim_model_arch, victim_dataset):
    num_class = datasets.num_class[victim_dataset]
    model = model_utils.get_net(model_name=victim_model_arch, n_output_classes=num_class, pretrained=False)
    model = model.to(device)
    victim_model_dir = main_path + "knockoffnets/models/victim/" +victim_dataset+"/"+victim_model_arch
    checkpoint_path = osp.join(victim_model_dir, 'checkpoint.pth.tar')
    print("=> loading checkpoint '{}'".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    epoch = checkpoint['epoch']
    best_test_acc = checkpoint['best_acc']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint (epoch {}, acc={:.2f})".format(epoch, best_test_acc))

    return model

def load_sur_model(victim_model_arch, sur_model_arch, victim_dataset, query_dataset):
    num_class = datasets.num_class[victim_dataset]
    model = model_utils.get_net(model_name=sur_model_arch, n_output_classes=num_class, pretrained=False)
    model = model.to(device)
    victim_model_dir = main_path + "/knockoffnets/models/adversary/" +victim_dataset+"/"+victim_model_arch+"/"+query_dataset
    
    modelfamily = datasets.dataset_to_modelfamily[query_dataset] #if params['modelfamily'] is None else params['modelfamily']
    transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    sur_data = datasets.__dict__[query_dataset](train=True, transform=transform) 
    n_query = len(sur_data)
    checkpoint_path = osp.join(victim_model_dir, 'checkpoint.'+str(n_query)+'.pth.tar')

    print("=> loading checkpoint '{}'".format(checkpoint_path))
    try:
        checkpoint = torch.load(checkpoint_path)
    except:
        return None
    epoch = checkpoint['epoch']
    best_test_acc = checkpoint['best_acc']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint (epoch {}, acc={:.2f})".format(epoch, best_test_acc))

    return model


def load_dataloader(dataset):
    
    modelfamily = datasets.dataset_to_modelfamily[dataset] #if params['modelfamily'] is None else params['modelfamily']
    transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    data = datasets.__dict__[dataset](train=False, transform=transform) #testing
    dataloader = DataLoader(data, batch_size=32, shuffle=False)

    return dataloader





victim_data_list = ["indoor67","caltech256","cubs200"]
victim_model_list = ['64']
sur_model_list = ['32']
device = "cuda:0"
query = "b60k"
for victim_data in victim_data_list: 
    for victim_dim in victim_model_list:
        victim_model_arch = "res50_"+victim_dim
        victim_dataset = victim_data + "_" + victim_dim

        
        print()
        print()
        print("$$$$$$$$$$$Victim model: ", victim_model_arch, victim_dataset) 
        
        victim_model = load_victim_model(victim_model_arch, victim_dataset)
        victim_model = victim_model.to(device)
        test_dataloader = load_dataloader(victim_dataset)
 
        for sur_dim in sur_model_list:    
            sur_model = load_sur_model(victim_model_arch, "res50_"+sur_dim, victim_dataset, query_dataset = "open_"+str(sur_dim)+'_im60k')
            if sur_model == None:
                print("Cannot find ",victim_data,victim_dim,sur_dim)
                continue
            sur_model = sur_model.to(device)
            sur_dataset = victim_data + "_" + sur_dim
            sur_dataloader = load_dataloader(sur_dataset)
            

            _ = model_utils.sur_fid_test_step_case2(victim_model, sur_model, int(sur_dim), test_dataloader, sur_dataloader, device)


            print()
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        print()
        print()
