# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 15:26:36 2020

@author: lps
"""

from torch.utils.data import Dataset, DataLoader
from torchvision .datasets import ImageFolder
import numpy as np
import torch
from detect_align import process_data
from torch.utils.data.dataloader import default_collate
import random


class FaceEmbed(TensorDataset):
    def __init__(self, data_path_list, same_prob=0.8):
        datasets = []
        # embeds = []
        self.N = []
        self.same_prob = same_prob
        for data_path in os.listdir(data_path_list):
            image_list = glob.glob('{:s}/{:s}/*.jpg'.format(data_path_list,data_path))
            datasets.append(image_list)
            self.N.append(len(image_list))

        self.datasets = datasets
        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, item):
        idx = 0
        while item >= self.N[idx]:
            item -= self.N[idx]
            idx += 1
        image_path = self.datasets[idx][item]

        Xs = cv2.imread(image_path)
        Xs = Image.fromarray(Xs)

        if random.random() > self.same_prob:
            image_path = random.choice(self.datasets[random.randint(0, len(self.datasets)-1)])
            Xt = cv2.imread(image_path)
            Xt = Image.fromarray(Xt)
            same_person = 0
        else:
            Xt = Xs.copy()
            same_person = 1
        return self.transforms(Xs), self.transforms(Xt), same_person

    def __len__(self):
        return sum(self.N)

    
def my_collate_fn(batch):
    """
    detect_align data to form a batch
    """
#    print(np.array(batch[0][0]).shape)
    data = [process_data(np.array(item[0])) for item in batch]   # No trans version
#    data = [process_data((item[0]*255).permute(1,2,0).numpy().astype(np.uint8())) for item in batch]
    target = [item[1] for item in batch]
    data = torch.FloatTensor(data/255).permute(0,2,3,1)
    target = torch.LongTensor(target)    
    return [data, target]


class SupplyCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dataset):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dataset = dataset
        

    def supply_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        data = []
        target = []
        bs = len(batch)
        for i in range(bs):
            try:
                data.append([process_data(np.array(batch[i][0]))] ) 
                target.append(batch[i][1])
            except:
                while True:
                    try:
                        new_index = random.randint(0, len(self.dataset)-1)
                        data.append([process_data(np.array(self.dataset[new_index][0]))] ) 
                        target.append(self.dataset[new_index][1])
                        break
                    except:
                        continue
                    
        data = torch.FloatTensor(data).squeeze(1).permute(0,3,1,2)/255        
        target = torch.LongTensor(target) 
#        print(data.shape, target.shape)
        assert len(data) == bs and len(target)==bs
        return [data, target]

    def __call__(self, batch):
        return self.supply_collate(batch)

