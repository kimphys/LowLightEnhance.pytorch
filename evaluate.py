import os
import sys

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

import torch
import torchvision
from torchvision.utils import save_image
import torchvision.transforms as transforms

import torch.optim as optim
from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from time import time
import math

from attunet import AttU_Net
from args import args

is_cuda = torch.cuda.is_available()
is_cuda = False

class MyTestDataset(Dataset):
    def __init__(self, img_path_file):
        f = open(img_path_file, 'r')
        img_list = f.read().splitlines()
        f.close()

        self.img_list = img_list
    
    def __getitem__(self, index):
        img = Image.open(self.img_list[index]).convert('RGB')
        
        custom_transform = transforms.Compose([transforms.Resize((args.HEIGHT,args.WIDTH)),
                                               transforms.ToTensor()])
        
        img = custom_transform(img)

        return img

    def __len__(self):

        return len(self.img_list)



if __name__ == '__main__':

    model = AttU_Net()

    checkpoint = torch.load(args.test_model, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    img_path_file = args.test_img

    testloader = DataLoader(MyTestDataset(img_path_file), batch_size=1, shuffle=False, num_workers=args.workers)

    if is_cuda:
        model.cuda()

    for i, (img) in enumerate(tqdm(testloader)):

        if is_cuda and args.gpu is not None:
            img = img.cuda()

        pred = model(img)

        save_image(pred,args.test_save_dir + '{}_eval.jpg'.format(i))
        
    print('Finished testing')

    