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

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'

class MyTrainDataset(Dataset):
    def __init__(self, img1_path_file, img2_path_file):
        f1 = open(img1_path_file, 'r')
        img1_list = f1.read().splitlines()
        f1.close()
        f2 = open(img2_path_file, 'r')
        img2_list = f2.read().splitlines()
        f2.close()

        self.img1_list = img1_list
        self.img2_list = img2_list

    def transform(self, img1, img2):
        resize = transforms.Resize(size=(args.HEIGHT,args.WIDTH))
        img1 = resize(img1)
        img2 = resize(img2)

        degree = transforms.RandomRotation.get_params([-20,20])

        img1 = transforms.functional.rotate(img1, degree)
        img2 = transforms.functional.rotate(img2, degree)

        if random.random() > 0.5:
            img1 = transforms.functional.hflip(img1)
            img2 = transforms.functional.hflip(img2)

        if random.random() > 0.5:
            img1 = transforms.functional.vflip(img1)
            img2 = transforms.functional.vflip(img2)

        # Transform to tensor
        img1 = transforms.functional.to_tensor(img1)
        img2 = transforms.functional.to_tensor(img2)
        return img1, img2
    
    def __getitem__(self, index):
        dark_img = Image.open(self.img1_list[index]).convert('RGB')
        bright_img = Image.open(self.img2_list[index]).convert('RGB')     

        dark_img, bright_img = self.transform(dark_img, bright_img)

        return dark_img, bright_img

    def __len__(self):

        return len(self.img1_list)

def setup(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destropy_process_group

def main():
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if args.gpu is not None:    
        if not len(args.gpu) > torch.cuda.device_count():
            ngpus_per_node = len(args.gpu)
        else:
            print("We will use all available GPUs")
            ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = torch.cuda.device_count()
    
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, world_size=args.world_size, rank=args.rank)

    model = AttU_Net()
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.5)

    epoch = 0

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    if args.resume:
        if args.gpu is None:
            checkpoint = torch.load(args.resume)
        else:
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    else:
        pass
    
    dark_path_file = args.dataset1
    bright_path_file = args.dataset2

    criterion = nn.L1Loss()

    trainloader = DataLoader(MyTrainDataset(dark_path_file, bright_path_file), batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        
    for ep in range(epoch, args.epochs):

        pbar = tqdm(trainloader)
        idx = 0

        for dark_img, bright_img in pbar:

            if is_cuda and args.gpu is not None:
                dark_img = dark_img.cuda(args.gpu, non_blocking=True)
                bright_img = bright_img.cuda(args.gpu, non_blocking=True)

            optimizer.zero_grad()

            pred = model(dark_img)
            loss = criterion(pred,bright_img)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            idx += 1
        if torch.cuda.current_device() == 0:
            print("GPU{} Total_loss:".format(torch.cuda.current_device()), loss)

        if (ep + 1) % args.save_per_epoch == 0:
            save_image(dark_img, '{}_inpt.png'.format(ep))
            save_image(pred, '{}_pred.png'.format(ep))
            save_image(bright_img, '{}_grtr.png'.format(ep))
            # Save model
            torch.save({
                        'epoch': ep,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
                    }, args.save_model_dir + 'ckpt_{}.pt'.format(ep))
        

    print('Finished training')

if __name__ == "__main__":
    main()