import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import gis_generator

os.makedirs("C:/Data/Output_Control", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=25, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)
Tensor = torch.FloatTensor

dataset = gis_generator.ImData("C:/Data/Images",opt.img_size)
train_size = int(.95*len(dataset))
val_size = len(dataset)-train_size
TrainDataset,ValDataset = torch.utils.data.random_split(dataset, [train_size,val_size])

dataloader = DataLoader(TrainDataset,opt.batch_size)

for i, imgs in enumerate(dataloader):
    # Adversarial ground truths
    valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
    fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

    # Configure input
    real_imgs = Variable(imgs.type(Tensor))

    if i % opt.sample_interval == 0:
        save_image(real_imgs.data[:25], "C:/Data/Output_Control/%d.png" % i, nrow=5, normalize=True)
