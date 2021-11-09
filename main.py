#main.py
import argparse
import logging
import os
import random
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from train import train
# from test1 import train1
from test import test
from Net.UNet import UNet, UNet_vit
# from Net.Mbcnn import MBCNN

from Net.MBCNN2 import MBCNN
#from class_tmp import MBCNN

parser = argparse.ArgumentParser()
parser.add_argument('--traindata_path', type=str,
                    default= '/databse4/jhkim/DataSet/8AIMDataset/train10000',    help='vit_patches_size, default is 16')
parser.add_argument('--testdata_path', type=str,
                    default= '/databse4/jhkim/DataSet/8AIMDataset/validation100', help='vit_patches_size, default is 16')
parser.add_argument('--testmode_path', type=str,
                    default= '/databse4/jhkim/DataSet/8AIMDataset/validation100', help='vit_patches_size, default is 16')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--num_worker', type=int, default=8,
                    help='number of workers')
parser.add_argument('--batchsize', type=int,default= 32,
                    help='mini batch size')
parser.add_argument('--max_epoch', type=int, default=800,
                    help='number of max_epoch')
parser.add_argument('--lr_decay', type=float,default=0.5,
                    help='adjucting lr_decay ')
parser.add_argument('--accumulation_steps', type=int ,default=1,
                    help='accumulation_steps but not used')
parser.add_argument('--loss_alpha', type=float,default= 0.8,
                    help='weight of charbonnier loss and L2loss, weight for charbonnier loss')
parser.add_argument('--save_every', type=int,default= 3,
                    help='saving period for pretrained weight ')
parser.add_argument('--name', type=str,default='MBCNN',
                    help='name for this experiment rate')
parser.add_argument('--psnr_axis_min', type=int,default=10,
                    help='mininum line for psnr graph')
parser.add_argument('--psnr_axis_max', type=int,default=70,
                    help='maximum line for psnr graph')
parser.add_argument('--psnrfolder', type=str,default='psnrfoler path was not configured',
                    help='psnrfoler path, define it first!!')
parser.add_argument('--pthfolder', type=str,default='pthfoler path was not configured',
                    help='pthfoler path, define it first!!')
parser.add_argument('--device', type=str, default='cuda or cpu',
                    help='device, define it first!!')
parser.add_argument('--save_prefix', type=str, default='/databse4/jhkim/PTHfolder/',
                    help='saving folder directory')
parser.add_argument('--bestperformance_saveevery', type=float, default=0.,
                    help='saving folder directory')
parser.add_argument('--bestperformance', type=float, default=0.,
                    help='saving folder directory')
parser.add_argument('--Train_pretrained_path', type=str, default = None,# '/databse4/jhkim/PTHfolder/211105_20:17_lr=150,3branch,loss_L1+ASL1/1pth_folder/Best_performance_MBCNN_ckpt_epoch045_psnr_31.3095_inputpsnr10.4399.tar',
                    help='saving folder directory')
parser.add_argument('--Test_pretrained_path', type=str, default = '/databse4/jhkim/PTHfolder/'
                                                                  '211107_24:22_IDW_TIP2018_train/1pth_folder/Best_performance_MBCNN_ckpt_epoch380_psnr_25.0115_inputpsnr18.6044',
                    help='saving folder directory')
parser.add_argument('--trainmode', type=bool, default=True,
                    help='saving folder directory')


args = parser.parse_args()
if __name__ == "__main__":
    # nFilters=64     multi = True    net = MBCNN(nFilters, multi)
    # net = UNet(3, 3)

    net = MBCNN(64).cuda()
    train(args, net)

    # train1(args, net)
    # test(args, net)


