import argparse
import os
import random
import sys
import numpy as np
import torch
from torch import nn
# from torch.nn import MSELoss, L1Loss
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
import matplotlib
import matplotlib.pyplot as plt
from Util.util_collections import tensor2im, save_single_image, PSNR, Time2Str
from dataset.dataset import Moire_dataset, AIMMoire_dataset,AIMMoire_dataset_test
from torchnet import meter
from skimage.metrics import peak_signal_noise_ratio
import torchvision


def test(args, model):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args.save_prefix = args.save_prefix + Time2Str()+'_Test_AIM_psnr'

    if not os.path.exists(args.save_prefix):    os.makedirs(args.save_prefix)

    print('torch devices = ', args.device)
    print('save_path = ', args.save_prefix)

    Moiredata_test = AIMMoire_dataset_test(args.testmode_path, crop = False)
    # Moiredata_test = TIP2018moire_dataset_test(args.testmode_path)
    test_dataloader = DataLoader(Moiredata_test,
                                 batch_size=4,
                                 shuffle=False,
                                 num_workers=args.num_worker,
                                 drop_last=False)

    model = nn.DataParallel(model)
    model = model.to(args.device)
    if args.Test_pretrained_path:
        file_ext = (args.Test_pretrained_path.split(".")[-1])
        if file_ext =='pth': # statedict
            model.load_state_dict(torch.load(args.Test_pretrained_path),strict = False)
        elif file_ext =='tar':
            checkpoint = torch.load(args.Test_pretrained_path)
            model.load_state_dict(checkpoint['model'])
    model.eval()

    psnr_output_meter = meter.AverageValueMeter()
    psnr_input_meter = meter.AverageValueMeter()


    image_train_path_moire = "{0}/{1}".format(args.save_prefix, "TEST_Moirefolder")
    image_train_path_clean = "{0}/{1}".format(args.save_prefix, "TEST_Cleanfolder")
    image_train_path_demoire = "{0}/{1}".format(args.save_prefix, "TEST_Demoirefolder")


    if not os.path.exists(image_train_path_moire):      os.makedirs(image_train_path_moire)
    if not os.path.exists(image_train_path_clean):      os.makedirs(image_train_path_clean)
    if not os.path.exists(image_train_path_demoire):    os.makedirs(image_train_path_demoire)


    for ii ,(moires,clears,labels) in tqdm(enumerate(test_dataloader)):
        with torch.no_grad():
            moires = moires.to(args.device)
            clears1 = clears[2].to(args.device) #batch,1,256,256(unet,moire)
            _,_,outputs1 = model(moires)


        # torchvision.utils.save_image(moires,'moires.png')

        moires = tensor2im(moires)
        outputs = tensor2im(outputs1)
        clears1 = tensor2im(clears1)


        bs = moires.shape[0]
        for jj in range(bs):
            moire, clear, label, output = moires[jj], clears1[jj], labels[jj], outputs[jj]

            psnr_output = peak_signal_noise_ratio(output, clear)
            psnr_output_meter.add(psnr_output)
            psnr_input = peak_signal_noise_ratio(moire, clear)
            psnr_input_meter.add(psnr_input)

            img_path1 = "{0}/{1}_moire_{2:.4f}.png".format(image_train_path_moire, label, psnr_input)
            save_single_image(moire, img_path1)
            img_path2 = "{0}/{1}_clean.png".format(image_train_path_clean, label)
            save_single_image(clear, img_path2)
            img_path3 = "{0}/{1}_demoire_{2:.4f}.png".format(image_train_path_demoire, label, psnr_output)
            save_single_image(output, img_path3)

    print('Test datset_PSNR = ',psnr_output_meter.value()[0])


