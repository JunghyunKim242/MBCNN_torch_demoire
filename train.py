import argparse
import os
import random
import sys
import time
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
from Net.LossNet import L2_LOSS, L1_LOSS, L1_Sobel_Loss, L1_Advanced_Sobel_Loss
from dataset.dataset import AIMMoire_dataset_test, AIMMoire_dataset, TIP2018moire_dataset_train
from torchnet import meter

import colour
import time
import torchvision
import math

def train(args, model):
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args.save_prefix = args.save_prefix + Time2Str()+'_MBCNN_AIMDataset_cuda_check'

    if not os.path.exists(args.save_prefix):    os.makedirs(args.save_prefix)

    print('torch devices = \t\t\t', args.device)
    print('save_path = \t\t\t\t', args.save_prefix)

    args.pthfoler       = os.path.join( args.save_prefix , '1pth_folder/')
    args.psnrfolder     = os.path.join( args.save_prefix , '1psnr_folder/')
    if not os.path.exists(args.pthfoler)        :   os.makedirs(args.pthfoler)
    if not os.path.exists(args.psnrfolder)      :   os.makedirs(args.psnrfolder)

    # AIMMoire_dataset
    Moiredata_train = AIMMoire_dataset(args.traindata_path)
    train_dataloader = DataLoader(Moiredata_train,
                                  batch_size=args.batchsize,
                                  shuffle=True,
                                  num_workers=args.num_worker,
                                  drop_last=True)

    Moiredata_test = AIMMoire_dataset_test(args.testdata_path, crop = False)
    test_dataloader = DataLoader(Moiredata_test,
                                 batch_size=2,
                                 shuffle=True,
                                 num_workers=args.num_worker,
                                 drop_last=False)

    #TIP2018moire_dataset_
    # Moiredata_train = TIP2018moire_dataset_train(args.traindata_path)
    # train_dataloader = DataLoader(Moiredata_train,
    #                               batch_size=args.batchsize,
    #                               shuffle=True,
    #                               num_workers=args.num_worker,
    #                               drop_last=True)
    #
    # Moiredata_test = TIP2018moire_dataset_train(args.testdata_path,crop = False)
    # test_dataloader = DataLoader(Moiredata_test,
    #                              batch_size=2,
    #                              shuffle=True,
    #                              num_workers=args.num_worker,
    #                              drop_last=False)



    lr = args.lr
    last_epoch = 0
    optimizer = optim.Adam(params=model.parameters(),
                           lr=lr,
                           # weight_decay=0.01 #0.005
                           )

    list_psnr_output = []
    list_loss_output = []
    list_psnr_input = []
    list_loss_input = []

    model = nn.DataParallel(model)
    # model = model.cuda()
    if args.Train_pretrained_path:
        checkpoint = torch.load(args.Train_pretrained_path)
        model.load_state_dict(checkpoint['model'])
        last_epoch = checkpoint["epoch"]
        optimizer_state = checkpoint["optimizer"]
        optimizer.load_state_dict(optimizer_state)
        lr = checkpoint['lr']
        list_psnr_output = checkpoint['list_psnr_output']
        list_loss_output = checkpoint['list_loss_output']
        list_psnr_input = checkpoint['list_psnr_input']
        list_loss_input = checkpoint['list_loss_input']
    model.train()

    # criterion_l2 = L2_LOSS()
    criterion_l1 = L1_LOSS()
    criterion_advanced_sobel_l1 = L1_Advanced_Sobel_Loss()

    psnr_meter  = meter.AverageValueMeter()
    Loss_meter1  = meter.AverageValueMeter()
    Loss_meter2 = meter.AverageValueMeter()
    Loss_meter3 = meter.AverageValueMeter()
    Loss_meter4 = meter.AverageValueMeter()

    for epoch in range(args.max_epoch):
        print('\nepoch = {} / {}'.format(epoch + 1, args.max_epoch))
        start = time.time()
        if epoch < last_epoch:
            continue
        # print('Test set : \tLoss = {0:0.4f},\tPSNR = {1:0.4f},\tinput_PSNR = {2:0.4f} '.format(loss_output, psnr_output, psnr_input))

        Loss_meter1.reset()
        Loss_meter2.reset()
        Loss_meter3.reset()
        Loss_meter4.reset()
        psnr_meter.reset()

        for  ii, (moires, clears_list, labels) in tqdm(enumerate(train_dataloader)):
            # if ii ==0:
            #     # model.pre_block1.ScaleLayer1
            #     for name, param in model.named_parameters:
            #         print('\n\n',name)
            #         print('param',param)

            moires = moires.cuda()
            clear3, clear2, clear1 = clears_list
            # clear1 = clears_list

            clear3 = clear3.to(args.device)
            clear2 = clear2.to(args.device)
            clear1 = clear1.to(args.device)
            # clear1 = clear1.cuda()

            output3, output2, output1 = model(moires) # 32,1,256,256 = 32,1,256,256
            # output1 = model(moires)

            Loss_l1                 = criterion_l1(output1, clear1)
            Loss_advanced_sobel_l1  = criterion_advanced_sobel_l1(output1, clear1)
            Loss_l12                 = criterion_l1(output2, clear2)
            Loss_advanced_sobel_l12  = criterion_advanced_sobel_l1(output2, clear2)
            Loss_l13                 = criterion_l1(output3, clear3)
            Loss_advanced_sobel_l13  = criterion_advanced_sobel_l1(output3, clear3)

            Loss1 = Loss_l1  + (0.25)*Loss_advanced_sobel_l1
            Loss2 = Loss_l12 + (0.25)*Loss_advanced_sobel_l12
            Loss3 = Loss_l13 + (0.25)*Loss_advanced_sobel_l13

            loss = Loss1 + Loss2 + Loss3

            loss_check1 = Loss1
            loss_check2 = Loss_l1
            loss_check3 = Loss_advanced_sobel_l1
            optimizer.zero_grad()
            loss.backward()            # loss.backward(retain_graph = True) # retain_graph = True
            optimizer.step()

            moires = tensor2im(moires)
            output1 = tensor2im(output1)
            clear1 = tensor2im(clear1)

            # psnr = colour.utilities.metric_psnr(outputs, clears)
            psnr = PSNR(output1, clear1)
            psnr_meter.add(psnr)
            Loss_meter1.add(loss.item())
            Loss_meter2.add(loss_check1.item())
            Loss_meter3.add(loss_check2.item())
            Loss_meter4.add(loss_check3.item())


        # print('training set : \tLoss = {0:0.4f},\tPSNR = {1:0.4f},\tLoss_Check = {0:0.4f},\tLoss_Check2 = {0:0.4f} \t'.format(Loss_meter.value()[0], psnr_meter.value()[0], Loss_meter2.value()[0],Loss_meter3.value()[0] ))
        print('training set : \tPSNR = {:f},\t loss = {:f},\t Loss1(scale) = {:f},\t Loss_L1 = {:f}, \t, Loss_sobel = {:f},\t '
              .format(psnr_meter.value()[0], Loss_meter1.value()[0],  Loss_meter2.value()[0], Loss_meter3.value()[0], Loss_meter4.value()[0] ))
        loss_output, psnr_output, loss_input, psnr_input = val(model, test_dataloader, epoch,args)
        print('Test set : \tLoss = {:0.4f},'.format(loss_output) + '\033[30m \033[43m' + ' PSNR = {:0.4f}, \t best PSNR ={:0.4f}'.format(psnr_output,args.bestperformance) + '\033[0m' )


        list_psnr_output.append( round(psnr_output,5) )
        list_loss_output.append( round(loss_output,5))
        list_psnr_input.append( round(psnr_input,5))
        list_loss_input.append( round(loss_input,5))


        if psnr_output > args.bestperformance  :  # 每5个epoch保存一次
            args.bestperformance = psnr_output
            file_name = args.pthfoler + 'Best_performance_{:}_statedict_epoch{:03d}_psnr{:}.pth'.format(args.name, epoch + 1,round(psnr_output,4))
            torch.save(model.state_dict(), file_name)


        if (epoch + 1) % args.save_every == 0 or epoch == 0 :  # 每5个epoch保存一次
            file_name = args.pthfoler + 'Best_performance_{:}_ckpt_epoch{:03d}_psnr_{:0.4f}_inputpsnr{:0.4f}.tar'.format(args.name,epoch+1, round(psnr_output,4),round(psnr_input,4) )
            checkpoint = {  'epoch': epoch + 1,
                            "optimizer": optimizer.state_dict(),
                            "model": model.state_dict(),
                            "lr": lr,
                            "list_psnr_output": list_psnr_output,
                            "list_loss_output": list_loss_output,
                            "list_psnr_input": list_psnr_input,
                            "list_loss_input": list_loss_input,
                            }
            torch.save(checkpoint, file_name)


            with open(args.save_prefix + "/PSNR_validation_set_output_psnr.txt", 'w') as f:
                f.write("psnr_output: {:}\n".format(list_psnr_output))
            with open(args.save_prefix + "/Loss_validation_set_output_loss.txt", 'w') as f:
                f.write("loss_output: {:}\n".format(list_loss_output))
            with open(args.save_prefix + "/PSNR_validation_set_Input_.txt", 'w') as f:
                f.write("input_psnr: {:}\n".format(list_psnr_input))
            with open(args.save_prefix + "/Loss_validation_set_Input_.txt", 'w') as f:
                f.write("input_loss: {:}\n".format(list_loss_input))


            if epoch >= 1:
                plt.figure()
                plt.plot(range(1, epoch + 2, 1), list_psnr_output, 'r', label='Validation_set')
                plt.plot(range(1, epoch + 2, 1), list_psnr_input , 'b', label='Input(validation)')
                plt.xlabel('Epochs')
                plt.ylabel('PSNR')
                plt.axis([1, epoch + 1, args.psnr_axis_min, args.psnr_axis_max])  # 10-50
                plt.title('PSNR per epoch')
                plt.grid(linestyle='--', color='lavender')
                plt.legend(loc='lower right')
                plt.savefig(args.psnrfolder + 'PSNR_graph_{name}_{epoch}.png'.format(name=args.name,epoch = epoch+1))
                plt.close('all')

        if epoch % 150 == 0 and epoch !=0:
            lr *= 0.3
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # if epoch > 5:
            # list_tmp = list_loss_output[-5:]
            # for j in range(4):
                # sub = 10 * (math.log10( ( list_tmp[j] / list_tmp[j+1] ) ))
                # if sub > 0.001: break
                # if j == 3:
                #     print('\033[30m \033[41m' + 'LR was Decreased!!!{:} > {:}\t\t\t\t\t\t\t\t\t'.format(lr,lr/2) + '\033[0m' )
                #     lr = lr * 0.5
                #     for param_group in optimizer.param_groups:  param_group['lr'] = lr
                # if lr < 1e-6:                                   exit()


        if epoch == (args.max_epoch-1):
            file_name2 = args.pthfoler + '{0}_stdc_epoch{1}.pth'.format(args.name, epoch + 1)
            torch.save(model.state_dict(), file_name2)


        print('1 epoch spends:{:.2f}sec\t remain {:2d}:{:2d} hours'.format(
            (time.time() - start),
            int((args.max_epoch - epoch) * (time.time() - start) // 3600) ,
            int((args.max_epoch - epoch) * (time.time() - start) % 3600 / 60 ) ))

    return "Training Finished!"




##after
def val(model, dataloader,epoch,args): # 맨처음 확인할때의 epoch == -1
    model.eval()

    # criterion_l2 = L2_LOSS()
    criterion_l1                = L1_LOSS()
    criterion_advanced_sobel_l1 = L1_Advanced_Sobel_Loss()

    loss_output_meter = meter.AverageValueMeter()
    loss_input_meter = meter.AverageValueMeter()
    psnr_output_meter = meter.AverageValueMeter()
    psnr_input_meter = meter.AverageValueMeter()
    loss_output_meter.reset()
    loss_input_meter.reset()
    psnr_output_meter.reset()
    psnr_input_meter.reset()

    image_train_path_demoire = "{0}/epoch_{1}_validation_set_{2}/".format(args.save_prefix, epoch + 1, "demoire")
    if not os.path.exists(image_train_path_demoire) and (epoch + 1) % args.save_every == 0 : os.makedirs(image_train_path_demoire)

    image_train_path_moire = "{0}/epoch_{1}_validation_set_{2}/".format(args.save_prefix, 1, "moire")
    image_train_path_clean = "{0}/epoch_{1}_validation_set_{2}/".format(args.save_prefix, 1, "clean")
    if not os.path.exists(image_train_path_moire): os.makedirs(image_train_path_moire)
    if not os.path.exists(image_train_path_clean): os.makedirs(image_train_path_clean)

    for ii, (val_moires, val_clears_list, labels) in tqdm(enumerate(dataloader)):

        with torch.no_grad():
            val_moires = val_moires.to(args.device)
            clear3, clear2, clear1 = val_clears_list
            # clear1 = val_clears_list
            clear3 = clear3.to(args.device)
            clear2 = clear2.to(args.device)
            clear1 = clear1.to(args.device)
            output3, output2, output1 = model(val_moires)
            # output1 = model(val_moires)

            # val_clears = val_clears.to(args.device)
            # val_outputs = model(val_moires)
            if ii ==0 and epoch ==0:  print('val_morie.shape & val_outputs \t\t',val_moires.shape,output1.shape)

        loss_l1                 = criterion_l1(output1, clear1)
        loss_advanced_sobel_l1  = criterion_advanced_sobel_l1(output1, clear1)
        Loss_l12                = criterion_l1(output2, clear2)
        Loss_advanced_sobel_l12 = criterion_advanced_sobel_l1(output2, clear2)
        Loss_l13                = criterion_l1(output3, clear3)
        Loss_advanced_sobel_l13 = criterion_advanced_sobel_l1(output3, clear3)

        Loss1 = loss_l1  + (0.25) * loss_advanced_sobel_l1
        Loss2 = Loss_l12 + (0.25) * Loss_advanced_sobel_l12
        Loss3 = Loss_l13 + (0.25) * Loss_advanced_sobel_l13
        loss  = Loss1 + Loss2 + Loss3

        loss_output_meter.add(loss.item())
        loss_l2_input = criterion_l1(val_moires, clear1)
        loss_input_meter.add(loss_l2_input.item())

        val_moires = tensor2im(val_moires) # type tensor to numpy .detach().cpu().float().numpy()
        output1 = tensor2im(output1)
        clear1 = tensor2im(clear1)

        bs = val_moires.shape[0]
        if epoch != -1:
            for jj in range(bs):
                output, clear, moire, label = output1[jj], clear1[jj], val_moires[jj], labels[jj]

                # psnr_output_individual = colour.utilities.metric_psnr(output, clear)
                psnr_output_individual = PSNR(output, clear)
                # psnr_input_individual = colour.utilities.metric_psnr(moire, clear)
                psnr_input_individual = PSNR(moire, clear)
                psnr_output_meter.add(psnr_output_individual)
                psnr_input_meter.add(psnr_input_individual)

                if (epoch + 1) % args.save_every == 0 or epoch == 0:  # 每5个epoch保存一次
                    img_path = "{0}/{1}_epoch:{2:04d}_demoire_PSNR:{3:.4f}_demoire.png".format(image_train_path_demoire, label,epoch+1 ,psnr_output_individual)
                    save_single_image(output, img_path)
                    img_path = "{0}/{1}_epoch:{2:04d}_clear_clear.png".format(image_train_path_demoire, label,epoch+1 )
                    save_single_image(clear, img_path)
                    img_path = "{0}/{1}_epoch:{2:04d}_moire_PSNR:{3:.4f}_moire.png".format(image_train_path_demoire, label,epoch+1 ,psnr_input_individual)
                    save_single_image(moire, img_path)

                if epoch == 0:
                    psnr_in_gt = PSNR(moire, clear)
                    img_path2 = "{0}/{1}_{2:.4f}_moire.png".format( image_train_path_moire, label, psnr_in_gt)
                    img_path3 = "{0}/{1}_clean.png".format(         image_train_path_clean, label)
                    save_single_image(moire, img_path2)
                    save_single_image(clear, img_path3)

    return loss_output_meter.value()[0], psnr_output_meter.value()[0],loss_input_meter.value()[0],psnr_input_meter.value()[0]


