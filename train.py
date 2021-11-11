import argparse
import os
# import random,sys, matplotlib, torchvision
import time
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from Util.util_collections import tensor2im, save_single_image, PSNR, Time2Str
from Net.LossNet import L1_LOSS, L1_Advanced_Sobel_Loss
from dataset.dataset import AIMMoire_dataset_test, AIMMoire_dataset, TIP2018moire_dataset_train
from torchnet import meter
from skimage.metrics import peak_signal_noise_ratio
import math

def train(args, model):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # print('device',args.device)
    # print('torch.cuda.is_available() = ',torch.cuda.is_available())
    # print('torch.cuda.device_count() = ', torch.cuda.device_count())
    # print('Current cuda device = ', torch.cuda.current_device())
    # print('torch.cuda.get_device_name(args.device)= ',torch.cuda.get_device_name(args.device)) # RTX3090

    args.save_prefix = args.save_prefix + Time2Str()+'_MBCNN_AIMData_Loss=sum,LR=0.3'
    if not os.path.exists(args.save_prefix):    os.makedirs(args.save_prefix)

    print('torch devices = \t\t\t', args.device)
    print('save_path = \t\t\t\t', args.save_prefix)

    args.pthfoler       = os.path.join( args.save_prefix , '1pth_folder/')
    args.psnrfolder     = os.path.join( args.save_prefix , '1psnr_folder/')
    if not os.path.exists(args.pthfoler)        :   os.makedirs(args.pthfoler)
    if not os.path.exists(args.psnrfolder)      :   os.makedirs(args.psnrfolder)

    # AIMMoire_dataset 10000장, 100장
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

    lr = args.lr
    last_epoch = 0
    optimizer = optim.Adam(params=model.parameters(),
                           lr=lr )

    list_psnr_output = []
    list_loss_output = []

    model = nn.DataParallel(model)
    model = model.cuda()
    if args.Train_pretrained_path:
        checkpoint = torch.load(args.Train_pretrained_path)
        model.load_state_dict(checkpoint['model'])
        last_epoch = checkpoint["epoch"]
        optimizer_state = checkpoint["optimizer"]
        optimizer.load_state_dict(optimizer_state)
        lr = checkpoint['lr']
        list_psnr_output = checkpoint['list_psnr_output']
        list_loss_output = checkpoint['list_loss_output']
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

        Loss_meter1.reset()
        Loss_meter2.reset()
        Loss_meter3.reset()
        Loss_meter4.reset()
        psnr_meter.reset()

        for  ii, (moires, clears_list, labels) in tqdm(enumerate(train_dataloader)):

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
            loss_check3 = (0.25)*Loss_advanced_sobel_l1
            optimizer.zero_grad()
            loss.backward()            # loss.backward(retain_graph = True) # retain_graph = True
            optimizer.step()

            moires = tensor2im(moires)
            output1 = tensor2im(output1)
            clear1 = tensor2im(clear1)

            psnr = peak_signal_noise_ratio(output1, clear1)
            psnr_meter.add(psnr)
            Loss_meter1.add(loss.item())
            Loss_meter2.add(loss_check1.item())
            Loss_meter3.add(loss_check2.item())
            Loss_meter4.add(loss_check3.item())


        print('training set : \tPSNR = {:f}\t loss = {:f}\t Loss1(scale) = {:f} \t Loss_L1 = {:f} + Loss_sobel = {:f},\t '
              .format(psnr_meter.value()[0], Loss_meter1.value()[0],  Loss_meter2.value()[0], Loss_meter3.value()[0], Loss_meter4.value()[0] ))
        psnr_output, loss_output1, loss_output2, loss_output3, loss_output4 = val(model, test_dataloader, epoch,args)
        print('Test set : \tloss = {:0.4f} \t Loss_1 = {:0.4f} \t Loss_L1 = {:0.4f} \t Loss_ASL = {:0.4f}'.format(loss_output1,loss_output2,loss_output3,loss_output4) )
        print('Test set : \t' + '\033[30m \033[43m' + 'PSNR = {:0.4f}'.format(psnr_output)+'\033[0m'+'\tbest PSNR ={:0.4f}'.format(args.bestperformance) )

        list_psnr_output.append( round(psnr_output,5) )
        list_loss_output.append( round(loss_output1,5))


        if psnr_output > args.bestperformance  :  # 每5个epoch保存一次
            args.bestperformance = psnr_output
            file_name = args.pthfoler + 'Best_performance_{:}_statedict_epoch{:03d}_psnr{:}.pth'.format(args.name, epoch + 1,round(psnr_output,4))
            torch.save(model.state_dict(), file_name)
            print('\033[30m \033[42m' + 'PSNR WAS UPDATED! '+'\033[0m')


        if (epoch + 1) % args.save_every == 0 or epoch == 0 :  # 每5个epoch保存一次
            file_name = args.pthfoler + 'Best_performance_{:}_ckpt_epoch{:03d}_psnr_{:0.4f}_.tar'.format(args.name,epoch+1, round(psnr_output,4) )
            checkpoint = {  'epoch': epoch + 1,
                            "optimizer": optimizer.state_dict(),
                            "model": model.state_dict(),
                            "lr": lr,
                            "list_psnr_output": list_psnr_output,
                            "list_loss_output": list_loss_output,
                            }
            torch.save(checkpoint, file_name)


            with open(args.save_prefix + "/1_PSNR_validation_set_output_psnr.txt", 'w') as f:
                f.write("psnr_output: {:}\n".format(list_psnr_output))
            with open(args.save_prefix + "/1_Loss_validation_set_output_loss.txt", 'w') as f:
                f.write("loss_output: {:}\n".format(list_loss_output))


            if epoch >= 1:
                plt.figure()
                plt.plot(range(1, epoch + 2, 1), list_psnr_output, 'r', label='Validation_set')
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
            print('\033[30m \033[41m' + 'LR was Decreased!!!{:} > {:}\t\t\t\t\t\t\t\t\t'.format(lr,lr/2) + '\033[0m' )
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # if epoch > 5:
        #     list_tmp = list_loss_output[-5:]
        #     for j in range(4):
        #         sub = 10 * (math.log10( ( list_tmp[j] / list_tmp[j+1] ) ))
        #         if sub > 0.001: break
        #         if j == 3:
        #             print('\033[30m \033[41m' + 'LR was Decreased!!!{:} > {:}\t\t\t\t\t\t\t\t\t'.format(lr,lr/2) + '\033[0m' )
        #             lr = lr * 0.5
        #             for param_group in optimizer.param_groups:  param_group['lr'] = lr
        #         if lr < 1e-6:                                   exit()


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

    psnr_output_meter = meter.AverageValueMeter()
    loss_meter1 = meter.AverageValueMeter()
    loss_meter2 = meter.AverageValueMeter()
    loss_meter3 = meter.AverageValueMeter()
    loss_meter4 = meter.AverageValueMeter()

    psnr_output_meter.reset()
    loss_meter1.reset()
    loss_meter2.reset()
    loss_meter3.reset()
    loss_meter4.reset()

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
            # if ii ==0 and epoch ==0:  print('val_morie.shape & val_outputs \t\t',val_moires.shape,output1.shape)

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

        loss_meter1.add(loss.item())
        loss_meter2.add(Loss1.item())
        loss_meter3.add(loss_l1.item())
        loss_meter4.add(loss_advanced_sobel_l1.item())

        val_moires = tensor2im(val_moires) # type tensor to numpy .detach().cpu().float().numpy()
        output1 = tensor2im(output1)
        clear1 = tensor2im(clear1)

        bs = val_moires.shape[0]
        if epoch != -1:
            for jj in range(bs):
                output, clear, moire, label = output1[jj], clear1[jj], val_moires[jj], labels[jj]

                psnr_output_individual = peak_signal_noise_ratio(output, clear)
                psnr_output_meter.add(psnr_output_individual)
                psnr_input_individual = peak_signal_noise_ratio(moire, clear)

                if (epoch + 1) % args.save_every == 0 or epoch == 0:  # 每5个epoch保存一次
                    img_path = "{0}/{1}_epoch:{2:04d}_demoire_PSNR:{3:.4f}_demoire.png".format(image_train_path_demoire, label,epoch+1 ,psnr_output_individual)
                    save_single_image(output, img_path)
                    # img_path = "{0}/{1}_epoch:{2:04d}_clear.png".format(image_train_path_demoire, label,epoch+1 )
                    # save_single_image(clear, img_path)
                    # img_path = "{0}/{1}_epoch:{2:04d}_moire_PSNR:{3:.4f}_moire.png".format(image_train_path_demoire, label,epoch+1 ,psnr_input_individual)
                    # save_single_image(moire, img_path)

                if epoch == 0:
                    psnr_in_gt = peak_signal_noise_ratio(moire, clear)
                    img_path2 = "{0}/{1}_moire_{2:.4f}_moire.png".format( image_train_path_moire, label, psnr_in_gt)
                    img_path3 = "{0}/{1}_clean_.png".format(         image_train_path_clean, label)
                    save_single_image(moire, img_path2)
                    save_single_image(clear, img_path3)

    return psnr_output_meter.value()[0], loss_meter1.value()[0], loss_meter2.value()[0], loss_meter3.value()[0], loss_meter4.value()[0]


