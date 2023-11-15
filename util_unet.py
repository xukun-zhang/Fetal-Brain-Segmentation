import sys
sys.path.append('..')

import numpy as np
import os
from datetime import datetime
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import sys
import SimpleITK as sitk
from scipy.ndimage import zoom
class dice_loss(nn.Module):
    # def forward(self, uout, label, label_1, label_2):

    def forward(self,uout, uout_1, uout_2, uout_3, label, label_1, label_2, label_3):
    # def forward(self, uout, uout_1, label, label_1):
        """soft dice loss"""
        eps = 1e-7

        iflat = uout.view(-1)
        tflat = label.view(-1)
        intersection = (iflat * tflat).sum()
        dice_0 = 1 - 2. * intersection / ((iflat ** 2).sum() + (tflat ** 2).sum() + eps)

        iflat = uout_1.view(-1)
        tflat = label_1.view(-1)
        intersection = (iflat * tflat).sum()
        dice_1 = 1 - 2. * intersection / ((iflat ** 2).sum() + (tflat ** 2).sum() + eps)

        iflat = uout_2.view(-1)
        tflat = label_2.view(-1)
        intersection = (iflat * tflat).sum()
        dice_2 = 1 - 2. * intersection / ((iflat ** 2).sum() + (tflat ** 2).sum() + eps)

        iflat = uout_3.view(-1)
        tflat = label_3.view(-1)
        intersection = (iflat * tflat).sum()
        dice_3 = 1 - 2. * intersection / ((iflat ** 2).sum() + (tflat ** 2).sum() + eps)

        dice_loss = (dice_0 + dice_1 + dice_2 + dice_3) / 4
        return dice_loss




def get_acc(uout, uout_1, uout_2, uout_3, label, label_1, label_2, label_3):
#def get_acc(uout, uout_1, label, label_1):
    """soft dice score"""
    eps = 1e-7

    iflat = (uout.view(-1) > 0.5).float()
    tflat = label.view(-1)
    intersection = (iflat * tflat).sum()
    dice_0 = 2. * intersection / ((iflat ** 2).sum() + (tflat ** 2).sum() + eps)

    iflat = (uout_1.view(-1) > 0.5).float()
    tflat = label_1.view(-1)
    intersection = (iflat * tflat).sum()
    dice_1 = 2. * intersection / ((iflat ** 2).sum() + (tflat ** 2).sum() + eps)

    iflat = (uout_2.view(-1) > 0.5).float()
    tflat = label_2.view(-1)
    intersection = (iflat * tflat).sum()
    dice_2 = 2. * intersection / ((iflat ** 2).sum() + (tflat ** 2).sum() + eps)

    iflat = (uout_3.view(-1) > 0.5).float()
    tflat = label_3.view(-1)
    intersection = (iflat * tflat).sum()
    dice_3 = 2. * intersection / ((iflat ** 2).sum() + (tflat ** 2).sum() + eps)

    return dice_0, dice_1, dice_2, dice_3




def get_test_acc(uout, label):
#def get_acc(uout, uout_1, label, label_1):
    """soft dice score"""
    eps = 1e-7
    # iflat = (uout.view(-1) > 0.5).float()
    iflat = uout.view(-1).float()
    tflat = label.view(-1)

    intersection = (iflat * tflat).sum()
    dice_0 = 2. * intersection / ((iflat ** 2).sum() + (tflat ** 2).sum() + eps)

    dice_list = []


    for i in range(len(uout)):
        i_out = uout[i].view(-1)

        i_mask = label[i].view(-1)
        intersection = (i_out * i_mask).sum()
        dice_i = 2. * intersection / ((i_out ** 2).sum() + (i_mask ** 2).sum() + eps)
        # print("----------:", conf)
        dice_list.append(dice_i)

    return dice_0, dice_list

'''
    是否可以设置一个3、5循环，然后train函数放在循环里面，每一次循环使用的train_data不同，
    但是每一次循环都加载上一次保存最优的那个模型！
'''


def train(net, train_data, valid_data, num_epochs, optimizer, criterion):
    if torch.cuda.is_available():
        net = net.cuda()
        print("使用了 cuda")
    else:
        print("没使用cuda")
    prev_time = datetime.now()
    # 超算上用于保存模型的路径
    save = 'G:/Code/Fetalbrain/coarse stage/save_stage_one/save_model'
    # 定义初始化正确率为 0
    best_acc = 0
    for epoch in range(num_epochs):
        train_loss = 0
        dice_0_acc = 0
        dice_1_acc = 0
        dice_2_acc = 0
        dice_3_acc = 0
        net = net.train()
        index = 0
        for im, label, name, content in train_data:
            im, label = im.permute(1, 0, 2, 3), label.permute(1, 0, 2, 3)
            # print("im.shape, label.shape:", im.shape, label.shape)
            index = index + 1
            label_1 = nn.functional.interpolate(label, scale_factor=0.5, mode="nearest")
            label_2 = nn.functional.interpolate(label, scale_factor=0.25, mode="nearest")
            label_3 = nn.functional.interpolate(label, scale_factor=0.125, mode="nearest")
            im, label = im.cuda(), label.cuda()
            label_1 = label_1.cuda()
            label_2 = label_2.cuda()
            label_3 = label_3.cuda()


            # forward
            uout, uout_1, uout_2, uout_3 = net(im)     # # [(bs, 1, h, w),(bs, 1, h, w),(bs, 1, h, w),(bs, 1, h, w)]
            loss = criterion(uout, uout_1, uout_2, uout_3, label, label_1, label_2, label_3)     # dice loss

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            train_loss += loss.item()

            dice_0, dice_1, dice_2, dice_3 = get_acc(uout, uout_1, uout_2, uout_3, label, label_1, label_2, label_3)
            dice_0_acc = dice_0_acc + dice_0
            dice_1_acc = dice_1_acc + dice_1
            dice_2_acc = dice_2_acc + dice_2
            dice_3_acc = dice_3_acc + dice_3

        print('index in train-data, and the length of train-data:', index)
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        index = 0
        if valid_data is not None:
            valid_loss = 0
            val_acc_0 = 0
            val_acc_1 = 0
            val_acc_2 = 0
            val_acc_3 = 0
            with torch.no_grad():
                net = net.eval()
                for im, label, name, content in valid_data:
                    index = index + 1
                    im, label = im.permute(1, 0, 2, 3), label.permute(1, 0, 2, 3)
                    label_1 = nn.functional.interpolate(label, scale_factor=0.5, mode="nearest")
                    label_2 = nn.functional.interpolate(label, scale_factor=0.25, mode="nearest")

                    label_3 = nn.functional.interpolate(label, scale_factor=0.125, mode="nearest")
                    im, label = im.cuda(), label.cuda()
                    label_1 = label_1.cuda()
                    label_2 = label_2.cuda()
                    label_3 = label_3.cuda()

                    uout, uout_1, uout_2, uout_3 = net(im)
                    loss = criterion(uout, uout_1, uout_2, uout_3, label, label_1, label_2, label_3)
                    valid_loss += loss.item()
                    dice_0, dice_1, dice_2, dice_3 = get_acc(uout, uout_1, uout_2, uout_3, label, label_1, label_2, label_3)
                    val_acc_0 = val_acc_0 + dice_0
                    val_acc_1 = val_acc_1 + dice_1
                    val_acc_2 = val_acc_2 + dice_2
                    val_acc_3 = val_acc_3 + dice_3

                    save_img, save_label, save_mask = im.cpu().permute(1, 0, 2, 3).contiguous().squeeze(0), label.cpu().permute(1, 0, 2, 3).contiguous().squeeze(0), uout.cpu().permute(1, 0, 2, 3).contiguous().squeeze(0)
                    save_mask[save_mask>0.5] = 1
                    save_img, save_label, save_mask = sitk.GetImageFromArray(save_img), sitk.GetImageFromArray(save_label), sitk.GetImageFromArray(save_mask)


                    sitk.WriteImage(save_img, str(name[0]) + "_img.nii.gz")
                    sitk.WriteImage(save_label, str(name[0]) + "_lab.nii.gz")
                    sitk.WriteImage(save_mask, str(name[0]) + "_mask.nii.gz")

            epoch_str = (
                        "Epoch %d. Train Loss: %f, Train dice 0: %f, Train dice 1: %f, Train dice 2: %f, Train dice 3: %f, Valid loss: %f, Valid dice 0: %f, Valid dice 1: %f, Valid dice 2: %f, Valid dice 3: %f, len(valid_data): %d"
                        % (epoch, train_loss / len(train_data),
                    dice_0_acc / len(train_data), dice_1_acc / len(train_data), dice_2_acc / len(train_data), dice_3_acc / len(train_data),
                    valid_loss / len(valid_data),
                    val_acc_0 / len(valid_data), val_acc_1 / len(valid_data), val_acc_2 / len(valid_data), val_acc_3 / len(valid_data),
                    len(valid_data)))
            print('index in valid-data, and the length of valid-data:', index)
            # print('dice list, and conf list:', dice_list, conf_list)
            sys.stdout.flush()
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str)
        '''
        保存最终的模型：
            torch.save(net.state_dict(), os.path.join(save, 'model_half.dat'))
        '''
        # Determine if model is the best
        if valid_data:
            if val_acc_0 / len(valid_data) > best_acc:
                best_acc = val_acc_0 / len(valid_data)
                # 打印现在最好的准确率
                print('New best acc: %.4f' % best_acc)
                torch.save(net.state_dict(), os.path.join(save, 'stage_one.dat'))
                # .pth 是pytorch种常用的一种保存文件格式的方式
                torch.save(net, os.path.join(save, 'stage_one.pth'))
        else:
            torch.save(net.state_dict(), os.path.join(save, 'no_valid_data.dat'))









def test(net, test_data):
    if torch.cuda.is_available():
        net = net.cuda()
        print("使用了 cuda")
    else:
        print("没使用cuda")

    with torch.no_grad():
        net = net.eval()
        for im, name, new_size, tran, content in test_data:
            im = im.permute(1, 0, 2, 3)
            im = im.cuda()
            uout, uout_1, uout_2, uout_3 = net(im)



            map_stack = uout.cpu().numpy()
            map_stack = map_stack.squeeze(1)
            map_stack = zoom(map_stack, (1, 1/new_size[0][1], 1/new_size[0][2]))
            map_stack = map_stack.transpose(tran[0])
            save_map = sitk.GetImageFromArray(map_stack)
            save_map.SetOrigin(content[0][0])
            save_map.SetSpacing(content[0][1])
            save_map.SetDirection(content[0][2])
            sitk.WriteImage(save_map, str(name[0]).split(".nii.gz")[0] + "_map.nii.gz")


            map_stack[map_stack<0.5] = 0
            map_stack[map_stack!=0] = 1
            save_mask = sitk.GetImageFromArray(map_stack)
            save_mask.SetOrigin(content[0][0])
            save_mask.SetSpacing(content[0][1])
            save_mask.SetDirection(content[0][2])
            sitk.WriteImage(save_mask, str(name[0]).split(".nii.gz")[0] + "_mask.nii.gz")
