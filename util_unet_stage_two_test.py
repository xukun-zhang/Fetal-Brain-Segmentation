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

def test(net, test_data):
    if torch.cuda.is_available():
        net = net.cuda()
        print("使用了 cuda")
    else:
        print("没使用cuda")

    with torch.no_grad():
        net = net.eval()
        for im, mask, sign_list, entropy_list, center_point, name, content, new_size, tran in test_data:

            im, mask, sign_list = im.cuda(), mask.cuda(), sign_list.cuda()

            one_high = []
            for i in range(len(sign_list)):
                if sign_list[i] == 1:
                    one_high.append(i)

            zero_low = []
            for i in range(len(sign_list)):
                if sign_list[i] == 0:
                    zero_low.append(i)

            uout, uout_1, uout_2, uout_3, high_list, low_list = net(im, sign_list, entropy_list)
            all_sort = []
            for i in range(len(high_list) + len(low_list)):
                if i in high_list:
                    index_i = high_list.index(i)
                    all_sort.append(uout[index_i])
                elif i in low_list:
                    index_i = low_list.index(i)
                    all_sort.append(uout_2[index_i])

            all_sort = torch.stack(all_sort, dim=0)
            print("all_sort.shape:", all_sort.shape)
            map_stack = all_sort.cpu().numpy()
            map_stack = map_stack.squeeze(1)




            mask_zero = np.zeros((map_stack.shape[0], 256, 256))
            print("center_point:", center_point, new_size, tran)
            mask_zero[:, center_point[0]-64 : center_point[0] + 64, center_point[1]-64:center_point[1]+64] = map_stack


            mask_zero = zoom(mask_zero, (1, 1/new_size[1], 1/new_size[2]))
            mask_zero[mask_zero > 0.5] = 1
            mask_zero[mask_zero != 1] = 0
            mask_zero = mask_zero.transpose(tran)

            save_mask = sitk.GetImageFromArray(mask_zero)
            save_mask.SetOrigin(content[0])
            save_mask.SetSpacing(content[1])
            save_mask.SetDirection(content[2])
            sitk.WriteImage(save_mask, str(name).split(".nii.gz")[0] + "_mask.nii.gz")
