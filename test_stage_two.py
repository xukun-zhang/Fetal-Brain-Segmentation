import sys
sys.path.append('..')
import numpy as np
import torch
from data_load_stage_two_test import custom_dataset
from torch.utils.data import DataLoader
from util_unet_stage_two_test import test

def entropy_fn(map, point_number):
    map[map < 0.5] = 0
    map = torch.tensor(map)
    entropy = ((-1) * map.contiguous().view(-1) * torch.log2(map.contiguous().view(-1) + 1e-7)).sum() / point_number
    return entropy.item()

# 重写collate_fn函数，其输入为一个batch的sample数据
def collate_fn(batch):

    img_sequence, mask_sequence, map_sequence, point_list, center_point, name, content = batch[0][0], batch[0][1], batch[0][2], batch[0][3], batch[0][4], batch[0][5], batch[0][6]
    new_size, trans = batch[0][7], batch[0][8]
    entropy_list = []
    sign_list = []
    for i in range(len(map_sequence)):
        entropy_list.append(entropy_fn(map_sequence[i], point_list[i]))
        sign_list.append(0)

    # print("entropy_list:", entropy_list)
    entropy_sort = sorted(enumerate(entropy_list), key=lambda x:x[1])
    for j in range(len(entropy_list)):
        if j > len(entropy_list)//2:
            sign_list[entropy_sort[j][0]] = 1
    for i in range(len(sign_list)):
        if sign_list[i] == 0:
            mask_tem = mask_sequence[i]     # mask_sequence can change
            mask_tem[mask_tem == 1] = 2
            mask_tem[mask_tem == 0] = 1
            # print("img_sequence[i].shape:", img_sequence[i].shape)
            img_sequence[i] = img_sequence[i] * mask_tem
            Mean = img_sequence[i].mean()
            Std = img_sequence[i].std()
            img_sequence[i] = (img_sequence[i] - Mean) / Std # 标准化，这个技巧之后会讲到
            # print("-----img_sequence[i].shape:", img_sequence[i].shape)
        else:
            img_sequence[i] = img_sequence[i] * 2
            Mean = img_sequence[i].mean()
            Std = img_sequence[i].std()
            img_sequence[i] = (img_sequence[i] - Mean) / Std  # 标准化，这个技巧之后会讲到
            # print("+++++img_sequence[i].shape:", img_sequence[i].shape)

    img_sequence = img_sequence[:, np.newaxis, :, :]
    mask_sequence = mask_sequence[:, np.newaxis, :, :]
    img_sequence = torch.Tensor(list(img_sequence))
    mask_sequence = torch.Tensor(list(mask_sequence))
    sign_list = torch.Tensor(list(sign_list))
    entropy_list = torch.Tensor(list(entropy_list))


    return img_sequence, mask_sequence, sign_list, entropy_list, center_point, name, content, new_size, trans

from PIL import Image
# image, label, mask
def data_tf(img, mask, map, point_list, center_point):
    img = np.array(img)
    mask = np.array(mask)
    y_center = center_point[0]
    x_center = center_point[1]
    img = img[:, y_center - 64:y_center + 64, x_center - 64:x_center + 64]
    mask = mask[:, y_center - 64:y_center + 64, x_center - 64:x_center + 64]
    map = map[:, y_center - 64:y_center + 64, x_center - 64:x_center + 64]
    img = np.array(img)
    mask = np.array(mask)

    return img, mask, map, point_list, center_point

test_dataset = custom_dataset('G:/Code/Fetalbrain/coarse stage/Data_21-30GWs', "G:/Code/Fetalbrain/coarse stage/Data_21-30GWs_stageone", transform=data_tf) # 读入 .pkl 文件
test_data = DataLoader(test_dataset, 1, shuffle=False, collate_fn=collate_fn) # batch size 设置为 8



net = torch.load('G:/Code/Fetalbrain/save/model_UNet_attention_weighted_bce_new20msclaelow.pth')


print("---- 开始训练：")
test(net, test_data)