import sys
sys.path.append('..')

import numpy as np
import torch
from data_load_stage_one_test import custom_dataset
from torch.utils.data import DataLoader

from util_unet import test

def collate_fn(batch):
    img = [item[0] for item in batch]
    name = [item[1] for item in batch]
    new_size = [item[2] for item in batch]
    tran = [item[3] for item in batch]
    content = [item[4] for item in batch]
    img = torch.Tensor(img)

    return img, name, new_size, tran, content


from PIL import Image


# image, label, mask
def data_tf(img):
    Mean = img.mean()
    Std = img.std()
    img = (img - Mean) / Std
    return img


test_dataset = custom_dataset('G:/Code/Fetalbrain/coarse stage/Data_21-30GWs', transform=data_tf) # 读入 .pkl 文件
test_data = DataLoader(test_dataset, 1, shuffle=False, collate_fn=collate_fn) # batch size 设置为 8


net = torch.load('G:/Code/Fetalbrain/coarse stage/UNet/save/model_coarse_1.pth')

test(net, test_data)