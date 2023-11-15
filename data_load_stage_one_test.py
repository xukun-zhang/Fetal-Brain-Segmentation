import os
from torch.utils.data import Dataset
import numpy as np
from scipy.ndimage import zoom
import SimpleITK as sitk
"""
    定义一个 读取数据集.pkl文件的类：
"""


# 定义一个子类叫 custom_dataset，继承与 Dataset
class custom_dataset(Dataset):
    def __init__(self, path, transform=None):
        self.transform = transform  # 传入数据预处理
        self.image_data = {}     # length is the number of cases, and each case have a list, that have a sequence data
        self.image_name = []     # length is the number of cases
        self.image_content = {}

        self.transpose = {}
        self.resize = {}

        file_list = []
        for dir_path, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(dir_path, file)
                if "\\" in file_path:
                    file_path = file_path.replace('\\', '/')
                file_list.append(file_path)

        for i in range(len(file_list)):
            image_sitk = sitk.ReadImage(file_list[i])  # Load data 加载当前读取的这个nii数据
            image = sitk.GetArrayFromImage(image_sitk)
            origin = image_sitk.GetOrigin()
            spacing = image_sitk.GetSpacing()
            direction = image_sitk.GetDirection()

            if image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]:
                tran = (0, 1, 2)
            if image.shape[1] < image.shape[0] and image.shape[1] < image.shape[2]:
                tran = (1, 0, 2)
            if image.shape[2] < image.shape[0] and image.shape[2] < image.shape[1]:
                tran = (2, 1, 0)

            image = image.transpose(tran)
            new_size = (1, 256/image.shape[1], 256/image.shape[2])
            image = zoom(image, new_size)
            tem = file_list[i].split("/")[-2] + "_" + file_list[i].split("/")[-1]

            self.image_name.extend([tem])
            self.image_data[tem] = image
            self.resize[tem] = new_size
            self.transpose[tem] = tran
            self.image_content[tem] = [origin, spacing, direction]
        print("读取的数据数量为：", len(file_list))




    def __getitem__(self, idx):  # 根据 idx 取出其中一个name
        name = self.image_name[idx]
        img = self.image_data[name]
        new_size = self.resize[name]
        tran = self.transpose[name]
        content = self.image_content[name]

        if self.transform is not None:
            img = self.transform(img)
        return img, name, new_size, tran, content

    def __len__(self):  # 总数据的多少
        return len(self.image_data)







# dataset = custom_dataset('G:/Code/Fetalbrain/coarse stage/Data_21-30GWs') # 读入 .pkl 文件
# print(dataset, len(dataset))
# # 取得其中一个数据检查一下
# for i in range(len(dataset)):
#    data, name, new_size, tran, content = dataset[i]
#    print("内容、大小：", data.shape, name, new_size, tran, content)

#print(data)
# data = np.squeeze(data)
# print(np.shape(data))
# cv2.imwrite('data.png', data)
# print("标签：")
# #print(mask)
# mask = np.squeeze(mask)
# mask = (mask*100).astype(np.uint8)
#
# # print
# for i in range(256):
#     for j in range(256):
#         if mask[i][j] != 0:
#             #print('---------------------------------------------------------------')
#             print(mask[i][j])
#             #print('---------------------------------------------------------------')
# print(np.shape(mask))
# cv2.imwrite('mask.png', mask)

# 取得其中一个数据检查一下
#data, mask = dataset[1]
#print("打印加载的数据集中的第一个数据，即pkl_dataset[0]的内容以及标签:")
#print("内容、大小：")
#print(data)
#print(np.shape(data))
#print("标签：")
#print(mask)
#print(np.shape(mask))
# print("***********************************************************************")
# pkl_dataset = custom_dataset('./test_data/test.pkl') # 读入 .pkl 文件
# print("打印加载进入的数据集pkl_dataset：")
# print(pkl_dataset)

# 取得其中一个数据检查一下
# data, label = pkl_dataset[1500]
# print("打印加载的数据集中的第一个数据，即pkl_dataset[0]的内容以及标签:")
# print("内容、大小：")
# print(data)
# print(np.shape(data))
# print("标签：")
# print(label)
