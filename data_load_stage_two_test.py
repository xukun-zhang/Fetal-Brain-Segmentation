import os
from torch.utils.data import Dataset
from scipy.ndimage import zoom
import numpy as np
from skimage.measure import label as la
import SimpleITK as sitk
"""
    定义一个 读取数据集.pkl文件的类：
"""


# 定义一个子类叫 custom_dataset，继承与 Dataset
class custom_dataset(Dataset):
    def __init__(self, path, mask_path, transform=None):
        self.transform = transform  # 传入数据预处理
        self.image_data = {}     # length is the number of cases, and each case have a list, that have a sequence data
        self.map_data = {}
        self.entropy_point = {}     # length is the number of entropy point
        self.mask_data = {}
        self.image_name = []     # length is the number of cases
        self.center_point = []
        self.img_content = {}
        self.transpose = {}
        self.resize = {}
        forget_name = []

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
            point_list = []
            if image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]:
                tran = (0, 1, 2)
            if image.shape[1] < image.shape[0] and image.shape[1] < image.shape[2]:
                tran = (1, 0, 2)
            if image.shape[2] < image.shape[0] and image.shape[2] < image.shape[1]:
                tran = (2, 1, 0)


            image = image.transpose(tran)
            new_size = (1, 256/image.shape[1], 256/image.shape[2])
            image = zoom(image, new_size)
            Mean = image.mean()
            Std = image.std()
            image = (image - Mean) / Std

            tem = file_list[i].split("/")[-2] + "_" + file_list[i].split("/")[-1]

            map_tem = tem.split(".nii.gz")[0] + "_map.nii.gz"
            mask_tem = tem.split(".nii.gz")[0] + "_mask.nii.gz"
            map_sitk = sitk.ReadImage(mask_path + "/" + map_tem)  # Load data 加载当前读取的这个nii数据
            map = sitk.GetArrayFromImage(map_sitk)

            mask_sitk = sitk.ReadImage(mask_path + "/" + mask_tem)  # Load data 加载当前读取的这个nii数据
            mask = sitk.GetArrayFromImage(mask_sitk)

            map = map.transpose(tran)
            map = zoom(map, new_size)
            mask = mask.transpose(tran)
            mask = zoom(mask, new_size, order=0)

            loc_img, num = la(mask, background=0, return_num=True, connectivity=2)
            max_label = 0
            max_num = 0
            for i in range(1, num + 1):
                if np.sum(loc_img == i) > max_num:
                    max_num = np.sum(loc_img == i)
                    max_label = i
            mcr = (loc_img == max_label)
            mcr = mcr + 0
            z_true, y_true, x_true = np.where(mcr)
            box = np.array([[np.min(z_true), np.max(z_true)], [np.min(y_true), np.max(y_true)], [np.min(x_true), np.max(x_true)]])

            z_min, z_max = box[0]
            y_min, y_max = box[1]
            x_min, x_max = box[2]
            y_center = (y_min + y_max) // 2
            x_center = (x_min + x_max) // 2



            if y_center - 64 < 0 or y_center + 64 > 256 or x_center - 64 < 0 or x_center + 64 > 256:
                forget_name.append(tem)
                continue
            com_map = map + 0
            com_map[com_map < 0.5] = 0
            # print("com_map.shape:", com_map.shape, tem)
            for n in range(len(com_map)):
                point_number = 1
                for k in range(y_center - 64, y_center + 64):
                    for m in range(x_center - 64, x_center + 64):
                        if com_map[n][k][m] != 0:
                            point_number = point_number + 1
                point_list.append(point_number)
            # print(len(point_list), point_list)
            self.center_point.append((y_center, x_center))
            self.image_name.extend([tem])
            self.image_data[tem] = image
            self.map_data[tem] = map
            self.entropy_point[tem] = point_list
            self.mask_data[tem] = mask
            self.img_content[tem] = [origin, spacing, direction]
            self.resize[tem] = new_size
            self.transpose[tem] = tran
        print("forget_name:", len(forget_name), forget_name)





    def __getitem__(self, idx):  # 根据 idx 取出其中一个name
        name = self.image_name[idx]
        # print('name:', name)
        img = self.image_data[name]
        mask = self.mask_data[name]
        map = self.map_data[name]
        point_list = self.entropy_point[name]
        center = self.center_point[idx]
        content = self.img_content[name]
        trans = self.transpose[name]
        new_size = self.resize[name]
        if self.transform is not None:
            img, mask, map, point_list, center = self.transform(img, mask, map, point_list, center)

        return img, mask, map, point_list, center, name, content, new_size, trans

    def __len__(self):  # 总数据的多少
        return len(self.image_data)


# dataset = custom_dataset('G:/Code/Fetalbrain/coarse stage/Data_21-30GWs', "G:/Code/Fetalbrain/coarse stage/Data_21-30GWs_stageone") # 读入 .pkl 文件
# print("打印加载进入的数据集pkl_dataset：")
# print(dataset, len(dataset))
#
# # 取得其中一个数据检查一下
# for i in range(len(dataset)):
#    data, mask, map, polit_list, center, name= dataset[i]
#    #print("打印加载的数据集中的第一个数据，即pkl_dataset[0]的内容以及标签:")
#    print("内容、大小：", data.shape, mask.shape, map.shape, polit_list, center, name)
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
