#!/usr/bin/env python3
# encoding: utf-8



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



'''
    Basic Block     
'''
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups=8):
        super(BasicBlock, self).__init__()
        self.gn1 = nn.GroupNorm(n_groups, in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        self.gn2 = nn.GroupNorm(n_groups, in_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x):
        residul = x
        x = self.relu1(self.gn1(x))
        x = self.conv1(x)

        x = self.relu2(self.gn2(x))
        x = self.conv2(x)
        x = x + residul

        return x


class Confidence(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups=8):
        super(Confidence, self).__init__()
        self.gn1 = nn.GroupNorm(n_groups, in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        self.FL = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            # torch.nn.Dropout(0.5),

            torch.nn.Linear(256, 1),
        )
    def forward(self, x):
        x = self.relu1(self.gn1(x))
        x = self.conv1(x)
        x = x.view(x.shape[0], -1)
        x = self.FL(x)



        return x


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    # x_0 guide x_1
    def forward(self, x_0, x_1):     # x_0: good; x_1: bad;     [2/1, 512, 8, 8]
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        # from x get the x0 and x1, x0 is the 0 sign with low entropy, x1 is the 1 sign with high entropy.
        # print("x.shape, zero_low.shape:", x.shape, torch.tensor(zero_low).cuda(async=True).shape)
        # x_0 = torch.index_select(x, 0, torch.tensor(zero_low).cuda(async=True))     # good cases
        # x_1 = torch.index_select(x, 0, torch.tensor(one_high).cuda(async=True))     # error cases
        batch_0, C_0, width_0, height_0 = x_0.size()
        batch_1, C_1, width_1, height_1 = x_1.size()
        # print("x_0.shape, x_1.shape:", x_0.shape, x_1.shape)
        proj_query = self.query_conv(x_0).view(batch_0, -1).permute(1, 0)     # good cases---> N_number*(512*8*8)---> (512*8*8)*N_number

        # print("x_0.shape:", x_0.shape)
        # print("proj_query.shape:", proj_query.shape)
        proj_key = self.key_conv(x_1).view(batch_1, -1)  # error cases---> K_number*(512*8*8)
        # print("x_1.shape:", x_1.shape)
        # print("proj_key.shape:", proj_key.shape)
        energy = torch.mm(proj_key, proj_query)  # transpose check, good cases wise dot the error cases, so should be N*K / as, (K_number*(512*8*8)) * ((512*8*8)*N_number) == K_number * N_number
        attention = self.softmax(energy)  # the shape are K_number * N_number
        proj_value = self.value_conv(x_0).view(batch_0, -1)  # good cases, the two conv process, output a N_number*(512*8*8)
        # print("proj_value.shape:", proj_value.shape)

        out = torch.mm(attention, proj_value)     # (K_number * N_number) * (N_number*(512*8*8)) output a tensor, the shape is K_number*(512*8*8)
        out = out.view(batch_1, C_1, width_1, height_1)     # output the shape is (K_number * 512 * 8 * 8)

        out = self.gamma * out + x_1     # (K_number * 512 * 8 * 8) attention + x_1 (error cases)
        return out, attention


class Self_multy_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_multy_Attn, self).__init__()
        self.chanel_in = in_dim
        # 下面的三者应该为固定的；
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)

        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.value_conv_out = nn.Conv2d(in_channels=in_dim // 4, out_channels=in_dim, kernel_size=1)
        self.gamma_a = nn.Parameter(torch.zeros(1))
        self.gamma_b = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    # x_0 guide x_1
    def forward(self, x_0, x_1):     # x_0: good; x_1: bad;     [2/1, 512, 8, 8]
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        # from x get the x0 and x1, x0 is the 0 sign with low entropy, x1 is the 1 sign with high entropy.
        # print("x.shape, zero_low.shape:", x.shape, torch.tensor(zero_low).cuda(async=True).shape)
        # x_0 = torch.index_select(x, 0, torch.tensor(zero_low).cuda(async=True))     # good cases
        # x_1 = torch.index_select(x, 0, torch.tensor(one_high).cuda(async=True))     # error cases
        batch_0, C_0, width_0, height_0 = x_0.size()
        # x_0_a, x_0_b = x_0[0], x_0[1]
        # print("x_0_a.shape:", x_0_a.size())
        batch_1, C_1, width_1, height_1 = x_1.size()
        # x_1_c = x_1[0]
        # print("x_1_c.size():", x_1_c.size())
        # print("x_0.shape, x_1.shape:", x_0.shape, x_1.shape)
        proj_query = self.query_conv(x_0)
        a_proj_q, b_proj_q = proj_query[0].view(proj_query.size()[1], -1).permute(1, 0), proj_query[1].view(proj_query.size()[1], -1).permute(1, 0)     # 这里应该先用torch.sequence(0).view(C_0, -1).permute(1, 0) -> [64, 512]
        proj_key = self.key_conv(x_1)

        c_proj_k = proj_key[0].view(proj_query.size()[1], -1)  # 同样要先用torch.sequence(0).view(C_0, -1).permute(1, 0) -> [64, 512]
        # print("c_proj_k.shape:", c_proj_k.size())

        energy_a, energy_b = torch.mm(c_proj_k, a_proj_q).unsqueeze(0), torch.mm(c_proj_k, b_proj_q).unsqueeze(0)  # 这里应该与每一个计算一次，所以得到两个[512, 512]
        energy = torch.cat((energy_a, energy_b), 0)
        # print("energy.shape:", energy.size())

        attention = self.softmax(energy)  # 这里对两个[512, 512]进行计算，即对上面的两个[512, 512]先合起来，为[2, 512, 512]，这样的话就可以softmax(new_energy, dim=0)
        atten_a, atten_b = attention[0], attention[1]


        a_proj_v, b_proj_v = self.value_conv(x_0)[0].view(proj_query.size()[1], -1), self.value_conv(x_0)[1].view(proj_query.size()[1], -1)  # 这里得到新的new x_0，大小为[512, 64]
        out_a = torch.mm(atten_a, a_proj_v).unsqueeze(0).view(batch_1, proj_query.size()[1], width_1, height_1)
        out_b = torch.mm(atten_b, b_proj_v).unsqueeze(0).view(batch_1, proj_query.size()[1], width_1, height_1)     # 这里得到了[2, 512, 64]
        out_a, out_b = self.value_conv_out(out_a), self.value_conv_out(out_b)

        out = self.gamma_a * out_a + self.gamma_b * out_b + x_1     # (K_number * 512 * 8 * 8) attention + x_1 (error cases)
        return out, attention



class Multi_cat(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Multi_cat, self).__init__()
    # x_0 guide x_1
    def forward(self, c5d, entropy_list):     # x_0: good; x_1: bad;     [2/1, 512, 8, 8]
        new_tensor_attentioned = []
        for i in range(c5d.shape[0]):
            if i == 0:
                if entropy_list[i] >= entropy_list[i + 1]:
                    # a = torch.unsqueeze(x, dim=0)
                    tem_tensor = torch.cat([torch.unsqueeze(c5d[i + 1], dim=0), torch.unsqueeze(c5d[i + 1], dim=0)])
                    # print("tem_tensor.shape, c5d[i].shape:", tem_tensor.shape, c5d[i].shape)
                    out_atte, attention = self.attention(tem_tensor, torch.unsqueeze(c5d[i], dim=0))
                    new_tensor_attentioned.append(out_atte)
                else:
                    tem_tensor = torch.cat([torch.unsqueeze(c5d[i], dim=0), torch.unsqueeze(c5d[i], dim=0)])
                    # print("tem_tensor.shape, c5d[i].shape:", tem_tensor.shape, c5d[i].shape)
                    out_atte, attention = self.attention(tem_tensor, torch.unsqueeze(c5d[i], dim=0))
                    new_tensor_attentioned.append(out_atte)
            elif i == c5d.shape[0] - 1:
                if entropy_list[i] >= entropy_list[i - 1]:
                    tem_tensor = torch.cat([torch.unsqueeze(c5d[i - 1], dim=0), torch.unsqueeze(c5d[i - 1], dim=0)])
                    out_atte, attention = self.attention(tem_tensor, torch.unsqueeze(c5d[i], dim=0))
                    new_tensor_attentioned.append(out_atte)
                else:
                    tem_tensor = torch.cat([torch.unsqueeze(c5d[i], dim=0), torch.unsqueeze(c5d[i], dim=0)])
                    out_atte, attention = self.attention(tem_tensor, torch.unsqueeze(c5d[i], dim=0))
                    new_tensor_attentioned.append(out_atte)

            else:
                if entropy_list[i] >= entropy_list[i + 1] and entropy_list[i] >= entropy_list[i - 1]:
                    tem_tensor = torch.cat([torch.unsqueeze(c5d[i - 1], dim=0), torch.unsqueeze(c5d[i + 1], dim=0)])
                    out_atte, attention = self.attention(tem_tensor, torch.unsqueeze(c5d[i], dim=0))
                    new_tensor_attentioned.append(out_atte)
                elif entropy_list[i] <= entropy_list[i + 1] and entropy_list[i] >= entropy_list[i - 1]:

                    tem_tensor = torch.cat([torch.unsqueeze(c5d[i - 1], dim=0), torch.unsqueeze(c5d[i], dim=0)])
                    out_atte, attention = self.attention(tem_tensor, torch.unsqueeze(c5d[i], dim=0))
                    new_tensor_attentioned.append(out_atte)

                elif entropy_list[i] >= entropy_list[i + 1] and entropy_list[i] <= entropy_list[i - 1]:

                    tem_tensor = torch.cat([torch.unsqueeze(c5d[i], dim=0), torch.unsqueeze(c5d[i + 1], dim=0)])
                    out_atte, attention = self.attention(tem_tensor, torch.unsqueeze(c5d[i], dim=0))
                    new_tensor_attentioned.append(out_atte)

                elif entropy_list[i] <= entropy_list[i + 1] and entropy_list[i] <= entropy_list[i - 1]:
                    tem_tensor = torch.cat([torch.unsqueeze(c5d[i], dim=0), torch.unsqueeze(c5d[i], dim=0)])
                    out_atte, attention = self.attention(tem_tensor, torch.unsqueeze(c5d[i], dim=0))
                    new_tensor_attentioned.append(out_atte)
                else:
                    print("???????????????????????", entropy_list[i - 1], entropy_list[i], entropy_list[i + 1])

        # print("type(new_tensor_attentioned):", type(new_tensor_attentioned))
        tensor_attentioned = torch.stack(new_tensor_attentioned, dim=0)
        tensor_attentioned = torch.squeeze(tensor_attentioned)
        # print("tensor_attentioned.shape:", tensor_attentioned.shape, type(tensor_attentioned))

        return tensor_attentioned







class UNet2D(nn.Module):
    """
    2d unet
    Ref:
        3D MRI brain tumor segmentation.
    Args:
        input_shape: tuple, (height, width, depth)
    """

    def __init__(self, in_channels=1, out_channels=1, init_channels=16, p=0.2):
        super(UNet2D, self).__init__()


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_channels = init_channels
        self.make_encoder()
        self.make_decoder()


    def make_encoder(self):
        init_channels = self.init_channels

        #self.avgpool = nn.AvgPool2d((2, 2), stride=(2, 2))
        # self.pool_1 = nn.functional.interpolate(scale_factor=0.5, mode="bilinear")
        # self.pool_2 = nn.functional.interpolate(scale_factor=0.25, mode="bilinear")
        # self.pool_3 = nn.functional.interpolate(scale_factor=0.125, mode="bilinear")

        self.conv1a = nn.Conv2d(self.in_channels, init_channels, 3, padding=1)
        self.conv1b = BasicBlock(init_channels, init_channels)  # 32

        self.ds1 = nn.Conv2d(init_channels, init_channels * 2, 3, stride=2, padding=1)  # down sampling and add channels

        self.conv2a = BasicBlock(init_channels * 2, init_channels * 2)
        self.conv2b = BasicBlock(init_channels * 2, init_channels * 2)

        self.ds2 = nn.Conv2d(init_channels * 2, init_channels * 4, 3, stride=2, padding=1)

        self.conv3a = BasicBlock(init_channels * 4, init_channels * 4)
        self.conv3b = BasicBlock(init_channels * 4, init_channels * 4)

        self.ds3 = nn.Conv2d(init_channels * 4, init_channels * 8, 3, stride=2, padding=1)

        self.conv4a = BasicBlock(init_channels * 8, init_channels * 8)
        self.conv4b = BasicBlock(init_channels * 8, init_channels * 8)

        self.ds4 = nn.Conv2d(init_channels * 8, init_channels * 16, 3, stride=2, padding=1)

        self.conv5a = BasicBlock(init_channels * 16, init_channels * 16)
        self.conv5b = BasicBlock(init_channels * 16, init_channels * 16)

        self.conf = Confidence(init_channels * 16, 1)

        # self.attention = Self_Attn(init_channels * 16)
        self.attention1 = Self_multy_Attn(init_channels * 1)
        self.attention2 = Self_multy_Attn(init_channels * 2)
        self.attention3 = Self_multy_Attn(init_channels * 4)
        self.attention4 = Self_multy_Attn(init_channels * 8)
        self.attention5 = Self_multy_Attn(init_channels * 16)
        # self.multicat = Multi_cat()
    def make_decoder(self):
        init_channels = self.init_channels
        self.up5conva = nn.Conv2d(init_channels * 16, init_channels * 8, 1)
        self.up5 = nn.Upsample(scale_factor=2)  # mode='bilinear'
        self.up5convb = BasicBlock(init_channels * 8, init_channels * 8)

        self.up4conva = nn.Conv2d(init_channels * 8, init_channels * 4, 1)
        self.up4 = nn.Upsample(scale_factor=2)  # mode='bilinear'
        self.up4convb = BasicBlock(init_channels * 4, init_channels * 4)

        self.up3conva = nn.Conv2d(init_channels * 4, init_channels * 2, 1)
        self.up3 = nn.Upsample(scale_factor=2)
        self.up3convb = BasicBlock(init_channels * 2, init_channels * 2)

        self.up2conva = nn.Conv2d(init_channels * 2, init_channels, 1)
        self.up2 = nn.Upsample(scale_factor=2)
        self.up2convb = BasicBlock(init_channels, init_channels)

        self.up1conv = nn.Conv2d(init_channels, self.out_channels, 1)


    def forward(self, x, sign_list, entropy_list):
        # #print("input - x :", x.shape)
        # x_1 = nn.functional.interpolate(x, scale_factor=0.5, mode="bilinear")
        # x_2 = nn.functional.interpolate(x, scale_factor=0.25, mode="bilinear")
        # x_3 = nn.functional.interpolate(x, scale_factor=0.125, mode="bilinear")
        c1 = self.conv1a(x)
        c1 = self.conv1b(c1)
        # print("c1.size():", c1.size())
        c1d = self.ds1(c1)
        c2 = self.conv2a(c1d)
        c2 = self.conv2b(c2)
        # print("c2.size():", c2.size())
        c2d = self.ds2(c2)
        c3 = self.conv3a(c2d)
        c3 = self.conv3b(c3)
        # print("c3.size():", c3.size())
        c3d = self.ds3(c3)
        c4 = self.conv4a(c3d)
        c4 = self.conv4b(c4)

        # print("c4.size():", c4.size())
        c4d = self.ds4(c4)
        c5 = self.conv5a(c4d)
        c5d = self.conv5b(c5)
        # print("c5d.shape:", c5d.shape)
        # print("encode_feture.shape, sign_list.shape, sign_list:", c5d.shape, sign_list.shape, sign_list)
        '''
        recuurent self attention
        for i in sequence:
            if entropy[i] < entropy[i+1]:
                self.attention(i, i+1 ,i+1)
            if entropy[i] < entropy[i+1] and entropy[i] < entropy[i-1]:
                self.attention(i, i+1, i-1)
            ... ... 
        '''
        new_tensor_attentioned_5, new_tensor_attentioned_4, new_tensor_attentioned_3, new_tensor_attentioned_2, new_tensor_attentioned_1 = [], [], [], [], []
        for i in range(c5d.shape[0]):
            if i == 0:
                if entropy_list[i] >= entropy_list[i + 1]:
                    # a = torch.unsqueeze(x, dim=0)
                    tem_tensor = torch.cat([torch.unsqueeze(c5d[i + 1], dim=0), torch.unsqueeze(c5d[i + 1], dim=0)])
                    tem_tensor_4 = torch.cat([torch.unsqueeze(c4[i + 1], dim=0), torch.unsqueeze(c4[i + 1], dim=0)])
                    tem_tensor_3 = torch.cat([torch.unsqueeze(c3[i + 1], dim=0), torch.unsqueeze(c3[i + 1], dim=0)])
                    tem_tensor_2 = torch.cat([torch.unsqueeze(c2[i + 1], dim=0), torch.unsqueeze(c2[i + 1], dim=0)])
                    tem_tensor_1 = torch.cat([torch.unsqueeze(c1[i + 1], dim=0), torch.unsqueeze(c1[i + 1], dim=0)])

                    # print("tem_tensor.shape, c5d[i].shape:", tem_tensor.shape, c5d[i].shape)
                    out_atte_5, attention_5 = self.attention5(tem_tensor, torch.unsqueeze(c5d[i], dim=0))
                    new_tensor_attentioned_5.append(out_atte_5)
                    out_atte_4, attention_4 = self.attention4(tem_tensor_4, torch.unsqueeze(c4[i], dim=0))
                    new_tensor_attentioned_4.append(out_atte_4)
                    out_atte_3, attention_3 = self.attention3(tem_tensor_3, torch.unsqueeze(c3[i], dim=0))
                    new_tensor_attentioned_3.append(out_atte_3)
                    out_atte_2, attention_2 = self.attention2(tem_tensor_2, torch.unsqueeze(c2[i], dim=0))
                    new_tensor_attentioned_2.append(out_atte_2)
                    out_atte_1, attention_1 = self.attention1(tem_tensor_1, torch.unsqueeze(c1[i], dim=0))
                    new_tensor_attentioned_1.append(out_atte_1)
                else:
                    tem_tensor = torch.cat([torch.unsqueeze(c5d[i], dim=0), torch.unsqueeze(c5d[i], dim=0)])
                    tem_tensor_4 = torch.cat([torch.unsqueeze(c4[i], dim=0), torch.unsqueeze(c4[i], dim=0)])
                    tem_tensor_3 = torch.cat([torch.unsqueeze(c3[i], dim=0), torch.unsqueeze(c3[i], dim=0)])
                    tem_tensor_2 = torch.cat([torch.unsqueeze(c2[i], dim=0), torch.unsqueeze(c2[i], dim=0)])
                    tem_tensor_1 = torch.cat([torch.unsqueeze(c1[i], dim=0), torch.unsqueeze(c1[i], dim=0)])

                    # print("tem_tensor.shape, c5d[i].shape:", tem_tensor.shape, c5d[i].shape)
                    out_atte_5, attention_5 = self.attention5(tem_tensor, torch.unsqueeze(c5d[i], dim=0))
                    new_tensor_attentioned_5.append(out_atte_5)
                    out_atte_4, attention_4 = self.attention4(tem_tensor_4, torch.unsqueeze(c4[i], dim=0))
                    new_tensor_attentioned_4.append(out_atte_4)
                    out_atte_3, attention_3 = self.attention3(tem_tensor_3, torch.unsqueeze(c3[i], dim=0))
                    new_tensor_attentioned_3.append(out_atte_3)
                    out_atte_2, attention_2 = self.attention2(tem_tensor_2, torch.unsqueeze(c2[i], dim=0))
                    new_tensor_attentioned_2.append(out_atte_2)
                    out_atte_1, attention_1 = self.attention1(tem_tensor_1, torch.unsqueeze(c1[i], dim=0))
                    new_tensor_attentioned_1.append(out_atte_1)
            elif i == c5d.shape[0] - 1:
                if entropy_list[i] >= entropy_list[i - 1]:
                    tem_tensor = torch.cat([torch.unsqueeze(c5d[i - 1], dim=0), torch.unsqueeze(c5d[i - 1], dim=0)])
                    tem_tensor_4 = torch.cat([torch.unsqueeze(c4[i - 1], dim=0), torch.unsqueeze(c4[i - 1], dim=0)])
                    tem_tensor_3 = torch.cat([torch.unsqueeze(c3[i - 1], dim=0), torch.unsqueeze(c3[i - 1], dim=0)])
                    tem_tensor_2 = torch.cat([torch.unsqueeze(c2[i - 1], dim=0), torch.unsqueeze(c2[i - 1], dim=0)])
                    tem_tensor_1 = torch.cat([torch.unsqueeze(c1[i - 1], dim=0), torch.unsqueeze(c1[i - 1], dim=0)])

                    out_atte_5, attention_5 = self.attention5(tem_tensor, torch.unsqueeze(c5d[i], dim=0))
                    new_tensor_attentioned_5.append(out_atte_5)
                    out_atte_4, attention_4 = self.attention4(tem_tensor_4, torch.unsqueeze(c4[i], dim=0))
                    new_tensor_attentioned_4.append(out_atte_4)
                    out_atte_3, attention_3 = self.attention3(tem_tensor_3, torch.unsqueeze(c3[i], dim=0))
                    new_tensor_attentioned_3.append(out_atte_3)
                    out_atte_2, attention_2 = self.attention2(tem_tensor_2, torch.unsqueeze(c2[i], dim=0))
                    new_tensor_attentioned_2.append(out_atte_2)
                    out_atte_1, attention_1 = self.attention1(tem_tensor_1, torch.unsqueeze(c1[i], dim=0))
                    new_tensor_attentioned_1.append(out_atte_1)
                else:
                    tem_tensor = torch.cat([torch.unsqueeze(c5d[i], dim=0), torch.unsqueeze(c5d[i], dim=0)])
                    tem_tensor_4 = torch.cat([torch.unsqueeze(c4[i], dim=0), torch.unsqueeze(c4[i], dim=0)])
                    tem_tensor_3 = torch.cat([torch.unsqueeze(c3[i], dim=0), torch.unsqueeze(c3[i], dim=0)])
                    tem_tensor_2 = torch.cat([torch.unsqueeze(c2[i], dim=0), torch.unsqueeze(c2[i], dim=0)])
                    tem_tensor_1 = torch.cat([torch.unsqueeze(c1[i], dim=0), torch.unsqueeze(c1[i], dim=0)])

                    out_atte_5, attention_5 = self.attention5(tem_tensor, torch.unsqueeze(c5d[i], dim=0))
                    new_tensor_attentioned_5.append(out_atte_5)
                    out_atte_4, attention_4 = self.attention4(tem_tensor_4, torch.unsqueeze(c4[i], dim=0))
                    new_tensor_attentioned_4.append(out_atte_4)
                    out_atte_3, attention_3 = self.attention3(tem_tensor_3, torch.unsqueeze(c3[i], dim=0))
                    new_tensor_attentioned_3.append(out_atte_3)
                    out_atte_2, attention_2 = self.attention2(tem_tensor_2, torch.unsqueeze(c2[i], dim=0))
                    new_tensor_attentioned_2.append(out_atte_2)
                    out_atte_1, attention_1 = self.attention1(tem_tensor_1, torch.unsqueeze(c1[i], dim=0))
                    new_tensor_attentioned_1.append(out_atte_1)

            else:
                if entropy_list[i] >= entropy_list[i + 1] and entropy_list[i] >= entropy_list[i - 1]:
                    tem_tensor = torch.cat([torch.unsqueeze(c5d[i - 1], dim=0), torch.unsqueeze(c5d[i + 1], dim=0)])
                    tem_tensor_4 = torch.cat([torch.unsqueeze(c4[i - 1], dim=0), torch.unsqueeze(c4[i + 1], dim=0)])
                    tem_tensor_3 = torch.cat([torch.unsqueeze(c3[i - 1], dim=0), torch.unsqueeze(c3[i + 1], dim=0)])
                    tem_tensor_2 = torch.cat([torch.unsqueeze(c2[i - 1], dim=0), torch.unsqueeze(c2[i + 1], dim=0)])
                    tem_tensor_1 = torch.cat([torch.unsqueeze(c1[i - 1], dim=0), torch.unsqueeze(c1[i + 1], dim=0)])

                    out_atte_5, attention_5 = self.attention5(tem_tensor, torch.unsqueeze(c5d[i], dim=0))
                    new_tensor_attentioned_5.append(out_atte_5)
                    out_atte_4, attention_4 = self.attention4(tem_tensor_4, torch.unsqueeze(c4[i], dim=0))
                    new_tensor_attentioned_4.append(out_atte_4)
                    out_atte_3, attention_3 = self.attention3(tem_tensor_3, torch.unsqueeze(c3[i], dim=0))
                    new_tensor_attentioned_3.append(out_atte_3)
                    out_atte_2, attention_2 = self.attention2(tem_tensor_2, torch.unsqueeze(c2[i], dim=0))
                    new_tensor_attentioned_2.append(out_atte_2)
                    out_atte_1, attention_1 = self.attention1(tem_tensor_1, torch.unsqueeze(c1[i], dim=0))
                    new_tensor_attentioned_1.append(out_atte_1)
                elif entropy_list[i] <= entropy_list[i + 1] and entropy_list[i] >= entropy_list[i - 1]:

                    tem_tensor = torch.cat([torch.unsqueeze(c5d[i - 1], dim=0), torch.unsqueeze(c5d[i], dim=0)])
                    tem_tensor_4 = torch.cat([torch.unsqueeze(c4[i - 1], dim=0), torch.unsqueeze(c4[i], dim=0)])
                    tem_tensor_3 = torch.cat([torch.unsqueeze(c3[i - 1], dim=0), torch.unsqueeze(c3[i], dim=0)])
                    tem_tensor_2 = torch.cat([torch.unsqueeze(c2[i - 1], dim=0), torch.unsqueeze(c2[i], dim=0)])
                    tem_tensor_1 = torch.cat([torch.unsqueeze(c1[i - 1], dim=0), torch.unsqueeze(c1[i], dim=0)])

                    out_atte_5, attention_5 = self.attention5(tem_tensor, torch.unsqueeze(c5d[i], dim=0))
                    new_tensor_attentioned_5.append(out_atte_5)
                    out_atte_4, attention_4 = self.attention4(tem_tensor_4, torch.unsqueeze(c4[i], dim=0))
                    new_tensor_attentioned_4.append(out_atte_4)
                    out_atte_3, attention_3 = self.attention3(tem_tensor_3, torch.unsqueeze(c3[i], dim=0))
                    new_tensor_attentioned_3.append(out_atte_3)
                    out_atte_2, attention_2 = self.attention2(tem_tensor_2, torch.unsqueeze(c2[i], dim=0))
                    new_tensor_attentioned_2.append(out_atte_2)
                    out_atte_1, attention_1 = self.attention1(tem_tensor_1, torch.unsqueeze(c1[i], dim=0))
                    new_tensor_attentioned_1.append(out_atte_1)

                elif entropy_list[i] >= entropy_list[i + 1] and entropy_list[i] <= entropy_list[i - 1]:

                    tem_tensor = torch.cat([torch.unsqueeze(c5d[i], dim=0), torch.unsqueeze(c5d[i + 1], dim=0)])
                    tem_tensor_4 = torch.cat([torch.unsqueeze(c4[i], dim=0), torch.unsqueeze(c4[i + 1], dim=0)])
                    tem_tensor_3 = torch.cat([torch.unsqueeze(c3[i], dim=0), torch.unsqueeze(c3[i + 1], dim=0)])
                    tem_tensor_2 = torch.cat([torch.unsqueeze(c2[i], dim=0), torch.unsqueeze(c2[i + 1], dim=0)])
                    tem_tensor_1 = torch.cat([torch.unsqueeze(c1[i], dim=0), torch.unsqueeze(c1[i + 1], dim=0)])

                    out_atte_5, attention_5 = self.attention5(tem_tensor, torch.unsqueeze(c5d[i], dim=0))
                    new_tensor_attentioned_5.append(out_atte_5)
                    out_atte_4, attention_4 = self.attention4(tem_tensor_4, torch.unsqueeze(c4[i], dim=0))
                    new_tensor_attentioned_4.append(out_atte_4)
                    out_atte_3, attention_3 = self.attention3(tem_tensor_3, torch.unsqueeze(c3[i], dim=0))
                    new_tensor_attentioned_3.append(out_atte_3)
                    out_atte_2, attention_2 = self.attention2(tem_tensor_2, torch.unsqueeze(c2[i], dim=0))
                    new_tensor_attentioned_2.append(out_atte_2)
                    out_atte_1, attention_1 = self.attention1(tem_tensor_1, torch.unsqueeze(c1[i], dim=0))
                    new_tensor_attentioned_1.append(out_atte_1)

                elif entropy_list[i] <= entropy_list[i + 1] and entropy_list[i] <= entropy_list[i - 1]:
                    tem_tensor = torch.cat([torch.unsqueeze(c5d[i], dim=0), torch.unsqueeze(c5d[i], dim=0)])
                    tem_tensor_4 = torch.cat([torch.unsqueeze(c4[i], dim=0), torch.unsqueeze(c4[i], dim=0)])
                    tem_tensor_3 = torch.cat([torch.unsqueeze(c3[i], dim=0), torch.unsqueeze(c3[i], dim=0)])
                    tem_tensor_2 = torch.cat([torch.unsqueeze(c2[i], dim=0), torch.unsqueeze(c2[i], dim=0)])
                    tem_tensor_1 = torch.cat([torch.unsqueeze(c1[i], dim=0), torch.unsqueeze(c1[i], dim=0)])

                    out_atte_5, attention_5 = self.attention5(tem_tensor, torch.unsqueeze(c5d[i], dim=0))
                    new_tensor_attentioned_5.append(out_atte_5)
                    out_atte_4, attention_4 = self.attention4(tem_tensor_4, torch.unsqueeze(c4[i], dim=0))
                    new_tensor_attentioned_4.append(out_atte_4)
                    out_atte_3, attention_3 = self.attention3(tem_tensor_3, torch.unsqueeze(c3[i], dim=0))
                    new_tensor_attentioned_3.append(out_atte_3)
                    out_atte_2, attention_2 = self.attention2(tem_tensor_2, torch.unsqueeze(c2[i], dim=0))
                    new_tensor_attentioned_2.append(out_atte_2)
                    out_atte_1, attention_1 = self.attention1(tem_tensor_1, torch.unsqueeze(c1[i], dim=0))
                    new_tensor_attentioned_1.append(out_atte_1)
                else:
                    print("???????????????????????", entropy_list[i - 1], entropy_list[i], entropy_list[i + 1])

        # print("type(new_tensor_attentioned):", type(new_tensor_attentioned))
        tensor_attentioned = torch.stack(new_tensor_attentioned_5, dim=0)
        tensor_attentioned = torch.squeeze(tensor_attentioned)
        tensor_attentioned_c4 = torch.stack(new_tensor_attentioned_4, dim=0)
        tensor_attentioned_c4 = torch.squeeze(tensor_attentioned_c4)
        tensor_attentioned_c3 = torch.stack(new_tensor_attentioned_3, dim=0)
        tensor_attentioned_c3 = torch.squeeze(tensor_attentioned_c3)
        tensor_attentioned_c2 = torch.stack(new_tensor_attentioned_2, dim=0)
        tensor_attentioned_c2 = torch.squeeze(tensor_attentioned_c2)
        tensor_attentioned_c1 = torch.stack(new_tensor_attentioned_1, dim=0)
        tensor_attentioned_c1 = torch.squeeze(tensor_attentioned_c1)

        # print("----------------------------------------------")
        # print("c5d.shape, len(entropy_list):", c5d.shape, len(entropy_list))


        #out_atte, attention = self.attention(c5d, entropy_list)
        # print("======================================================")
        '''
        The Decode
        '''
        one_high = []
        for i in range(len(sign_list)):
            if sign_list[i] == 1:
                one_high.append(i)

        zero_low = []
        for i in range(len(sign_list)):
            if sign_list[i] == 0:
                zero_low.append(i)

        u5 = self.up5conva(tensor_attentioned)
        u5 = self.up5(u5)

        
        u5 = u5 + tensor_attentioned_c4
        u5 = self.up5convb(u5)
        u4 = self.up4conva(u5)
        u4 = self.up4(u4)




        
        u4 = u4 + tensor_attentioned_c3
        u4 = self.up4convb(u4)
        u3 = self.up3conva(u4)
        u3 = self.up3(u3)




        u3 = u3 + tensor_attentioned_c2
        u3 = self.up3convb(u3)
        u2 = self.up2conva(u3)
        u2 = self.up2(u2)

        

        u2 = u2 + tensor_attentioned_c1
        u2 = self.up2convb(u2)
        uout = self.up1conv(u2)
        uout = torch.sigmoid(uout)
        uout_half = nn.functional.interpolate(uout, scale_factor=0.5, mode="bilinear")

        uout_high = torch.index_select(uout, 0, torch.tensor(one_high).cuda())
        uout_high_1 = torch.index_select(uout_half, 0, torch.tensor(one_high).cuda())

        uout_low = torch.index_select(uout, 0, torch.tensor(zero_low).cuda())
        uout_low_1 = torch.index_select(uout_half, 0, torch.tensor(zero_low).cuda())
        # 差的，差的一半；好的，好的一半；
        return uout_high, uout_high_1, uout_low, uout_low_1, one_high, zero_low