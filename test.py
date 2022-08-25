from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch
import math
import util.metrics as metrics
import time


class GoProData(Dataset):
    def __init__(self, blur_image_files, sharp_image_files):
        self.root = '../dataset/GOPRO Large'
        self.train_blur_list = open(blur_image_files, 'r').readlines()
        self.train_sharp_list = open(sharp_image_files, 'r').readlines()
        self.crop_size = 512  # 一般都是256

    def __getitem__(self, item):
        # 切片[0:-1]能去除末尾换行符000057.png\n
        blur_image = Image.open(os.path.join(self.root, self.train_blur_list[item][0:-1])).convert('RGB')
        sharp_image = Image.open(os.path.join(self.root, self.train_sharp_list[item][0:-1])).convert('RGB')
        # totensor = transforms.ToTensor()
        blur_tensor = transforms.ToTensor()(blur_image)
        sharp_tensor = transforms.ToTensor()(sharp_image)
        if self.crop_size:
            h = blur_tensor.size(1)
            w = blur_tensor.size(2)
            W = np.random.randint(0, w - self.crop_size - 1, 1)[0]
            H = np.random.randint(0, h - self.crop_size - 1, 1)[0]
            blur_tensor = blur_tensor[:, H:H + self.crop_size, W:W + self.crop_size]
            sharp_tensor = sharp_tensor[:, H:H + self.crop_size, W:W + self.crop_size]
        return blur_tensor, sharp_tensor

    def __len__(self):
        return len(self.train_blur_list)


class ResBlock(nn.Module):
    def __init__(self, channel, ksize=3):
        super(ResBlock, self).__init__()
        padding = ksize // 2
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=ksize, padding=padding)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=ksize, padding=padding)

    def forward(self, x):
        net = self.conv1(x)
        act_net = nn.functional.relu_(net)
        net = self.conv2(act_net)
        net = net + x
        return net


class Encoder(nn.Module):
    def __init__(self, inchannel=3, ksize=3):
        super(Encoder, self).__init__()
        padding = ksize // 2
        self.layer1 = nn.Conv2d(inchannel, 32, kernel_size=ksize, padding=padding)
        self.layer2 = ResBlock(32, ksize=ksize)
        self.layer3 = ResBlock(32, ksize=ksize)
        self.layer4 = ResBlock(32, ksize=ksize)
        self.layer4p = ResBlock(32, ksize=ksize)  # 为什么有p

        self.layer5 = nn.Conv2d(32, 64, kernel_size=ksize, padding=padding, stride=2)
        self.layer6 = ResBlock(64, ksize=ksize)
        self.layer7 = ResBlock(64, ksize=ksize)
        self.layer8 = ResBlock(64, ksize=ksize)
        self.layer8p = ResBlock(64, ksize=ksize)

        self.layer9 = nn.Conv2d(64, 128, kernel_size=ksize, padding=padding, stride=2)
        self.layer10 = ResBlock(128, ksize=ksize)
        self.layer11 = ResBlock(128, ksize=ksize)
        self.layer12 = ResBlock(128, ksize=ksize)
        self.layer12p = ResBlock(128, ksize=ksize)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        enc1_4 = self.layer4p(x)
        x = self.layer5(enc1_4)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        enc2_4 = self.layer8p(x)
        x = self.layer9(enc2_4)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer12p(x)

        return enc1_4, enc2_4, x


class Decoder(nn.Module):
    def __init__(self, outchannel=3, ksize=3):
        super(Decoder, self).__init__()
        self.layer13p = ResBlock(128, ksize=ksize)
        self.layer13 = ResBlock(128, ksize=ksize)
        self.layer14 = ResBlock(128, ksize=ksize)
        self.layer15 = ResBlock(128, ksize=ksize)
        self.layer16 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)

        self.layer17p = ResBlock(64, ksize=ksize)
        self.layer17 = ResBlock(64, ksize=ksize)
        self.layer18 = ResBlock(64, ksize=ksize)
        self.layer19 = ResBlock(64, ksize=ksize)
        self.layer20 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)

        self.layer21p = ResBlock(32, ksize=ksize)
        self.layer21 = ResBlock(32, ksize=ksize)
        self.layer22 = ResBlock(32, ksize=ksize)
        self.layer23 = ResBlock(32, ksize=ksize)
        self.layer24 = nn.Conv2d(32, outchannel, kernel_size=ksize, padding=ksize // 2)

    def forward(self, enc1_4, enc2_4, x):
        x = self.layer13p(x)
        x = self.layer13(x)
        x = self.layer14(x)
        x = self.layer15(x)
        x = self.layer16(x)

        x = self.layer17p(x + enc2_4)
        x = self.layer17(x)
        x = self.layer18(x)
        x = self.layer19(x)
        x = self.layer20(x)

        x = self.layer21p(x + enc1_4)
        x = self.layer21(x)
        x = self.layer22(x)
        x = self.layer23(x)
        x = self.layer24(x)
        return x


class Mode(nn.Module):
    def __init__(self):
        super(Mode, self).__init__()
        self.encode1 = Encoder(ksize=3)
        self.encode2 = Encoder(inchannel=6, ksize=3)

        self.decoder1 = Decoder(ksize=3)
        self.decoder2 = Decoder(ksize=3)

    def encode_decode_level(self, x, last_scale_out):
        enc1_4, enc2_4, feature2 = self.encode2(torch.cat([x, last_scale_out], dim=1))  # 拼接
        residual2 = self.decoder2(enc1_4, enc2_4, feature2)
        enc1_4, enc2_4, tmp = self.encode1(x + residual2)
        feature1 = tmp + feature2
        y = self.decoder1(enc1_4, enc2_4, feature1)
        return y

    def forward(self, x):
        output = []
        B3 = nn.functional.interpolate(x, scale_factor=1 / 4, mode='bilinear')
        I3 = self.encode_decode_level(B3, B3)
        output.append(I3)
        I3.detach()  # 切断梯度
        B2 = nn.functional.interpolate(x, scale_factor=1 / 2, mode='bilinear')
        I3 = nn.functional.interpolate(I3, scale_factor=2, mode='bilinear')
        I2 = self.encode_decode_level(B2, I3)
        output.append(I2)
        I2.detach()
        I2 = nn.functional.interpolate(I2, scale_factor=2, mode='bilinear')
        I1 = self.encode_decode_level(x, I2)
        output.append(I1)
        return output


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, pred, gt):
        loss = 0
        for item in pred:  # 遍历batch个图像
            [b, c, h, w] = item.shape
            gt_i = nn.functional.interpolate(gt, size=(h, w), mode='bilinear')
            loss = loss + nn.functional.mse_loss(item, gt_i)
        return loss


def weight_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, 0.5 * math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


train_blur_image_files = './datas/GoPro/train_blur_file.txt'
train_sharp_image_files = './datas/GoPro/train_sharp_file.txt'
test_blur_image_files = './datas/GoPro/test_blur_file.txt'
test_sharp_image_files = './datas/GoPro/test_sharp_file.txt'

train_data = GoProData(blur_image_files=train_blur_image_files, sharp_image_files=train_sharp_image_files)
train_dataloader = DataLoader(dataset=train_data, num_workers=16, batch_size=2, shuffle=True)
test_data = GoProData(blur_image_files=test_blur_image_files, sharp_image_files=test_sharp_image_files)
test_dataloader = DataLoader(dataset=test_data, num_workers=1, batch_size=1, shuffle=False)

model = Mode()
model.apply(weight_init).cuda()
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
lr_decay = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1000)
loss_fn = Loss().cuda()

# for epoch in range(0, 1):
#     start = time.time()
#     for iter, images in enumerate(train_dataloader):
#         blur = (images[0] - 0.5).cuda()
#         gt = (images[1] - 0.5).cuda()
#         output = model(blur)
#         loss_val = loss_fn(output, gt)
#         optimizer.zero_grad()
#         loss_val.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 3)  # 梯度裁剪
#         optimizer.step()
#         if not iter % 10:
#             print(
#                 'epoch:{}\titer:{}\tloss:{}\tlr:{}\ttime:{}'.format(epoch, iter, loss_val.item(), lr_decay.get_lr()[0],
#                                                                     (time.time() - start)))
#             start = time.time()
#     lr_decay.step(epoch)  # 缺少训练时的评估
#
# print("训练完成，开始测试")
#
# for iter, images in enumerate(test_dataloader):
#     # blur = (images[0] - 0.5).unsqueeze(0).cuda()#这里为什么要增加一个维度
#     blur = (images[0] - 0.5).cuda()
#     gt = images[1]
#     [b, c, h, w] = blur.shape
#     new_h = h - h % 16
#     new_w = w - w % 16
#     blur = nn.functional.interpolate(blur, size=(new_h, new_w), mode="bilinear")
#     start = time.time()
#     deblur = model(blur)[-1]
#     print(len(deblur))
#     duration = time.time() - start
#     print("image deblur time:{}".format(duration))
#     print(deblur.shape)
#     print(gt.shape)
#     break

from trainer.base import Base
print(Base)
print(os.listdir('./experiment'))