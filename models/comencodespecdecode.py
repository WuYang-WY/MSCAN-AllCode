import torch.nn as nn
import torch.nn.functional as F
import torch


class ResBlock(nn.Module):

    def __init__(self, channel, ksize=3, spp=(4, 2, 1)):
        super(ResBlock, self).__init__()
        padding = ksize // 2
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=ksize, padding=padding)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=ksize, padding=padding)

    def forward(self, x):
        net = self.conv1(x)
        act_net = F.relu(net)
        net = self.conv2(act_net)
        return x + net


class Encode(nn.Module):

    def __init__(self, ksize=3):
        super(Encode, self).__init__()
        padding = ksize // 2
        self.layer1 = nn.Conv2d(3, 32, kernel_size=ksize, padding=padding)
        self.layer2 = ResBlock(32, ksize=ksize)
        self.layer3 = ResBlock(32, ksize=ksize)
        self.layer4 = ResBlock(32, ksize=ksize)
        # Conv2
        self.layer5 = nn.Conv2d(32, 64, kernel_size=ksize, stride=2, padding=padding)
        self.layer6 = ResBlock(64, ksize=ksize)
        self.layer7 = ResBlock(64, ksize=ksize)
        self.layer8 = ResBlock(64, ksize=ksize)
        # Conv3
        self.layer9 = nn.Conv2d(64, 128, kernel_size=ksize, stride=2, padding=padding)
        self.layer10 = ResBlock(128, ksize=ksize)
        self.layer11 = ResBlock(128, ksize=ksize)
        self.layer12 = ResBlock(128, ksize=ksize)

    def forward(self, x):
        # Conv1
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Conv2
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        # Conv3
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        return x


class Decode(nn.Module):

    def __init__(self, ksize=3):
        super(Decode, self).__init__()
        # Deconv3
        self.layer13 = ResBlock(128, ksize=ksize)
        self.layer14 = ResBlock(128, ksize=ksize)
        self.layer15 = ResBlock(128, ksize=ksize)
        self.layer16 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        # Deconv2
        self.layer17 = ResBlock(64, ksize=ksize)
        self.layer18 = ResBlock(64, ksize=ksize)
        self.layer19 = ResBlock(64, ksize=ksize)
        self.layer20 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        # Deconv1
        self.layer21 = ResBlock(32, ksize=ksize)
        self.layer22 = ResBlock(32, ksize=ksize)
        self.layer23 = ResBlock(32, ksize=ksize)
        self.layer24 = nn.Conv2d(32, 3, kernel_size=ksize, padding=ksize // 2)

    def forward(self, x):
        # Deconv3
        x = self.layer13(x)
        x = self.layer14(x)
        x = self.layer15(x)
        x = self.layer16(x)
        # Deconv2
        x = self.layer17(x)
        x = self.layer18(x)
        x = self.layer19(x)
        x = self.layer20(x)
        # Deconv1
        x = self.layer21(x)
        x = self.layer22(x)
        x = self.layer23(x)
        x = self.layer24(x)
        return x


class ComEncodeSpecDecode(nn.Module):

    def __init__(self, is_test=False):
        super(ComEncodeSpecDecode, self).__init__()
        self.encode = Encode()
        self.blur_decode = Decode()
        self.gt_decode = Decode()
        self.is_test = is_test

    def test(self):
        self.is_test = True

    def train(self):
        self.is_test = False

    def forward(self, blur, gt):
        if not self.is_test:
            blur_encode = self.encode(blur)
            re_blur = self.blur_decode(blur_encode)
            gt_encode = self.encode(gt)
            re_gt = self.gt_decode(gt_encode)
            return re_blur, re_gt, blur_encode, gt_encode
        else:
            blur_encoder = self.encode(blur)
            re_gt = self.gt_decode(blur_encoder)
            return re_gt, None, None, None

