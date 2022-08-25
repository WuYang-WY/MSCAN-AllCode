import torch.nn as nn
import torch.nn.functional as F


class MultiScaleLoss(nn.Module):
    '''
    多尺度损失
    '''

    def __init__(self):
        super(MultiScaleLoss, self).__init__()

    def forward(self, pred, gt):
        # pred 是一个数组
        loss = 0
        for item in pred:  # 遍历batch个预测图像
            [b, c, h, w] = item.shape
            gt_i = F.interpolate(gt, size=(h, w), mode='bilinear')  # 将清晰图像gt用双线性插值下采样到预测图像的尺寸(h*w)
            loss += F.mse_loss(item, gt_i)  # 累计损失
        return loss
