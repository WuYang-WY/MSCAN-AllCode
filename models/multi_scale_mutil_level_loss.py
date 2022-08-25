import torch.nn as nn
import torch.nn.functional as F


class MultiScaleMultiLevelLoss(nn.Module):
    '''
    多尺度损失
    '''

    def __init__(self):
        super(MultiScaleMultiLevelLoss, self).__init__()

    def forward(self, pred, gt):
        # pred 是一个dict字典
        loss = 0
        # R3
        R3 = pred['R3']
        I3 = pred['I3']
        _, _, h, w = R3.shape
        gt_3 = F.interpolate(gt, size=(h, w), mode='bilinear')
        loss += F.mse_loss(R3, gt_3)
        loss += F.mse_loss(I3, gt_3)
        # R2
        R2 = pred['R2']
        I2 = pred['I2']
        _, _, h, w = R2.shape
        gt_2 = F.interpolate(gt, size=(h, w), mode='bilinear')
        loss += F.mse_loss(R2, gt_2)
        loss += F.mse_loss(I2, gt_2)
        # R1
        R1 = pred['R1']
        I1 = pred['I1']
        loss += F.mse_loss(R1, gt)
        loss += F.mse_loss(I1, gt)
        return loss
