import torch.nn as nn
import torch.nn.functional as F


class EncodeDecodeLoss(nn.Module):

    def __init__(self):
        super(EncodeDecodeLoss, self).__init__()

    def forward(self, blur, re_blur, gt, re_gt):
        loss = F.mse_loss(blur, re_blur)
        loss += F.mse_loss(gt, re_gt)
        return loss