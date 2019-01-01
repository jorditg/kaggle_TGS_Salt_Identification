import torch
import torch.nn as nn
import torch.nn.functional as F


smooth = 1e-4
def fb_loss(preds, trues, beta):
    beta2 = beta*beta
    batch = preds.size(0)
    classes = preds.size(1)
    preds = preds.view(batch, classes, -1)
    trues = trues.view(batch, classes, -1)
    weights = torch.clamp(trues.sum(-1), 0., 1.)
    TP = (preds * trues).sum(2)
    FP = (preds * (1-trues)).sum(2)
    FN = ((1-preds) * trues).sum(2)
    Fb = ((1+beta2) * TP + smooth)/((1+beta2) * TP + beta2 * FN + FP + smooth)
    Fb = Fb * weights
    score = Fb.sum() / weights.sum()
    return torch.clamp(score, 0., 1.)


class TGSLoss(nn.Module):
    def __init__(self, beta=1.5, bce_w=0):
        # Dice Loss beta=1
        # beta=1.5 experimentally best results in https://arxiv.org/pdf/1803.11078.pdf
        super().__init__()
        self.bce_w = bce_w
        self.beta = beta
        self.bce = nn.BCELoss()

    def forward(self, input, target):
        if self.bce_w > 0:
            bce_loss = self.bce(input, target)
        else:
            bce_loss = 0

        fbloss = 1-fb_loss(input[:,0,:,:].unsqueeze(1), target[:,0,:,:].unsqueeze(1), self.beta)

        return fbloss + self.bce_w * bce_loss


class TGSDiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, input, target):
        #target = self.pad(target)
        iflat = input.view(-1)
        tflat = target.view(-1)
        smooth = self.smooth
        intersection = (iflat*tflat).sum()
        loss = 1. - ((2.*intersection + smooth)/(iflat.sum() + tflat.sum() + smooth))
        return loss
