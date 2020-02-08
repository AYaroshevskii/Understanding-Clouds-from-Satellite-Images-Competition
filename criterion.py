import torch.nn as nn
import numpy as np

loss_bce = nn.BCEWithLogitsLoss()

# DICE metric
def dice(img1, img2):

    if (img1.sum() == 0) and (img2.sum() == 0):
        return 1

    SMOOTH = 1e-7

    img1 = np.asarray(img1).astype(np.bool)
    img2 = np.asarray(img2).astype(np.bool)

    intersection = np.logical_and(img1, img2)

    return 2.0 * intersection.sum() / (img1.sum() + img2.sum() + SMOOTH)


# BCE_DICE_Loss


def dice_loss(pred, target, smooth=1.0):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = 1 - (
        (2.0 * intersection + smooth)
        / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
    )

    return loss.mean()


def bce_dice_loss(y_true, y_pred, w=0.75):
    return w * nn.BCELoss()(y_true, y_pred) + (1 - w) * dice_loss(y_true, y_pred)
