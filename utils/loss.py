import torch
import tensorflow.keras.backend as K
import tensorflow as tf
import torch.nn.functional as F
import numpy as np
import math

def p2p_loss(landmark_moving, landmark_fixed):
    return ((landmark_moving - landmark_fixed) ** 2).sum(axis=2).mean()

def focal_loss(landmark_moving, landmark_fixed):
    a = 1
    c = 2
    dist = torch.sqrt(((landmark_moving - landmark_fixed) ** 2).sum(axis=2).mean())
    return (dist ** 2) / (1 + (torch.exp(a * (c - dist))))

def pw_focal_loss(landmark_moving, landmark_fixed):
    a = 2
    c = 1.2
    dist = torch.sqrt(((landmark_moving - landmark_fixed) ** 2).sum(axis=2))
    l = (dist ** 2) / (1 + (torch.exp(a * (c - dist))))
    return l.mean()

def dice_loss(A, B):
    return 1 - 2 * (A * B).sum() / ((A + B).sum() + 10e-6)


class CrossCorrLoss:
    def __init__(self, reduction='mean'):
        self.reduction = reduction

    def loss(self, pred, target):
        
        pred_mean = torch.mean(pred)
        target_mean = torch.mean(target)
        pred_var = pred - pred_mean
        target_var = target - target_mean
        
        nominator = torch.sum(pred_var * target_var)
        denominator = torch.sqrt(torch.sum(pred_var * pred_var) * torch.sum(target_var * target_var)) + 1e-8
        
        if self.reduction == 'mean':
            temp = (nominator / denominator) / torch.numel(pred)
        elif self.reduction == 'sum':
            temp = nominator / denominator
        else:
            raise RuntimeError("reduction should be 'mean' or 'sum'")

        loss = -temp
        
        return loss

class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)

class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)