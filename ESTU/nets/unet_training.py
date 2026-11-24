import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def CE_Loss(inputs, target, cls_weights, num_classes=4):
    """
    Cross Entropy Loss for multi-class segmentation

    Args:
        inputs: model predictions [B, C, H, W]
        target: ground truth labels [B, H, W]
        cls_weights: class weights tensor
        num_classes: number of classes (default: 4 for forest segmentation)
    """
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()

    if h != ht or w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_target)
    return CE_loss


def Focal_Loss(inputs, target, cls_weights, num_classes=4, alpha=0.5, gamma=2):
    """
    Focal Loss for handling class imbalance in segmentation

    Args:
        inputs: model predictions [B, C, H, W]
        target: ground truth labels [B, H, W]
        cls_weights: class weights tensor
        num_classes: number of classes (default: 4 for forest segmentation)
        alpha: focal loss alpha parameter
        gamma: focal loss gamma parameter
    """
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()

    if h != ht or w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    logpt = -nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes, reduction='none')(temp_inputs,
                                                                                                 temp_target)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss


def Dice_loss(inputs, target, beta=1, smooth=1e-5):
    """
    Fixed Dice Loss for multi-class segmentation

    Args:
        inputs: model predictions [B, C, H, W]
        target: ground truth labels [B, H, W] (NOT [B, H, W, C])
        beta: dice loss beta parameter
        smooth: smoothing parameter to avoid division by zero
    """
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()  # Fixed: target should be 3D, not 4D

    if h != ht or w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    # Apply softmax to get probabilities
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)

    # Convert target to one-hot encoding
    temp_target = target.view(n, -1).long()  # [B, H*W]
    temp_target_one_hot = torch.zeros(n, ht * wt, c).type_as(inputs)
    temp_target_one_hot.scatter_(2, temp_target.unsqueeze(2), 1)  # [B, H*W, C]

    # Calculate dice loss for all classes
    tp = torch.sum(temp_target_one_hot * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target_one_hot, axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss


def Dice_loss_simple(inputs, target, smooth=1e-5):
    """
    Alternative simplified Dice Loss implementation

    Args:
        inputs: model predictions [B, C, H, W]
        target: ground truth labels [B, H, W]
        smooth: smoothing parameter
    """
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()

    if h != ht or w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    # Apply softmax to get probabilities
    inputs = F.softmax(inputs, dim=1)

    # Convert target to one-hot encoding
    target_one_hot = F.one_hot(target.long(), num_classes=c).permute(0, 3, 1, 2).float()

    # Calculate intersection and union for each class
    intersection = torch.sum(inputs * target_one_hot, dim=(2, 3))
    union = torch.sum(inputs, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))

    # Calculate Dice coefficient
    dice = (2. * intersection + smooth) / (union + smooth)

    # Return average loss across all classes
    return 1 - torch.mean(dice)


def weights_init(net, init_type='normal', init_gain=0.02):
    """
    Initialize network weights

    Args:
        net: network to initialize
        init_type: initialization type ('normal', 'xavier', 'kaiming', 'orthogonal')
        init_gain: initialization gain
    """

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('Initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('Initialize network with %s type' % init_type)
    net.apply(init_func)


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.05, warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.05, step_num=10):
    """
    Get learning rate scheduler function

    Args:
        lr_decay_type: 'cos' for cosine annealing, 'step' for step decay
        lr: initial learning rate
        min_lr: minimum learning rate
        total_iters: total number of iterations
        warmup_iters_ratio: ratio of warmup iterations
        warmup_lr_ratio: ratio of warmup learning rate
        no_aug_iter_ratio: ratio of no augmentation iterations
        step_num: number of steps for step decay
    """

    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # Quadratic warmup
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            # Cosine annealing
            lr = min_lr + 0.5 * (lr - min_lr) * (
                    1.0 + math.cos(
                math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must be above 1.")
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    """
    Set optimizer learning rate based on scheduler

    Args:
        optimizer: PyTorch optimizer
        lr_scheduler_func: learning rate scheduler function
        epoch: current epoch
    """
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr