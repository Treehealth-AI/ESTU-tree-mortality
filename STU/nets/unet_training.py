import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial


def CE_Loss(cls_weights, num_classes=4):
    """
    Cross Entropy Loss function compatible with existing training framework
    Args:
        cls_weights: Class weights for handling class imbalance
        num_classes: Number of classes
    Returns:
        Loss function
    """

    def ce_loss_fn(inputs, target):
        n, c, h, w = inputs.size()
        nt, ht, wt = target.size()

        if h != ht or w != wt:
            inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

        # Reshape for cross entropy calculation
        inputs = inputs.permute(0, 2, 3, 1).contiguous().view(-1, c)
        target = target.view(-1).long()

        # Handle class weights
        if cls_weights is not None:
            if isinstance(cls_weights, np.ndarray):
                weights = torch.from_numpy(cls_weights).float()
            else:
                weights = cls_weights
            if weights.device != inputs.device:
                weights = weights.to(inputs.device)
            ce_loss = F.cross_entropy(inputs, target, weight=weights)
        else:
            ce_loss = F.cross_entropy(inputs, target)

        return ce_loss

    return ce_loss_fn


def Dice_loss(num_classes, beta=1, smooth=1e-5):
    """
    Dice Loss function compatible with existing training framework
    Args:
        num_classes: Number of classes
        beta: Weight parameter for F-beta score
        smooth: Smoothing parameter to avoid division by zero
    Returns:
        Dice loss function
    """

    def dice_loss_fn(inputs, target):
        n, c, h, w = inputs.size()
        nt, ht, wt = target.size()

        if h != ht or w != wt:
            inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

        temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
        temp_target = target.view(n, -1, 1)

        # Convert target to one-hot encoding
        if inputs.device.type == 'cuda':
            temp_target = torch.zeros(n, ht * wt, c).cuda().scatter_(2, temp_target.long(), 1)
        else:
            temp_target = torch.zeros(n, ht * wt, c).scatter_(2, temp_target.long(), 1)

        tp = torch.sum(temp_target[..., :-1] * temp_inputs[..., :-1], axis=[0, 1])
        fp = torch.sum(temp_inputs[..., :-1], axis=[0, 1]) - tp
        fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp

        score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
        dice_loss = 1 - torch.mean(score)
        return dice_loss

    return dice_loss_fn


def Focal_Loss(num_classes, alpha=0.5, gamma=2.):
    """
    Focal Loss function compatible with existing training framework
    Args:
        num_classes: Number of classes
        alpha: Alpha parameter for addressing class imbalance
        gamma: Gamma parameter for focusing on hard examples
    Returns:
        Focal loss function
    """

    def focal_loss_fn(inputs, target):
        n, c, h, w = inputs.size()
        nt, ht, wt = target.size()

        if h != ht or w != wt:
            inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

        # Reshape inputs and targets
        inputs = inputs.permute(0, 2, 3, 1).contiguous().view(-1, c)
        target = target.view(-1).long()

        # Calculate cross entropy
        ce_loss = F.cross_entropy(inputs, target, reduction='none')
        pt = torch.exp(-ce_loss)

        # Calculate focal loss
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()

    return focal_loss_fn


def weights_init(net, init_type='normal', init_gain=0.02):
    """
    Initialize network weights
    Args:
        net: Network to initialize
        init_type: Initialization type ('normal', 'xavier', 'kaiming', 'orthogonal')
        init_gain: Initialization gain
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
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
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
        lr_decay_type: Type of learning rate decay ('cos' or 'step')
        lr: Initial learning rate
        min_lr: Minimum learning rate
        total_iters: Total number of iterations
        warmup_iters_ratio: Warmup iterations ratio
        warmup_lr_ratio: Warmup learning rate ratio
        no_aug_iter_ratio: No augmentation iterations ratio
        step_num: Number of steps for step decay
    Returns:
        Learning rate scheduler function
    """

    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                    1.0 + np.cos(
                np.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
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
    Set optimizer learning rate
    Args:
        optimizer: PyTorch optimizer
        lr_scheduler_func: Learning rate scheduler function
        epoch: Current epoch
    """
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Test the functions
if __name__ == "__main__":
    print("Testing UNet training functions...")

    # Test parameters
    num_classes = 4
    batch_size = 2
    height, width = 256, 256

    # Create dummy data
    predictions = torch.randn(batch_size, num_classes, height, width)
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    cls_weights = np.array([1.0, 1.2, 1.5, 1.8])

    print(f"Input shapes - Predictions: {predictions.shape}, Targets: {targets.shape}")

    # Test CE Loss
    ce_loss_fn = CE_Loss(cls_weights, num_classes)
    ce_loss = ce_loss_fn(predictions, targets)
    print(f"✅ CE Loss: {ce_loss.item():.4f}")

    # Test Dice Loss
    dice_loss_fn = Dice_loss(num_classes)
    dice_loss = dice_loss_fn(predictions, targets)
    print(f"✅ Dice Loss: {dice_loss.item():.4f}")

    # Test Focal Loss
    focal_loss_fn = Focal_Loss(num_classes)
    focal_loss = focal_loss_fn(predictions, targets)
    print(f"✅ Focal Loss: {focal_loss.item():.4f}")

    # Test learning rate scheduler
    lr_scheduler = get_lr_scheduler('cos', 1e-3, 1e-5, 100)
    test_lr = lr_scheduler(50)
    print(f"✅ LR Scheduler test - LR at epoch 50: {test_lr:.6f}")

    print("🎉 All training function tests passed!")