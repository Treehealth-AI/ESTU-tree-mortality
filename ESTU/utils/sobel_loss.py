import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EdgeAwareLoss(nn.Module):
    pass

    def __init__(self, edge_weight=0.5, dice_weight=1.0, focal_weight=1.0):
        super(EdgeAwareLoss, self).__init__()
        self.edge_weight = edge_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight


        sobel_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0)

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def extract_edges_from_mask(self, mask):
        pass
        # mask: [B, H, W] -> [B, 1, H, W]
        if mask.dim() == 3:
            mask = mask.unsqueeze(1).float()
        elif mask.dim() == 4 and mask.shape[1] > 1:

            mask = torch.argmax(mask, dim=1, keepdim=True).float()


        grad_x = F.conv2d(mask, self.sobel_x, padding=1)
        grad_y = F.conv2d(mask, self.sobel_y, padding=1)


        edges = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        edges = (edges > 0.1).float()

        return edges.squeeze(1)  # [B, H, W]

    def dice_loss(self, pred, target, smooth=1e-6):
        pass
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(1)
        union = pred_flat.sum(1) + target_flat.sum(1)

        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()

    def focal_loss(self, pred, target, alpha=0.25, gamma=2.0):
        pass


        pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)


        pred_logits = torch.log(pred / (1 - pred))


        ce_loss = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()

    def edge_loss(self, pred_edges, target_edges):
        pass


        pred_edges = torch.clamp(pred_edges, min=1e-7, max=1 - 1e-7)
        pred_logits = torch.log(pred_edges / (1 - pred_edges))

        bce_loss = F.binary_cross_entropy_with_logits(pred_logits, target_edges, reduction='none')


        edge_weight_map = target_edges * 2.0 + 1.0
        weighted_loss = bce_loss * edge_weight_map

        return weighted_loss.mean()

    def forward(self, seg_pred, edge_pred, seg_target):
        pass


        device = seg_pred.device


        seg_prob = F.softmax(seg_pred, dim=1)


        target_edges = self.extract_edges_from_mask(seg_target).to(device)


        total_loss = 0


        if self.dice_weight > 0 or self.focal_weight > 0:
            for c in range(seg_pred.shape[1]):
                if c == 0:
                    continue


                class_target = (seg_target == c).float()
                class_pred = seg_prob[:, c]

                if self.dice_weight > 0:
                    dice_loss = self.dice_loss(class_pred, class_target)
                    total_loss += self.dice_weight * dice_loss

                if self.focal_weight > 0:
                    focal_loss = self.focal_loss(class_pred, class_target)
                    total_loss += self.focal_weight * focal_loss


        if self.edge_weight > 0:
            edge_pred_squeezed = edge_pred.squeeze(1)  # [B, H, W]

            edge_pred_prob = torch.sigmoid(edge_pred_squeezed)
            edge_loss = self.edge_loss(edge_pred_prob, target_edges)
            total_loss += self.edge_weight * edge_loss

        return total_loss


class CombinedLoss(nn.Module):
    pass

    def __init__(self, num_classes, dice_loss=True, focal_loss=True,
                 cls_weights=None, edge_weight=0.3):
        super(CombinedLoss, self).__init__()
        self.num_classes = num_classes
        self.dice_loss = dice_loss
        self.focal_loss = focal_loss
        self.edge_weight = edge_weight

        if cls_weights is not None:
            if isinstance(cls_weights, np.ndarray):
                cls_weights = torch.from_numpy(cls_weights).float()
            self.register_buffer('cls_weights', cls_weights)
        else:
            self.cls_weights = None


        self.edge_aware_loss = EdgeAwareLoss(
            edge_weight=edge_weight,
            dice_weight=1.0 if dice_loss else 0.0,
            focal_weight=1.0 if focal_loss else 0.0
        )


        self.ce_loss = nn.CrossEntropyLoss(weight=self.cls_weights)

    def forward(self, predictions, targets):
        pass


        if isinstance(predictions, tuple):
            seg_pred, edge_pred = predictions

            loss = self.edge_aware_loss(seg_pred, edge_pred, targets)
        else:

            seg_pred = predictions
            loss = self.ce_loss(seg_pred, targets.long())

        return loss


def calculate_loss(predictions, targets, dice_loss=True, focal_loss=True,
                   cls_weights=None, num_classes=4):
    pass
    loss_fn = CombinedLoss(
        num_classes=num_classes,
        dice_loss=dice_loss,
        focal_loss=focal_loss,
        cls_weights=cls_weights,
        edge_weight=0.3
    )

    return loss_fn(predictions, targets)