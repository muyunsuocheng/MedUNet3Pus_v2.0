import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff

class SegmentationMetrics:
    def __init__(self, num_classes, epsilon=1e-6):
        self.num_classes = num_classes
        self.epsilon = epsilon

    def _one_hot(self, tensor):
        """将标签转换为one-hot编码"""
        return torch.zeros(tensor.shape[0], self.num_classes, *tensor.shape[1:]).scatter_(
            1, tensor.unsqueeze(1), 1)

    def dice_score(self, pred, target):
        """计算Dice系数"""
        pred = self._one_hot(pred.argmax(dim=1))
        target = self._one_hot(target)
        
        intersection = (pred * target).sum(dim=(2,3))
        union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
        return (2. * intersection + self.epsilon) / (union + self.epsilon)

    def iou_score(self, pred, target):
        """计算IoU"""
        pred = self._one_hot(pred.argmax(dim=1))
        target = self._one_hot(target)
        
        intersection = (pred * target).sum(dim=(2,3))
        union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) - intersection
        return (intersection + self.epsilon) / (union + self.epsilon)

    def hausdorff_distance(self, pred, target):
        """计算Hausdorff距离（CPU实现）"""
        pred_np = pred.argmax(dim=1).cpu().numpy().astype(bool)
        target_np = target.cpu().numpy().astype(bool)
        
        distances = []
        for p, t in zip(pred_np, target_np):
            if np.sum(p) == 0 or np.sum(t) == 0:
                distances.append(np.nan)
                continue
            distances.append(max(
                directed_hausdorff(p, t)[0],
                directed_hausdorff(t, p)[0]
            ))
        return torch.tensor(np.nanmean(distances))

    def calculate_all(self, pred, target):
        """返回所有指标字典"""
        return {
            'Dice': self.dice_score(pred, target).mean(),
            'IoU': self.iou_score(pred, target).mean(),
            'HD': self.hausdorff_distance(pred, target)
        }