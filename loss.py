import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 1. Dice Loss 实现
class DiceLoss(nn.Module):
    """Dice Loss for handling class imbalance in building localization"""
    def __init__(self, smooth=1.0, ignore_background=False):
        super().__init__()
        self.smooth = smooth
        self.ignore_background = ignore_background
        
    def forward(self, inputs, targets):
        """
        inputs: predicted probability map [B, H, W]
        targets: binary ground truth [B, H, W]
        """
        # Flatten the tensors
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        
        if self.ignore_background:
            # Only consider foreground pixels (buildings)
            mask = (targets > 0.5)
            inputs = inputs[mask]
            targets = targets[mask]
        
        # Calculate intersection and union
        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()
        
        # Calculate Dice coefficient
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Return 1 - dice for minimization
        return 1 - dice

# 2. Lovasz Softmax Loss 实现
def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_softmax_flat(inputs, targets, classes='present', ignore_index=0):
    """
    Multi-class Lovasz-Softmax loss
    inputs: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
    targets: [P] Tensor, ground truth labels (between 0 and C - 1)
    classes: 'all' for all, 'present' for classes present in ground truth, or a list of classes to average.
    ignore_index: specifies a target value that is ignored
    """
    # Filter out ignore_index pixels
    valid = (targets != ignore_index)
    inputs = inputs[valid]
    targets = targets[valid]
    
    if inputs.numel() == 0:
        return torch.zeros(1, requires_grad=True).to(inputs.device)
    
    C = inputs.size(1)
    losses = []
    classes_to_use = range(C)
    
    for c in classes_to_use:
        if c == ignore_index:
            continue
            
        target_c = (targets == c).float()
        if classes == 'present' and target_c.sum() == 0:
            continue
            
        class_prob = inputs[:, c]
        errors = (target_c - class_prob).abs()
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        target_c_sorted = target_c[perm]
        
        # Compute the gradient
        grad = lovasz_grad(target_c_sorted)
        
        # Compute the loss
        loss = torch.dot(F.relu(errors_sorted), grad)
        losses.append(loss)
    
    if len(losses) == 0:
        return torch.zeros(1, requires_grad=True).to(inputs.device)
    
    return sum(losses) / len(losses)

def lovasz_softmax(inputs, targets, classes='present', per_image=False, ignore_index=0):
    """
    Multi-class Lovasz-Softmax loss
    inputs: [B, C, H, W] class probabilities
    targets: [B, H, W] ground truth labels
    """
    if per_image:
        loss = 0
        for i in range(inputs.shape[0]):
            loss += lovasz_softmax_flat(
                inputs[i].permute(1, 2, 0).contiguous().view(-1, inputs.shape[1]),
                targets[i].view(-1),
                classes, ignore_index=ignore_index
            )
        return loss / inputs.shape[0]
    else:
        return lovasz_softmax_flat(
            inputs.permute(0, 2, 3, 1).contiguous().view(-1, inputs.shape[1]),
            targets.view(-1),
            classes, ignore_index=ignore_index
        )

class LovaszSoftmaxLoss(nn.Module):
    """
    Wrapper for Lovasz Softmax loss to handle multi-class segmentation
    with class imbalance, especially for damage assessment
    """
    def __init__(self, ignore_index=0, per_image=False):
        super().__init__()
        self.ignore_index = ignore_index
        self.per_image = per_image
        
    def forward(self, inputs, targets):
        """
        inputs: [B, C, H, W] logits (not probabilities)
        targets: [B, H, W] class indices
        """
        # Apply softmax to convert logits to probabilities
        inputs_softmax = F.softmax(inputs, dim=1)
        return lovasz_softmax(
            inputs_softmax, targets, 
            classes='present', 
            per_image=self.per_image,
            ignore_index=self.ignore_index
        )

# 3. 建筑定位损失 (BCE + Dice)
class BuildingLocalizationLoss(nn.Module):
    """Combined loss for building localization task"""
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_loss = nn.BCELoss(reduction='mean')
        self.dice_loss = DiceLoss(smooth=1.0)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
    def forward(self, inputs, targets):
        """
        inputs: predicted building probability map [B, 1, H, W] or [B, H, W]
        targets: binary building mask [B, H, W]
        """
        # Ensure inputs have the right shape
        if inputs.dim() == 4:
            inputs = inputs.squeeze(1)
            
        # BCE Loss
        bce_loss = self.bce_loss(inputs, targets)
        
        # Dice Loss
        dice_loss = self.dice_loss(inputs, targets)
        
        # Combined loss
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

# 4. 损伤分类损失 (CE + Lovasz)
class DamageClassificationLoss(nn.Module):
    """Combined loss for damage classification task"""
    def __init__(self, ce_weight=0.5, lovasz_weight=0.5, ignore_index=0):
        super().__init__()
        # CrossEntropyLoss ignores background (class 0) by default
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')
        self.lovasz_loss = LovaszSoftmaxLoss(ignore_index=ignore_index)
        self.ce_weight = ce_weight
        self.lovasz_weight = lovasz_weight
        
    def forward(self, inputs, targets):
        """
        inputs: [B, C, H, W] logits for each damage class
        targets: [B, H, W] class indices (0=background, 1=no damage, 2=minor, 3=major, 4=destroyed)
        """
        # Cross Entropy Loss
        ce_loss = self.ce_loss(inputs, targets)
        
        # Lovasz Softmax Loss
        lovasz_loss = self.lovasz_loss(inputs, targets)
        
        # Combined loss
        return self.ce_weight * ce_loss + self.lovasz_weight * lovasz_loss

# 5. 总体损失函数 (Loverall = Lloc + αLdam)
class DamFormerLoss(nn.Module):
    """Overall loss function for DamFormer architecture"""
    def __init__(self, alpha=1.0, bce_weight=0.5, dice_weight=0.5, 
                 ce_weight=0.5, lovasz_weight=0.5, ignore_index=0):
        super().__init__()
        self.alpha = alpha
        self.building_loss = BuildingLocalizationLoss(
            bce_weight=bce_weight, 
            dice_weight=dice_weight
        )
        self.damage_loss = DamageClassificationLoss(
            ce_weight=ce_weight,
            lovasz_weight=lovasz_weight,
            ignore_index=ignore_index
        )
        
    def forward(self, predictions, targets):
        """
        predictions: dict with keys 'building' and 'damage'
            - 'building': [B, 1, H, W] probability map
            - 'damage': [B, C, H, W] logits for damage classes
        targets: dict with keys 'building' and 'damage'
            - 'building': [B, H, W] binary mask
            - 'damage': [B, H, W] class indices
        """
        # Building localization loss
        building_loss = self.building_loss(predictions['building'], targets['building'])
        
        # Damage classification loss
        damage_loss = self.damage_loss(predictions['damage'], targets['damage'])
        
        # Overall loss
        total_loss = building_loss + self.alpha * damage_loss
        
        return {
            'total_loss': total_loss,
            'building_loss': building_loss,
            'damage_loss': damage_loss
        }

# 6. 损失函数使用示例
# if __name__ == "__main__":
#     # Create dummy predictions and targets
#     batch_size = 2
#     num_classes = 5  # background + 4 damage levels
#     height, width = 256, 256
    
#     # Random predictions
#     building_pred = torch.sigmoid(torch.randn(batch_size, 1, height, width))
#     damage_pred = torch.randn(batch_size, num_classes, height, width)
    
#     # Random targets (for demonstration only)
#     building_target = (torch.rand(batch_size, height, width) > 0.7).float()
#     damage_target = torch.randint(0, num_classes, (batch_size, height, width))
#     # Set background where there are no buildings
#     damage_target[building_target == 0] = 0
    
#     # Create loss function
#     criterion = DamFormerLoss(alpha=1.0)
    
#     # Compute loss
#     predictions = {
#         'building': building_pred,
#         'damage': damage_pred
#     }
    
#     targets = {
#         'building': building_target,
#         'damage': damage_target
#     }
    
#     losses = criterion(predictions, targets)
    
#     print(f"Total Loss: {losses['total_loss'].item():.4f}")
#     print(f"Building Loss: {losses['building_loss'].item():.4f}")
#     print(f"Damage Loss: {losses['damage_loss'].item():.4f}")