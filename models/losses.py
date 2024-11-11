import torch
import torch.nn as nn
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = alpha
        else:
            self.alpha = None
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Ensure alpha is on the same device as inputs
        if self.alpha is not None:
            self.alpha = self.alpha.to(inputs.device)

        # Compute Cross Entropy Loss
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')

        # Compute the probability of the true class
        pt = torch.exp(-ce_loss)

        # Apply class weights
        if self.alpha is not None:
            at = self.alpha.gather(0, targets)
            ce_loss = at * ce_loss

        # Compute Focal Loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
