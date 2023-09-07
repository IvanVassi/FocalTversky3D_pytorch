import torch
import torch.nn as nn

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        # Ensure the predictions are in the same dimension as y_true
        y_pred = torch.sigmoid(y_pred)
        
        # Calculate the Tversky loss
        tp = (y_true * y_pred).sum(dim=(2, 3, 4))
        fn = (y_true * (1 - y_pred)).sum(dim=(2, 3, 4))
        fp = ((1 - y_true) * y_pred).sum(dim=(2, 3, 4))
        
        tversky_index = tp / (tp + self.alpha * fn + self.beta * fp)
        
        # Calculate the Focal Tversky loss
        loss = (1 - tversky_index).pow(self.gamma)
        
        return loss.mean()