import torch

class RocAucLoss(torch.nn.Module):
    
    def __init__(self):
        super(RocAucLoss,self).__init__()
        
    def forward(self, y_pred, y_true):
        pos = y_pred[y_true.byte()]
        neg = y_pred[~y_true.byte()]

        pos = pos.unsqueeze(0)
        neg = neg.unsqueeze(1)
        
        gamma = 0.2
        p = 3
        
        difference = torch.zeros_like(pos * neg) + pos - neg - gamma
        masked = difference[difference < 0.0]
        
        return  torch.sum(torch.pow(-masked, p))
