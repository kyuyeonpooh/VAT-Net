import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineTripletLoss(nn.Module):
    def __init__(self, margin, loss_weight=1.):
        super(CosineTripletLoss, self).__init__()
        self.margin = margin
        self.loss_weight = loss_weight
    
    def forward(self, z_x, z_y):
        """
        Parameters
            z_x: L2-normalized embeddings of one modality (shape: B x N)
            z_y: L2-normalized embeddings of the other modality (shape: B x N)
        """
        assert z_x.size() == z_y.size()
        batch_size = z_x.size(0)

        sim_neg = torch.mm(z_x, z_y.T)  # B x B
        sim_pos = torch.diag(sim_neg).unsqueeze(1)
        
        loss_matrix = sim_neg - sim_pos + self.margin
        loss_matrix = torch.clamp(loss_matrix, min=0.)
        loss = torch.sum(loss_matrix) - self.margin * batch_size
        loss /= (batch_size ** 2 - batch_size)

        return self.loss_weight * loss


if __name__ == "__main__":
    z_x = F.normalize(torch.randn(64, 256))
    z_y = F.normalize(torch.randn(64, 256))
    criterion = CosineTripletLoss(margin=.5)
    loss = criterion(z_x, z_y)
