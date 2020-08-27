import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineTripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(CosineTripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, z_x, z_y):
        """
        Parameters
            z_x: L2-normalized embeddings of one modality (batch_size, embedding_length)
            z_y: L2-normalized embeddings of the other modality (batch_size, embedding_length)
        """
        assert z_x.size() == z_y.size()
        batch_size, embedding_length = z_x.size()

        sim_xy = torch.mm(z_x, z_y.T)
        sim_yx = sim_xy.T
        sim_xx = torch.mm(z_x, z_x.T).clone().detach().requires_grad_(False)

        diag_mask = torch.eye(batch_size, dtype=torch.bool)

        sim_xx_neg = sim_xx[~diag_mask].view(batch_size, -1)
        sim_xx_neg = self.margin * torch.sigmoid(-sim_xx_neg)
        del sim_xx

        #pos_xy_rank = sim_xy.argsort(dim=1, descending=True)
        #pos_xy_rank = pos_xy_rank[diag_mask].view(batch_size, -1)
        sim_xy_pos = sim_xy[diag_mask].view(batch_size, -1)
        sim_xy_neg = sim_xy[~diag_mask].view(batch_size, -1)
        #sim_xy_neg, _ = sim_xy_neg.sort(dim=1, descending=True)
        #sim_xy_neg = sim_xy_neg[:, :batch_size // 2]
        #print(pos_xy_rank.flatten())
        #loss_xy = (1 + 1 / (batch_size - (pos_xy_rank + 1) + 1)) * F.relu_(sim_xy_neg - sim_xy_pos + self.margin)
        loss_xy = F.relu_(sim_xy_neg - sim_xy_pos + sim_xx_neg)
        
        #pos_yx_rank = sim_yx.argsort(dim=1, descending=True)
        #pos_yx_rank = pos_yx_rank[diag_mask].view(batch_size, -1)
        sim_yx_pos = sim_yx[diag_mask].view(batch_size, -1)
        sim_yx_neg = sim_yx[~diag_mask].view(batch_size, -1)
        #sim_yx_neg, _ = sim_yx_neg.sort(dim=1, descending=True)
        #sim_yx_neg = sim_yx_neg[:, :batch_size // 2]
        #print(pos_yx_rank.flatten())
        #loss_yx = (1 + 1 / (batch_size - (pos_yx_rank + 1) + 1)) * F.relu_(sim_yx_neg - sim_yx_pos + self.margin)
        loss_yx = F.relu_(sim_yx_neg - sim_yx_pos + 0.2)
        del diag_mask

        loss = loss_xy.mean() + loss_yx.mean()

        return loss, sim_xx_neg.mean().item()


if __name__ == "__main__":
    z_x = F.normalize(torch.randn(64, 256))
    z_y = F.normalize(torch.randn(64, 256))
    criterion = CosineTripletLoss(margin=0.2)
    loss = criterion(z_x, z_y)
