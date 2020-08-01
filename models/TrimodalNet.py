import torch
import torch.nn as nn
import torch.nn.functional as F

from models import AudioNet, VisualNet


class TrimodalNet(nn.Module):
    def __init__(self):
        super(TrimodalNet, self).__init__()

        self.visnet = VisualNet.VisualNet()  # (B, 512)
        self.vis_head = nn.Sequential(
            nn.Linear(2048, 512),  # (B, 512)
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),  # (B, 512)
            nn.BatchNorm1d(512)
        )

        self.audnet = AudioNet.AudioNet()
        self.aud_head = nn.Sequential(
            nn.Linear(2048, 512)
        )

        self.fc1_va = nn.Linear(512, 512)
        self.relu_va = nn.ReLU(inplace=True)
        self.fc2_va = nn.Linear(512, 1)
        self.sigmoid_va = nn.Sigmoid()

        self.apply(self.init_bn)

    def init_bn(self, m):
        if isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)            

    def forward(self, img, aud):
        v = self.visnet(img)
        v = self.vis_head(v)
        z_v = F.normalize(v, p=2, dim=1)

        a = self.audnet(aud)
        a = self.aud_head(a)
        z_a = F.normalize(a, p=2, dim=1)

        z_va = z_v * z_a
        z_va = self.relu_va(self.fc1_va(z_va))
        out_va = self.sigmoid_va(self.fc2_va(z_va))

        return out_va.squeeze()
