import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.TextNet import TextNet
from modules.VisualNet import VisualNet


class VTNet(nn.Module):
    def __init__(self):
        super(VTNet, self).__init__()
        
        self.visual_net = VisualNet()
        self.text_net = TextNet()

    def forward(self, v, t):
        z_v = self.visual_net(v)
        z_t = self.text_net(t)
        
        return z_v, z_t
