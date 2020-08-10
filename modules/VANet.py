import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.AudioNet import AudioNet
from modules.VisualNet import VisualNet


class VANet(nn.Module):
    def __init__(self):
        super(VANet, self).__init__()

        self.v_net = VisualNet()
        self.a_net = AudioNet()

    def forward(self, v, a):
        z_v = self.v_net(v)
        z_a = self.a_net(a)
        
        return z_v, z_a
