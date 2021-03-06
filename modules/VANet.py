import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.AudioNet import AudioNet
from modules.VisualNet import VisualNet


class VANet(nn.Module):
    def __init__(self):
        super(VANet, self).__init__()
        
        self.visual_net = VisualNet()
        self.audio_net = AudioNet()

    def forward(self, v, a):
        z_v = self.visual_net(v)
        z_a = self.audio_net(a)
        
        return z_v, z_a
