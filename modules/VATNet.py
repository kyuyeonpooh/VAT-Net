import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.AudioNet import AudioNet
from modules.TextNet import TextNet
from modules.VisualNet import VisualNet


class VATNet(nn.Module):
    def __init__(self):
        super(VATNet, self).__init__()
        
        self.visual_net = VisualNet()
        self.audio_net = AudioNet()
        self.text_net = TextNet()

    def forward(self, v, w, mode):        
        if mode == "va":
            z_v = self.visual_net(v)
            z_w = self.audio_net(w)
            return z_v, z_w
        elif mode == "vt":
            z_v = self.visual_net(v)
            z_w = self.text_net(w)
            return z_v, z_w
        elif mode == "vat":
            z_v = self.visual_net(v)
            z_a = self.audio_net(w[0])
            z_t = self.text_net(w[1])
            return z_v, z_a, z_t
        
        
