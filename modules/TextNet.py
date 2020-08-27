import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# from config import config


class TextNet(nn.Module):
    def __init__(self):
        super(TextNet, self).__init__()

        with open("../ssd/CocoFlickr-word2vec.npy", "rb") as npy:
            pretrained_embedding = np.load(npy)
            pretrained_embedding = torch.FloatTensor(pretrained_embedding)
        self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=True, padding_idx=0)

        self.relu = nn.ReLU(True)
        self.max_pool = nn.MaxPool1d(2)

        self.conv1_1 = nn.Conv1d(300, 128, kernel_size=2, bias=False)
        self.bn1_1 = nn.BatchNorm1d(128)
        self.conv1_2 = nn.Conv1d(128, 128, kernel_size=2, bias=False)
        self.bn1_2 = nn.BatchNorm1d(128)

        self.conv2_1 = nn.Conv1d(128, 256, kernel_size=2, bias=False)
        self.bn2_1 = nn.BatchNorm1d(256)
        self.conv2_2 = nn.Conv1d(256, 256, kernel_size=2, bias=False)
        self.bn2_2 = nn.BatchNorm1d(256)

        self.conv3_1 = nn.Conv1d(256, 512, kernel_size=2, bias=False)
        self.bn3_1 = nn.BatchNorm1d(512)
        self.conv3_2 = nn.Conv1d(512, 512, kernel_size=2, bias=False)
        self.bn3_2 = nn.BatchNorm1d(512)

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256)
        )

        self.apply(self.init_weights)
    
    def init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)  # from (batch_size, length, channel) to (batch_size, channel, length)
        
        x = self.relu(self.bn1_1(self.conv1_1(x)))
        x = F.pad(x, pad=(0, 1))
        x = self.relu(self.bn1_2(self.conv1_2(x)))
        x = F.pad(x, pad=(0, 1))

        x = self.relu(self.bn2_1(self.conv2_1(x)))
        x = F.pad(x, pad=(0, 1))
        x = self.relu(self.bn2_2(self.conv2_2(x)))
        x = F.pad(x, pad=(0, 1))
        x = self.max_pool(x)

        x = self.relu(self.bn3_1(self.conv3_1(x)))
        x = F.pad(x, pad=(0, 1))
        x = self.relu(self.bn3_2(self.conv3_2(x)))
        x = F.pad(x, pad=(0, 1))
        x = self.max_pool(x)

        x = self.adaptive_pool(x)
        x = x.flatten(1)
        x = self.fc_head(x)
        x = F.normalize(x, p=2, dim=1)

        return x
