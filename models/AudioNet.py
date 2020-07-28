import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class AudioNet(nn.Module):
    """ AudioNet is following ResNet18 architecture,
    but with little modification on the first conv layer.
    """
    def __init__(self):
        super(AudioNet, self).__init__()

        conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(conv1.weight, mode="fan_out", nonlinearity="relu")
        
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.conv1 = conv1  # Change first conv layer's in_channel to 1
        self.resnet18.fc = nn.Identity()  # Discard last fc layer

    def forward(self, x):
        return self.resnet18(x)  # [B, 512]

"""
class AudioNet(nn.Module):
    def __init__(self, n_block=5, pool_type="avg"):
        super(AudioNet, self).__init__()

        conv_blocks = [ConvBlock(1, 64, pool_type=pool_type)]
        for i in range(n_block - 1):
            conv_blocks.append(
                ConvBlock(
                    in_channels=64 * (2 ** i),
                    out_channels=64 * (2 ** (i + 1)),
                    pool_type="none" if i == n_block - 2 else pool_type,
                )
            )
        self.conv_blocks = nn.Sequential(*conv_blocks)

        self.apply(self.init_weights)

    def init_weights(self, layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(layer.weight)
            if hasattr(layer, "bias") and layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)
        else:
            pass

    def forward(self, x):
        x = F.adaptive_avg_pool2d(self.conv_blocks(x), output_size=1)
        x = torch.flatten(x, 1)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_type="avg"):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        if pool_type == "avg":
            self.pool = nn.AvgPool2d(2)
        elif pool_type == "max":
            self.pool = nn.MaxPool2d(2)
        elif pool_type == "none":
            self.pool = nn.Identity()
        else:
            raise ValueError("Unexpected pool_type argument: {}".format(self.pool_type))

    def forward(self, x):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return x
"""

if __name__ == "__main__":
    # Model summary
    model = AudioNet()
    print(model)

    # Forward passing test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    x = torch.rand((128, 1, 224, 224)).to(device)
    y = model(x)
    print("Output shape: {}".format(y.shape))
    print("Forward passing test succeeded.")

    # Backward propgation test
    y_gt = torch.rand((128, 512)).to(device)
    loss = nn.MSELoss()(y, y_gt)
    loss.backward()
    print("Backward propagation test succeeded.")

    # Number of parameters
    print(
        "Number of parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
