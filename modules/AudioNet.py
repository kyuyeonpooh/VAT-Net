import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class AudioNet(nn.Module):
    def __init__(self):
        super(AudioNet, self).__init__()

        conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(conv1.weight, mode="fan_out", nonlinearity="relu")
        
        resnet18 = models.resnet18(pretrained=True)
        resnet18_modules = [conv1] + list(resnet18.children())[1:-1]
        self.resnet18 = nn.Sequential(*resnet18_modules)
        self.fc_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256)
        )

    def forward(self, x):
        x = self.resnet18(x).squeeze()
        x = self.fc_head(x)
        x = F.normalize(x, p=2, dim=1) 
        return x


if __name__ == "__main__":
    # Model summary
    model = AudioNet()
    print(model)

    # Forward passing test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    x = torch.rand((64, 1, 224, 224)).to(device)
    y = model(x)
    print("Output shape: {}".format(y.shape))
    print("Forward passing test succeeded.")

    # Backward propgation test
    y_gt = torch.rand((64, 256)).to(device)
    loss = nn.MSELoss()(y, y_gt)
    loss.backward()
    print("Backward propagation test succeeded.")

    # Number of parameters
    print(
        "Number of parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
