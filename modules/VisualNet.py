import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class VisualNet(nn.Module):
    def __init__(self):
        super(VisualNet, self).__init__()   

        resnet18 = models.resnet18(pretrained=True)
        self.resnet18 = nn.Sequential(*list(resnet18.children())[:-1])
        self.fc_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256)
        )

    def forward(self, x):
        x = self.resnet18(x)
        x = x.flatten(1)
        x = self.fc_head(x)
        x = F.normalize(x, p=2, dim=1)
        
        return x


if __name__ == "__main__":
    # Model summary
    model = VisualNet()
    print(model)

    # Forward passing test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    x = torch.rand((64, 3, 224, 224)).to(device)
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
