import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class VisualNet(nn.Module):
    """ VisualNet is following ResNet18 architecture.
    """
    def __init__(self):
        super(VisualNet, self).__init__()
        
        self.resnet50 = models.resnet50(pretrained=False)
        self.resnet50.fc = nn.Identity()  # Discard last fc layer

    def forward(self, x):
        return self.resnet50(x)  # (B, 2024)


if __name__ == "__main__":
    # Model summary
    model = VisualNet()
    print(model)

    # Forward passing test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    x = torch.rand((128, 3, 224, 224)).to(device)
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
