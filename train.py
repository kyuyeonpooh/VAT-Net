import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam

from datasets.VGGSound import VGGSound
from models.TrimodalNet import TrimodalNet


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU for training:", torch.cuda.get_device_name())
    else:
        print("Failed to find GPU, using CPU instead.")
        device = torch.device("cpu")
    print("Current device:", device)

    model = TrimodalNet()
    model.to(device)

    run_name = "first"
    batch_size = 128
    lr = 1e-4
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-7)
    epochs = 400
    num_workers = 7
    log_step = 100
    ckpt_dir = os.path.join("save", run_name)

    vggsound_ = VGGSound(mode="train")
    vggsound = DataLoader(vggsound_, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)

    writer = SummaryWriter(os.path.join("runs", run_name))
    global_step = 0

    for epoch in range(1, epochs + 1):
        print("Epoch", epoch)

        model.train()
        train_loss, train_crt = 0., 0
        for i, (img, aud, label) in enumerate(vggsound):
            optimizer.zero_grad()
            img, aud, label = img.to(device), aud.to(device), label.to(device)
            pred = model(img, aud)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                train_loss += loss.item()
                binary_pred = (pred >= 0.5).float()
                train_crt += (binary_pred == label).sum().item()

            if (i + 1) % log_step == 0:
                model.eval()
                global_step += log_step
                train_loss /= log_step
                train_acc = train_crt / (batch_size * log_step)
                
                writer.add_scalar("train_loss", train_loss, global_step=global_step)
                writer.add_scalar("train_acc", train_acc, global_step=global_step)
                print("train_loss: {:.4f}, train_acc: {:.4f}".format(train_loss, train_acc))

                train_loss, train_crt = 0., 0
                model.train()
        
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "{:03d}.pt".format(epoch)))
