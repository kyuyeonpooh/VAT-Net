import os

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam

from datasets.VGGSound import VGGSound
from modules.TrimodalNet import TrimodalNet


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name())
    else:
        raise RuntimeError("GPU is required")
    print("Current device:", device)

    model = TrimodalNet()
    model = nn.DataParallel(model)
    model.to(device)

    run_name = "lr_0.0001_decay_1e-07_sched_True"
    batch_size = 128
    criterion = nn.BCELoss()
    epochs = [80]
    num_workers = 16
    ckpt_dir = os.path.join("../ssd/save", run_name)

    vggsound_ = VGGSound(mode="test")
    vggsound = DataLoader(vggsound_, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    for epoch in epochs:
        writer = SummaryWriter(os.path.join("../ssd/runs", "test" + str(epoch)))
        print("Epoch", epoch)
        load = torch.load(os.path.join(ckpt_dir, "{:03d}.pt".format(epoch)))
        model.load_state_dict(load["weights"])

        model.eval()
        val_loss, val_crt, val_total = 0., 0, 0
        for i, (img, aud, label, _) in enumerate(tqdm(vggsound)):
            with torch.no_grad():
                img, aud, label = img.to(device), aud.to(device), label.to(device)         
                pred, z_v, z_a = model(img, aud)
                loss = criterion(pred, label)
                val_loss += loss.item()
                binary_pred = (pred >= 0.5).float()
                val_crt += (binary_pred == label).sum().item()
                val_total += label.size(0)
            if i % 50 == 0:
                writer.add_embedding(z_v, tag="z_v", global_step=i)
                writer.add_embedding(z_a, tag="z_a", global_step=i)
                writer.add_embedding(z_va, tag="z_va", global_step=i)
        val_loss /= len(vggsound)
        val_acc = val_crt / val_total
        print("test_loss: {:.4f}, test_acc: {:.4f}".format(val_loss, val_acc))
