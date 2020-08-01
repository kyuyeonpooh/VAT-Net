import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam

from datasets.VGGSound import VGGSound
from models.TrimodalNet import TrimodalNet


parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight-decay", type=float, default=1e-5)
parser.add_argument("--epochs", type=int, default=80)
parser.add_argument("--num-workers", type=int, default=16)
parser.add_argument("--use-scheduler", type=bool, default=True)
args = parser.parse_args()

if __name__ == "__main__":
    model = TrimodalNet()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name())
    else:
        raise RuntimeError("GPU is required")
    model = nn.DataParallel(model)
    model.to(device)

    run_name = "lr_{}_decay_{}_sched_{}".format(args.lr, args.weight_decay, args.use_scheduler)
    batch_size = 128

    lr = args.lr  
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        scheduler = None
    criterion = nn.BCELoss()

    epochs = args.epochs
    num_workers = args.num_workers

    log_step = 100
    ckpt_dir = os.path.join("../ssd/save", run_name)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join("../ssd/runs", run_name))
    logfile = open(os.path.join("../ssd/save", run_name + ".txt"), "w")
    global_step = 0

    vggsound_ = VGGSound(mode="train")
    vggsound = DataLoader(vggsound_, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)
    vggsound_val_ = VGGSound(mode="val")
    vggsound_val = DataLoader(vggsound_val_, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    
    for epoch in range(1, epochs + 1):
        print("Epoch", epoch)
        logfile.write("Epoch " + str(epoch) + "\n")
        logfile.flush()

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
                
                with torch.no_grad():
                    val_loss, val_crt, val_total = 0., 0, 0     
                    for j, (img, aud, label) in enumerate(vggsound_val):
                        if j == 32:
                            break
                        img, aud, label = img.to(device), aud.to(device), label.to(device)
                        pred = model(img, aud)
                        loss = criterion(pred, label)
                        val_loss += loss.item()
                        binary_pred = (pred >= 0.5).float()
                        val_crt += (binary_pred == label).sum().item()
                        val_total += label.size(0)
                # val_loss /= len(vggsound_val)
                val_loss /= 32
                val_acc = val_crt / val_total

                writer.add_scalar("train_loss", train_loss, global_step=global_step)
                writer.add_scalar("train_acc", train_acc, global_step=global_step)
                writer.add_scalar("val_loss", val_loss, global_step=global_step)
                writer.add_scalar("val_acc", val_acc, global_step=global_step)
                logstr = "train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(train_loss, train_acc, val_loss, val_acc)
                logfile.write(logstr + "\n")
                logfile.flush()
                print(logstr)

                train_loss, train_crt = 0., 0
                model.train()
        
        if scheduler is not None:
            scheduler.step()

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)
        torch.save({"epoch": epoch, "weights": model.state_dict()}, os.path.join(ckpt_dir, "{:03d}.pt".format(epoch)))

    logfile.close()