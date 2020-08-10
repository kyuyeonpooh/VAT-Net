import argparse
import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import config
from datasets.VGGSound import VGGSound
from modules.VANet import VANet
from modules.CosineTripletLoss import CosineTripletLoss

ckpt_dir = config.train.ckpt_dir
tensorboard_dir = config.train.tensorboard_dir

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, choices=["VA", "VT", "VAT"])
parser.add_argument("--epoch", type=int, default=80)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--learning-rate", type=float, default=1e-4)
parser.add_argument("--weight-decay", type=float, default=1e-5)
parser.add_argument("--margin", type=float, default=0.2,
                    help="Margin for triplet loss")
parser.add_argument("--num-workers", type=int, default=16,
                    help="Number of processes to be forked for batch generation")
parser.add_argument("--log-step", type=int, default=100,
                    help="Logging period on tensorboard")
parser.add_argument("--use-tensorboard", type=int, default=1, choices=[0, 1],
                    help="Whether to use tensorboard for logging")
parser.add_argument("--use-scheduler", type=int, default=1, choices=[0, 1],
                    help="Whether to use learning rate step scheduler")
parser.add_argument("--resume", type=int, default=0, choices=[0, 1],
                    help="Whether to resume from the checkpoint if exists")
args = parser.parse_args()

model_name = args.model
epochs = args.epoch
batch_size = args.batch_size
lr = args.learning_rate
weight_decay = args.weight_decay
margin = args.margin
num_workers = args.num_workers
log_step = args.log_step
use_tensorboard = args.use_tensorboard
use_scheduler = args.use_scheduler
resume = args.resume
train_name = "{}_B{}_LR{:.0e}_D{:.0e}_M{}".format(
              model_name, batch_size, lr, weight_decay, margin)


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name())
    else:
        raise RuntimeError("GPU is required")
    return device


def get_model():
    if model_name == "VA":
        model = VANet()
    else:
        raise ValueError("Error")
    return model


def get_tensorboard():
    tensorboard_dir_ = os.path.join(tensorboard_dir, train_name)
    if os.path.exists(tensorboard_dir_):
        print("Warning: tensorboard directory of current run already exists!")
    tensorboard = SummaryWriter(tensorboard_dir_)
    return tensorboard


def get_ckpt_weight(ckpt_dir_):
    ckpt_dir_ = os.path.join(ckpt_dir, train_name)
    if not os.path.exists(ckpt_dir_):
        print("Checkpoint not found")
        os.makedirs(ckpt_dir_)
        return False, 0, None
    ckpt_list = sorted(os.listdir(ckpt_dir_))
    if len(ckpt_list) == 0:
        print("Checkpoint not found")
        return False, 0, None
    model_path = os.path.join(ckpt_dir_, ckpt_list[-1])
    ckpt_bundle = torch.load(model_path)
    ckpt_epoch, ckpt_weights = ckpt_bundle["epoch"], ckpt_bundle["weights"]
    print("Found checkpoint, starting from epoch", ckpt_epoch)
    return True, ckpt_epoch, ckpt_weights


def get_data_loader(mode):
    if not mode in ["train", "val"]:
        raise ValueError("Unexpected argument while getting dataset: {}".format(mode))
    if model_name == "VA":
        dataset = VGGSound(mode=mode)
        drop_last = True if mode == "train" else False
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last,
                                num_workers=num_workers, pin_memory=True)
    else:
        raise ValueError("Error")
    return dataloader


def train():
    # Get device and model
    device = get_device()
    model = get_model()
    model = nn.DataParallel(model)
    model.to(device)
    
    # Get optimizer and loss function
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1) if use_scheduler else None
    criterion = CosineTripletLoss(margin=margin)

    # Get tensorboard
    global_step = 0
    tensorboard = get_tensorboard() if use_tensorboard else None

    # Starts from checkpoint if exists
    ckpt_dir_ = os.path.join(ckpt_dir, train_name)
    if resume:
        success, ckpt_epoch, ckpt_weight = get_ckpt_weight(ckpt_dir_)
        if success:
            model.load_state_dict(ckpt_weight)
        epoch_range = range(ckpt_epoch + 1, epochs + 1)
    else:        
        os.makedirs(ckpt_dir_, exist_ok=True)
        epoch_range = range(1, epochs + 1)

    # Get data loader
    train_loader = get_data_loader("train")
    val_loader = get_data_loader("val")

    # Train starts here
    for epoch in epoch_range:
        print("Epoch", epoch)

        model.train()
        train_loss = 0.
        for i, (v, a, _) in enumerate(train_loader):
            optimizer.zero_grad()
            v, a  = v.to(device), a.to(device)
            z_v, z_a = model(v, a)
            loss = criterion(z_v, z_a)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if (i + 1) % log_step == 0:
                global_step += log_step
                train_loss /= log_step

                model.eval()
                val_loss = validate(model, criterion, val_loader, device)
                log_progress(tensorboard, train_loss, val_loss, global_step)

                train_loss = 0.
                model.train()
        
        if scheduler is not None:
            scheduler.step()
        torch.save({"epoch": epoch, "weights": model.state_dict()}, os.path.join(ckpt_dir_, "{:03d}.pt".format(epoch)))


def validate(model, criterion, val_loader, device):
    with torch.no_grad():
        val_loss = 0.
        for v, a, _ in val_loader:
            v, a = v.to(device), a.to(device)
            z_v, z_a = model(v, a)
            loss = criterion(z_v, z_a)
            val_loss += loss.item()
        val_loss /= len(val_loader)
    return val_loss

def log_progress(tensorboard, train_loss, val_loss, global_step):    
    if tensorboard is not None:
        tensorboard.add_scalar("train_loss", train_loss, global_step=global_step)
        tensorboard.add_scalar("val_loss", val_loss, global_step=global_step)
    print("train_loss: {:.4f}, val_loss: {:.4f}".format(train_loss, val_loss))


if __name__ == "__main__":
    train()