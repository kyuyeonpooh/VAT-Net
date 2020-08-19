import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import config
from datasets.CocoFlickr import CocoFlickr
from modules.VTNet import VTNet

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name())
else:
    raise RuntimeError("GPU is required")
print("Current device:", device)

batch_size = 64
num_workers = 16
model_path = "../ssd/save/VT_B512_LR1e-04_D1e-05_M0.2/026.pt"

writer = SummaryWriter("../ssd/runs/bin/embedding-vt-026")

cocoflickr_ = CocoFlickr(mode="val")
cocoflickr_val = DataLoader(cocoflickr_, batch_size=batch_size, num_workers=num_workers, pin_memory=True)


def load_model():
    load = torch.load(model_path)
    model = VTNet()
    model = nn.DataParallel(model)
    model.to(device)
    model.load_state_dict(load["weights"])
    return model

def generate_embeddings():
    model = load_model()
    model.eval()

    img_embs = None
    txt_embs = None
    captions = list()

    for img, txt, caption in tqdm(cocoflickr_val):
        with torch.no_grad():
            img, txt = img.to(device), txt.to(device)
            img_emb, txt_emb = model(img, txt)

            if img_embs is None:
                img_embs = img_emb.cpu().clone()
            else:
                img_embs = torch.cat((img_embs, img_emb.cpu().clone()))
            if txt_embs is None:
                txt_embs = txt_emb.cpu().clone()
            else:
                txt_embs = torch.cat((txt_embs, txt_emb.cpu().clone()))
            captions += list(caption)
    
    return img_embs, txt_embs, captions

if __name__ == "__main__":
    img_embs, txt_embs, captions = generate_embeddings()

    writer.add_embedding(img_embs, tag="embedding-img-target", metadata=captions)
    writer.add_embedding(txt_embs, tag="embedding-txt-target", metadata=captions)
    
    global_embs = torch.cat((img_embs, txt_embs))
    img_classes = [c.strip() + "(img)" for c in captions]
    txt_classes = [c.strip() + "(txt)" for c in captions]
    writer.add_embedding(global_embs, tag="embedding-all-target", metadata=img_classes + txt_classes)
    
    torch.save({"img_emb": img_embs, "txt_emb": txt_embs, "captions": captions}, "emb_vt.pt")