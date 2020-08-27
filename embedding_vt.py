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
from modules.VATNet import VATNet

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name())
else:
    raise RuntimeError("GPU is required")
print("Current device:", device)

batch_size = 1
num_workers = 16
model_path = "../ssd/save/VAT_B512_LR1e-04_D1e-05_M0.2_easy/033.pt"

writer = SummaryWriter("../ssd/runs/bin/emb_coco_vat_v1")

cocoflickr_ = CocoFlickr(mode="val")
cocoflickr_val = DataLoader(cocoflickr_, batch_size=batch_size, num_workers=num_workers, pin_memory=True)


def load_model():
    load = torch.load(model_path)
    model = VATNet()
    
    model.load_state_dict(load["weights"])
    model = nn.DataParallel(model)
    model.to(device)
    return model

def generate_embeddings():
    model = load_model()
    model.eval()

    img_embs = None
    txt_embs = None
    img_idx = list()
    captions = list()
    cap2idx = dict()
    idx2cap = dict()

    for i, (img, txt, caption) in enumerate(tqdm(cocoflickr_val)):
        with torch.no_grad():
            txt = txt.view(-1, 16)
            img = img.repeat(txt.shape[0], 1, 1, 1)
            img, txt = img.to(device), txt.to(device)
            img_emb, txt_emb = model(img, txt, mode="vt")

            if img_embs is None:
                img_embs = img_emb[0].cpu().clone()
            else:
                img_embs = torch.cat((img_embs, img_emb[0].cpu().clone()))
            if txt_embs is None:
                txt_embs = txt_emb.cpu().clone()
            else:
                txt_embs = torch.cat((txt_embs, txt_emb.cpu().clone()))
            
            img_idx.append(i)
            idx2cap[i] = caption
            for c in caption:
                cap2idx[c[0]] = i
                captions.append(c[0])
    
    return img_embs, txt_embs, captions, img_idx, cap2idx, idx2cap

if __name__ == "__main__":
    img_embs, txt_embs, captions, img_idx, cap2idx, idx2cap = generate_embeddings()
    """
    writer.add_embedding(img_embs, tag="embedding-img-target", metadata=captions)
    writer.add_embedding(txt_embs, tag="embedding-txt-target", metadata=captions)
    
    global_embs = torch.cat((img_embs, txt_embs))
    img_classes = ["[IMG]" + c.strip() for c in captions]
    txt_classes = ["[TXT]" + c.strip() for c in captions]
    writer.add_embedding(global_embs, tag="embedding-all-target", metadata=img_classes + txt_classes)
    """    
    torch.save({"img_emb": img_embs, "txt_emb": txt_embs, "img_idx": img_idx, "captions": captions, "cap2idx": cap2idx, "idx2cap": idx2cap}, "emb_coco_vat_v1.pt")