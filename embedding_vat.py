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
from datasets.MSRVTT import MSRVTT
from modules.VATNet import VATNet

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name())
else:
    raise RuntimeError("GPU is required")
print("Current device:", device)

batch_size = 64
num_workers = 16
model_path = "../ssd/save/VAT_B512_LR1e-04_D1e-05_M0.4_versatile_loss_v2_detach/032.pt"

writer = SummaryWriter("../ssd/runs/bin/emb_msrvtt_vat_v4")

msrvtt = MSRVTT(mode="test")
msrvtt_loader = DataLoader(msrvtt, batch_size=batch_size, num_workers=num_workers, pin_memory=True)


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

    img_embs = []
    aud_embs = []
    txt_embs = []
    captions = []

    for i, (v, a, t, caption) in enumerate(tqdm(msrvtt_loader)):
        with torch.no_grad():
            v = v.view(-1, 3, 224, 224)
            a = a.view(-1, 1, 128, 301)
            z_v, z_a, z_t = model(v, (a, t), mode="vat")
            z_v = z_v.view(z_t.shape[0], 4, z_t.shape[1])
            z_v = z_v.mean(dim=1)
            z_a = z_a.view(z_t.shape[0], 4, z_t.shape[1])
            z_a = z_a.mean(dim=1)
            img_embs.append(z_v.cpu().numpy())
            aud_embs.append(z_a.cpu().numpy())
            txt_embs.append(z_t.cpu().numpy())
            captions += caption
    
    img_embs = np.concatenate(img_embs)
    aud_embs = np.concatenate(aud_embs)
    txt_embs = np.concatenate(txt_embs)

    return img_embs, aud_embs, txt_embs, captions

if __name__ == "__main__":
    img_embs, aud_embs, txt_embs, captions = generate_embeddings()
    
    writer.add_embedding(img_embs, tag="embedding-img", metadata=captions)
    writer.add_embedding(aud_embs, tag="embedding-aud", metadata=captions)
    writer.add_embedding(txt_embs, tag="embedding-txt", metadata=captions)
    
    global_embs = np.concatenate((img_embs, aud_embs, txt_embs))
    img_classes = ["[IMG]" + c.strip() for c in captions]
    aud_classes = ["[AUD]" + c.strip() for c in captions]
    txt_classes = ["[TXT]" + c.strip() for c in captions]
    writer.add_embedding(global_embs, tag="embedding-global", metadata=img_classes + aud_classes + txt_classes)

    torch.save({"img_emb": img_embs, "aud_emb": aud_embs, "txt_emb": txt_embs, "caption": captions}, "emb_msrvtt_vat_v4.pt")