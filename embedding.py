import csv
import os
from itertools import compress

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import *
from datasets.VGGSound import VGGSound
from modules.VANet import VANet

target_classes = ["toilet flushing",
                "machine gun shooting",
                "child speech, kid speaking",
                "playing violin, fiddle",
                "playing hammond organ",
                "playing electric guitar",
                "driving motorcycle",
                "playing banjo",
                "playing cello",
                "rowboat, canoe, kayak rowing",
                "vehicle horn, car horn, honking",
                "playing piano",
                "playing drum kit",
                "railroad car, train wagon",
                "people whistling",
                "male speech, man speaking",
                "playing acoustic guitar",
                "motorboat, speedboat acceleration",
                "engine accelerating, revving, vroom",
                "female singing",
                "playing clarinet",
                "female speech, woman speaking",
                "police car (siren)",
                "race car, auto racing",
                "people clapping",
                "goose honking",
                "dog barking",
                "dog howling",
                "cat purring",
                "waterfall burbling",
                "ocean burbling",
                "splashing water",
                "duck quacking"]

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name())
else:
    raise RuntimeError("GPU is required")
print("Current device:", device)

batch_size = 512
num_workers = 16
model_path = "../ssd/save/VA_B512_LR1e-04_D1e-05_M0.2_scratch/021.pt"

writer = SummaryWriter("../ssd/runs/bin/embedding-v2-021-scratch")

vggsound_ = VGGSound(mode="test")
vggsound_test = DataLoader(vggsound_, batch_size=batch_size, num_workers=num_workers, pin_memory=True)


def load_model():
    load = torch.load(model_path)
    model = VANet()
    model = nn.DataParallel(model)
    model.to(device)
    model.load_state_dict(load["weights"])
    return model

def generate_embeddings():
    model = load_model()
    model.eval()

    img_embs = None
    aud_embs = None
    classes = list()

    for img, aud, class_ in tqdm(vggsound_test):
        with torch.no_grad():
            img, aud = img.to(device), aud.to(device)
            img_emb, aud_emb = model(img, aud)

            if img_embs is None:
                img_embs = img_emb.cpu().clone()
            else:
                img_embs = torch.cat((img_embs, img_emb.cpu().clone()))
            if aud_embs is None:
                aud_embs = aud_emb.cpu().clone()
            else:
                aud_embs = torch.cat((aud_embs, aud_emb.cpu().clone()))
            classes += list(class_)
    
    return img_embs, aud_embs, classes

if __name__ == "__main__":
    img_embs, aud_embs, classes = generate_embeddings()
    
    target_idx = [c in target_classes for c in classes]
    img_embs, aud_embs = img_embs[target_idx], aud_embs[target_idx]
    classes = list(compress(classes, target_idx))

    writer.add_embedding(img_embs, tag="embedding-img-target", metadata=classes)
    writer.add_embedding(aud_embs, tag="embedding-aud-target", metadata=classes)
    
    global_embs = torch.cat((img_embs, aud_embs))
    img_classes = [c + "(img)" for c in classes]
    aud_classes = [c + "(aud)" for c in classes]
    writer.add_embedding(global_embs, tag="embedding-all-target", metadata=img_classes + aud_classes)
    
    torch.save({"img_emb": img_embs, "aud_emb": aud_embs, "class": classes}, "emb.pt")