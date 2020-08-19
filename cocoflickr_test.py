import matplotlib.pyplot as plt

from datasets import CocoFlickr
from torch.utils.data import DataLoader

batch_size = 1
dataset = CocoFlickr.CocoFlickr(mode="train")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for (img, txt, caption) in dataloader:
    print(img.shape)
    print(txt)
    print(caption)
    break