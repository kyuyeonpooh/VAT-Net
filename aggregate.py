import csv
import os
import json
import pickle

from config import config

coco_train_dict = dict()
coco_val_dict = dict()
flickr_dict = dict()

train_dict = dict()
val_dict = dict()


# COCO 2017 Captioning Dataset (Train)
with open(config.cocoflickr.coco_train_captions) as coco_train_json:
    coco_train = json.load(coco_train_json)
annotations = coco_train["annotations"]

for ann in annotations:
    image_id, caption = ann["image_id"], ann["caption"]
    if image_id not in coco_train_dict:
        coco_train_dict[image_id] = [caption]
    else:
        coco_train_dict[image_id].append(caption)

for i, key in enumerate(sorted(coco_train_dict.keys())):
    train_dict[i] = {
        "dataset": "coco",
        "image_file": "{:012d}.jpg".format(key),
        "caption": coco_train_dict[key]
    }


# COCO 2017 Captioning Dataset (Validations)
with open(config.cocoflickr.coco_val_captions) as coco_val_json:
    coco_val = json.load(coco_val_json)
annotations = coco_val["annotations"]

for ann in annotations:
    image_id, caption = ann["image_id"], ann["caption"]
    if image_id not in coco_val_dict:
        coco_val_dict[image_id] = [caption]
    else:
        coco_val_dict[image_id].append(caption)

for i, key in enumerate(sorted(coco_val_dict.keys())):
    val_dict[i] = {
        "dataset": "coco",
        "image_file": "{:012d}.jpg".format(key),
        "caption": coco_val_dict[key]
    }


# Flickr30k Dataset
with open(config.cocoflickr.flickr_captions) as flickr_train_csv:
    reader = csv.DictReader(flickr_train_csv, delimiter="|", fieldnames=["image_file", "number", "caption"], skipinitialspace=True)
    for i, row in enumerate(reader):
        if i == 0:
            continue
        image_file, caption = row["image_file"], row["caption"]
        image_id = int(image_file.split(".")[0])
        if image_id not in flickr_dict:
            flickr_dict[image_id] = [caption]
        else:
            flickr_dict[image_id].append(caption)

start_len = len(train_dict)
for i, key in enumerate(sorted(flickr_dict.keys())):
    train_dict[start_len + i] = {
        "dataset": "flickr",
        "image_file": "{}.jpg".format(key),
        "caption": flickr_dict[key]
    }

# Combine train and validation annotations
annotations = {"train": train_dict, "val": val_dict}
with open(config.cocoflickr.annotations, "wb") as pkl:
    pickle.dump(annotations, pkl, pickle.HIGHEST_PROTOCOL)