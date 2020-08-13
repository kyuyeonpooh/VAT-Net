import csv
import os
import json
import pickle

coco_dict = dict()
flickr_dict = dict()
aggregated_dict = dict()


# COCO 2017 Captioning Dataset
with open("../ssd/COCO2017/annotations/captions_train2017.json") as coco_train_json:
    coco_train = json.load(coco_train_json)
annotations = coco_train["annotations"]

for ann in annotations:
    image_id, caption = ann["image_id"], ann["caption"]
    if image_id not in coco_dict:
        coco_dict[image_id] = [caption]
    else:
        coco_dict[image_id].append(caption)

for i, key in enumerate(sorted(coco_dict.keys())):
    aggregated_dict[i] = {"dataset": "coco",
                          "image_file": "{:012d}.jpg".format(key),
                          "caption": coco_dict[key]}


# Flickr30k Dataset
with open("../ssd/Flickr30k/results.csv") as flickr_train_csv:
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

start_len = len(aggregated_dict)
for i, key in enumerate(sorted(flickr_dict.keys())):
    aggregated_dict[start_len + i] = {"dataset": "flickr",
                                      "image_file": "{}.jpg".format(key),
                                      "caption": flickr_dict[key]}

with open("../ssd/CocoFlickr.pkl", "wb") as pkl:
    pickle.dump(aggregated_dict, pkl, pickle.HIGHEST_PROTOCOL)