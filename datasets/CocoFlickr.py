import os
import random
import re
import pickle

import gensim
import nltk
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import config


class CocoFlickr(Dataset):
    def __init__(self, mode):
        if mode in ["train", "val"]:
            self.mode = mode
        else:
            raise ValueError("Unknown type of dataset mode: {}".format(mode))
        
        if mode == "train":
            self.img_dir = {
                "coco": config.cocoflickr.coco_train_images,
                "flickr": config.cocoflickr.flickr_images
            }
        else:
            self.img_dir = {
                "coco": config.cocoflickr.coco_val_images
            }

        with open(config.cocoflickr.annotations, "rb") as ann_file:
            annotations = pickle.load(ann_file)
            self.annotations = annotations[self.mode]

        if mode == "train":
            img_list = os.listdir(self.img_dir["coco"]) + os.listdir(self.img_dir["flickr"])
        else:
            img_list = os.listdir(self.img_dir["coco"])
        assert len(img_list) == len(self.annotations)
        self.length = len(img_list)

        self.image_transforms = self._get_image_transforms()
        self.stop_words = nltk.corpus.stopwords.words("english")
        with open(config.cocoflickr.vocabulary, "rb") as vocab_pkl:
            self.vocabulary = pickle.load(vocab_pkl)
        self.max_words = config.cocoflickr.max_words

    def _get_image_transforms(self):
        if self.mode == "train":
            image_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.ColorJitter(brightness=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])
            ])
        else:
            image_transforms = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])
            ])
        return image_transforms

    def _tokenize(self, caption):
        caption = re.sub("[^a-zA-z]", " ", caption).lower().strip()
        while " " * 2 in caption:
            caption = caption.replace(" " * 2, " ")
        words = caption.split()
        return words

    def _remove_stopwords(self, words):
        words = filter(lambda word: word not in self.stop_words, words)
        return list(words)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, i):
        # Pick single random caption with corresponding image
        n_captions = len(self.annotations[i]["caption"])
        if self.mode == "train":
            j = random.randint(0, n_captions - 1)
        else:
            j = 0
        
        # Get caption
        # if self.mode == "train":
        caption = self.annotations[i]["caption"][j]
        caption_words = self._tokenize(caption)
        caption_words = self._remove_stopwords(caption_words)
        txt = [self.vocabulary.get(word, 1) for word in caption_words]
        txt = txt[:self.max_words]
        if len(txt) < self.max_words:
            txt += [0] * (self.max_words - len(txt))
        txt = torch.LongTensor(txt)
        assert len(txt) == self.max_words
        """
        else:
            caption = list()
            txt = list()
            for j in range(n_captions):
                caption_ = self.annotations[i]["caption"][j]
                caption.append(caption_)
                caption_words = self._tokenize(caption_)
                caption_words = self._remove_stopwords(caption_words)
                txt_ = [self.vocabulary.get(word, 1) for word in caption_words]
                txt_ = txt_[:self.max_words]
                if len(txt_) < self.max_words:
                    txt_ += [0] * (self.max_words - len(txt_))
                assert len(txt_) == self.max_words
                txt += txt_
            txt = torch.LongTensor(txt)
        """

        # Get image
        dataset = self.annotations[i]["dataset"]
        img_file = self.annotations[i]["image_file"]
        img_path = os.path.join(self.img_dir[dataset], img_file)
        
        img = Image.open(img_path)
        img = img.convert("RGB")
        img = self.image_transforms(img)
        assert img.shape == (3, 224, 224)

        return img, txt, caption