import csv
import os
import random
import pickle
import re

import nltk
import librosa
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms



class MSRVTT(Dataset):
    def __init__(self, mode):
        if mode in ["test"]:
            self.mode = mode
        else:
            raise ValueError("Unknown type of dataset mode: {}".format(mode))
        
        data_dir = os.path.join("../ssd/MSR-VTT/proc", mode)
        self.img_dir = os.path.join(data_dir, "image")
        self.aud_dir = os.path.join(data_dir, "audio")

        self.img_list, self.aud_list, self.txt_list = self._load_csv()
        assert len(self.img_list) == len(self.aud_list) == len(self.txt_list)
        self.length = len(self.img_list)
        
        self.n_clips = 4
        self.stop_words = nltk.corpus.stopwords.words("english")
        with open("../ssd/CocoFlickr-vocabulary.pkl", "rb") as vocab_pkl:
            self.vocabulary = pickle.load(vocab_pkl)
        self.max_words = 16
        
        self.image_transforms = self._get_image_transforms()
        self.audio_transforms = self._get_audio_transforms()

    def _load_csv(self):
        img_list, aud_list, txt_list = list(), list(), list()
        with open("../ssd/MSR-VTT/msrvtt_test_new.csv") as msrvtt_csv:
            reader = csv.DictReader(msrvtt_csv, fieldnames=["key", "vid_key", "video_id", "sentence"], skipinitialspace=True)
            for i, row in enumerate(reader):
                img_list.append(row["video_id"])
                aud_list.append(row["video_id"] + ".wav")
                txt_list.append(row["sentence"])
        return img_list, aud_list, txt_list

    def _get_image_transforms(self):
        image_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])
        ])
        return image_transforms
    
    def _get_audio_transforms(self):
        audio_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[-40.], std=[40.])
        ])
        return audio_transforms

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
        # Get intervals
        last_frame = sorted(os.listdir(os.path.join(self.img_dir, self.img_list[i])))[-1]
        last_frame = float(last_frame.split(".")[1])
        intervals = np.linspace(2.0, last_frame - 2., self.n_clips)

        # Get images
        images = torch.zeros(self.n_clips, 3, 224, 224)
        for k, interval in enumerate(intervals):
            t = int(interval)
            img_file = "{}.{:02d}.jpg".format(self.img_list[i], t)
            img_path = os.path.join(self.img_dir, self.img_list[i], img_file)
            img = Image.open(img_path)
            img = self.image_transforms(img)
            assert img.shape == (3, 224, 224)
            images[k] = img
        
        # Get audio spectrograms
        aud_file = self.aud_list[i]
        aud_path = os.path.join(self.aud_dir, self.aud_list[i])
        y, sr = librosa.load(aud_path, sr=44100)
        audios = torch.zeros(self.n_clips, 1, 128, 301)
        for k, interval in enumerate(intervals):
            t = (interval - 0.5 - 1.5, interval - 0.5 + 1.5)
            y_t = y[int(sr * t[0]) : int(sr * t[0]) + sr * 3]
            aud = librosa.feature.melspectrogram(y=y_t, sr=sr, n_fft=1024, win_length=882, hop_length=441, n_mels=128)
            aud = librosa.amplitude_to_db(aud, ref=np.max)
            aud = self.audio_transforms(aud)
            assert aud.shape == (1, 128, 301)
            audios[k] = aud        

        # Get caption
        caption = self.txt_list[i]
        caption_words = self._tokenize(caption)
        caption_words = self._remove_stopwords(caption_words)
        txt = [self.vocabulary.get(word, 1) for word in caption_words]
        txt = txt[:self.max_words]
        if len(txt) < self.max_words:
            txt += [0] * (self.max_words - len(txt))
        txt = torch.LongTensor(txt)
        assert len(txt) == self.max_words

        return images, audios, txt, caption
