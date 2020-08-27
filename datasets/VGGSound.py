import csv
import os
import random

import gensim
import librosa
import nltk
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import config


def get_ytid_to_class_map():
    ytid_to_class = dict()
    with open(config.vggsound.download.csv) as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=["ytid", "start", "label", "split"], skipinitialspace=True)
        for row in reader:
            if row["ytid"] == "#":
                continue
            ytid_to_class[row["ytid"]] = row["label"]
    return ytid_to_class


def get_ytid(filename):
    return filename.split(".")[0]


class VGGSound(Dataset):
    def __init__(self, mode):
        if mode in ["train", "val", "test"]:
            self.mode = mode
        else:
            raise ValueError("Unknown type of dataset mode: {}".format(mode))
        
        data_dir = os.path.join(config.vggsound.extract.dest_dir, mode)
        self.img_dir = os.path.join(data_dir, "image")
        self.aud_dir = os.path.join(data_dir, "audio")

        self.img_list = sorted(os.listdir(self.img_dir))
        self.aud_list = sorted(os.listdir(self.aud_dir))
        assert len(self.img_list) == len(self.aud_list)
        self.length = len(self.img_list)

        img_id_list = self.img_list
        aud_id_list = [aud_file.rsplit(".", 1)[0] for aud_file in self.aud_list]
        assert set(img_id_list) == set(aud_id_list)
        self.ytid_to_class = get_ytid_to_class_map()

        self.image_transforms = self._get_image_transforms()
        self.audio_transforms = self._get_audio_transforms()

    def _get_image_transforms(self):
        if self.mode == "train":
            image_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(224),
                transforms.ColorJitter(brightness=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])
            ])
        else:
            image_transforms = transforms.Compose([
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

    def __len__(self):
        return self.length
    
    def __getitem__(self, i):
        """ With given video, a random timestamp is selected.
        The timestamp integer t is selected randomly from 2 to 8.

        If t is N, the image comes from N - 0.5 second timestamp of the video.
        Thus, the corresponding 3-second audio interval of image above is from
        2 to 5 second, derived by [4 - 0.5 - 1.5,  4 - 0.5 + 1.5].

        This is the reason why the minimum and maximum value of t is set to 2
        and 9. The mid-point timestamp t should be at least 2 to have first
        1.5-second interval, and at most 9 to have latter 1.5-second interval.
        However, we found that duration of some audios are less than 10 seconds,
        so the maximum timestamp is finally set to 8. 

        To generate negative pair, random audio interval from different video
        is chosen for given video frame.
        """
        # Pick single random frame with corresponding audio
        if self.mode == "train":
            t = random.randint(2, 8)
        else:
            t = 9 // 2
        img_file = "{}.{:02d}.jpg".format(self.img_list[i], t)
        aud_file = self.aud_list[i]
        img_path = os.path.join(self.img_dir, self.img_list[i], img_file)
        aud_path = os.path.join(self.aud_dir, self.aud_list[i])

        # Get image
        img = Image.open(img_path)
        img = self.image_transforms(img)

        # Get corresponding log mel-spectrogram of 3-second audio
        y, sr = librosa.load(aud_path, sr=44100)
        interval = (t - 0.5 - 1.5, t - 0.5 + 1.5)
        y = y[sr * int(interval[0]) : sr * int(interval[1])]
        aud = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, win_length=882, hop_length=441, n_mels=128)
        aud = librosa.amplitude_to_db(aud, ref=np.max)
        aud = self.audio_transforms(aud)
        if aud.shape != (1, 128, 301):
            raise RuntimeError("Unexpected shape of spectrogram made from" + aud_file)

        # Get class of given sample
        ytid = get_ytid(img_file)
        assert ytid == get_ytid(aud_file)
        class_ = self.ytid_to_class[ytid]

        return img, aud, class_
