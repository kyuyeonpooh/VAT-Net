import argparse
import os
import random
import shutil
import subprocess
from functools import partial
from itertools import product
from multiprocessing import Pool

from config import *


data_dir = config.vggsound.extract.data_dir
dest_dir = config.vggsound.extract.dest_dir
fps = config.vggsound.extract.fps
img_shape = config.vggsound.extract.img_shape
audio_sr = config.vggsound.extract.audio_sr
seed = config.vggsound.extract.seed
remove_fail = config.vggsound.extract.remove_fail

parser = argparse.ArgumentParser()
parser.add_argument("--ncpu", type=int, default=8,
                    help="Number of processes to be forked for multi-processing")
args = parser.parse_args()
n_cpu = args.ncpu


def train_val_split(train_list, val_size=0.1):
    train_len = len(train_list)
    val_len = int(val_size * train_len)
    random.seed(2020)
    random.shuffle(train_list)
    return train_list[:-val_len], train_list[-val_len:]


def extract_image(dirname, vidname):
    if dirname not in ["train", "val", "test"]:
        raise ValueError("Unknown type of directory name: {}".format(dirname))    
    in_dir = "train" if dirname == "val" else dirname
    out_dir = dirname

    """ The command below extract one frame per second from the video.
    We observed that extracted frames come from the mid-point of each second.
    For example, <youtube_id>.<start_sec>.05.jpg is a frame that approximately
    comes from 4.5 second timestamp.
    For another example, <youtube_id>.<start_sec>.09.jpg comes from 8.5 second
    timestamp.
    Thus, the corresponding 3-second audio interval of <youtube_id>.<start_sec>.N.jpg
    is from N - 2 to N + 1 second, whose mid-point is N - 0.5 second.
    """
    try:
        vid_id = vidname.rsplit(".", 1)[0]
        in_path = os.path.join(data_dir, in_dir, vidname)
        if not os.path.exists(in_path):
            raise FileNotFoundError
        out_dir = os.path.join(dest_dir, out_dir, "image", vid_id)
        out_path = os.path.join(out_dir, "{}.%02d.jpg".format(vid_id))
        os.makedirs(out_dir, exist_ok=True)
        ret = subprocess.call([
                "ffmpeg",
                "-y",                                # Overwrite without asking
                "-i", in_path,                       # Input video path
                "-vf", "fps={}".format(fps),         # Frame to extract per second
                "-s", "{}x{}".format(img_shape[0], img_shape[1]),   # Output image size
                "-loglevel", "error",
                "-hide_banner",
                "--", out_path                       # Output image path
            ])
        if ret != 0:
            raise RuntimeError("Unable to extract frames from {}".format(vidname))
    except FileNotFoundError:
        print("Video {} is not found or already deleted.".format(vidname))
        return False
    except Exception as e:
        shutil.rmtree(out_dir, ignore_errors=True)
        print("Removed image directory of {}".format(vidname))        
        if remove_fail and os.path.exists(in_path):
            os.remove(in_path)
            print("Removed video {}".format(vidname))
        return False

    return True


def extract_audio(dirname, vidname):
    if dirname not in ["train", "val", "test"]:
        raise ValueError("Unknown type of directory name: {}".format(dirname))    
    in_dir = "train" if dirname == "val" else dirname
    out_dir = dirname

    try:
        vid_id = vidname.rsplit(".", 1)[0]
        in_path = os.path.join(data_dir, in_dir, vidname)
        if not os.path.exists(in_path):
            raise FileNotFoundError
        out_dir = os.path.join(dest_dir, out_dir, "audio")
        out_path = os.path.join(out_dir, vidname.replace(".mp4", ".wav"))
        os.makedirs(out_dir, exist_ok=True)
        ret = subprocess.call([
                "ffmpeg",
                "-y",                   # Overwrite without asking
                "-i", in_path,          # Input video path
                "-ac", "1",             # Number of audio channel (mono)
                "-ar", str(audio_sr),   # Sampling rate (44,100 Hz)
                "-vn",                  # Save audio only
                "-loglevel", "error",
                "-hide_banner",
                "--", out_path          # Output audio path
            ])
        if ret != 0:
            raise RuntimeError("Unable to extract audio from {}".format(vidname))
    except FileNotFoundError:
        print("Video {} is not found or already deleted.".format(vidname))
        return False
    except Exception as e:
        # Corresponding image directory also should be deleted for synchronization
        img_dir = os.path.join(dest_dir, dirname, "image", vid_id)
        shutil.rmtree(img_dir, ignore_errors=True)
        print("Removed image directory of {}".format(vidname))
        if os.path.exists(out_path):
            os.remove(out_path)
        print("Removed audio of {}".format(vidname))
        if remove_fail and os.path.exists(in_path):
            os.remove(in_path)
            print("Removed video {}".format(vidname))
        return False

    return True


if __name__ == "__main__":
    dirlist_1 = ["train", "val", "test"]
    dirlist_2 = ["image", "audio"]

    # Prepare train and validation split
    vid_train = sorted(os.listdir(os.path.join(data_dir, "train")))
    vid_test = sorted(os.listdir(os.path.join(data_dir, "test")))
    vid_train, vid_val = train_val_split(vid_train, 0.1)
    vid_train, vid_val = sorted(vid_train), sorted(vid_val)
    
    # Map directory name with video list or extracting function
    dir_map = dict(zip(["train", "val", "test"], [vid_train, vid_val, vid_test]))
    func_map = dict(zip(["image", "audio"], [extract_image, extract_audio]))
    
    # Run extraction
    fail_count = 0
    with Pool(n_cpu) as pool:
        for dir_i, dir_j in product(dirlist_1, dirlist_2):
            print("Starting {} extraction of {} set...".format(dir_j, dir_i))
            for i, success in enumerate(pool.imap(partial(func_map[dir_j], dir_i), dir_map[dir_i])):
                print("Progress: {} / {}".format(i + 1, len(vid_train)), end="\r")
                fail_count += 1 if not success else 0
            print("\nFinished {} extraction of {} set.".format(dir_j, dir_i))
    
    print("Extraction completed! (Failed: {})".format(fail_count))
