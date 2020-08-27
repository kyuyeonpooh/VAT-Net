import argparse
import os
import random
import shutil
import subprocess
from functools import partial
from itertools import product
from multiprocessing import Pool

from config import *


data_dir = "../ssd/MSR-VTT/raw"
dest_dir = "../ssd/MSR-VTT/proc"
fps = 1
img_shape = (256, 256)
audio_sr = 44100
# seed = config.vggsound.extract.seed
# remove_fail = config.vggsound.extract.remove_fail

parser = argparse.ArgumentParser()
parser.add_argument("--ncpu", type=int, default=24,
                    help="Number of processes to be forked for multi-processing")
args = parser.parse_args()
n_cpu = args.ncpu


def extract_image(dirname, vidname):
    if dirname not in ["test"]:
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
        return False

    return True


def extract_audio(dirname, vidname):
    if dirname not in ["test"]:
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
        return False

    return True


if __name__ == "__main__":
    dirlist_1 = ["test"]
    dirlist_2 = ["image", "audio"]

    # Prepare train and validation split
    vid_test = sorted(os.listdir(os.path.join(data_dir, "test")))
    
    # Map directory name with video list or extracting function
    dir_map = dict(zip(["test"], [vid_test]))
    func_map = dict(zip(["image", "audio"], [extract_image, extract_audio]))
    
    # Run extraction
    fail_count = 0
    with Pool(n_cpu) as pool:
        for dir_i, dir_j in product(dirlist_1, dirlist_2):
            print("Starting {} extraction of {} set...".format(dir_j, dir_i))
            for i, success in enumerate(pool.imap(partial(func_map[dir_j], dir_i), dir_map[dir_i])):
                print("Progress: {} / {}".format(i + 1, len(dir_map[dir_i])), end="\r")
                fail_count += 1 if not success else 0
            print("\nFinished {} extraction of {} set.".format(dir_j, dir_i))
    
    print("Extraction completed! (Failed: {})".format(fail_count))
