import csv
import os
import subprocess
import time
from collections import Counter

import youtube_dl

VGGSound_csv = os.path.join("VGGSound", "vggsound.csv")  # VGGSound csv file path
data_dir = os.path.join("VGGSound")                      # Directory where video will be temporally saved
# vid_dir = "../sgvrnas"                                 # Directory where video frames will be saved
# aud_dir = "../sgvrnas"                                 # Directory where audio will be saved
delay = 2                               # Delay between download (in second) to prevent IP ban
timeout = 100                           # Time limit for each subprocess
remove_vid = False                      # Remove video after extracting frames and audio
row_start = 0                           # CSV file row start point (starts from 0)
row_end = -1                            # CSV file row end point (exclusive, max row if -1)
start_time = time.time()                # Start point to measure the elapsed time

def load_csv(csvpath):
    csvfile = open(csvpath)
    reader = csv.DictReader(csvfile, fieldnames=["ytid", "start", "label", "split"], skipinitialspace=True)
    return csvfile, reader    


def download(row, ytid_counter, remove_vid=False):
    ytid = row["ytid"]      # YouTube video id
    start = row["start"]    # Start second
    # end = row["end"]      # End second
    split = row["split"]    # train or test spilt
    if split != "train" and split != "test":
        raise ValueError("Unknown value {} for key split.".format(split))

    ydl_opts = {
        "start_time": int(float(start)),
        "end_time": int(float(start) + 10.),
        "format": "mp4[height<=360]",
    }

    vid_file = "{}.{:05d}.{}".format(ytid, int(float(start)), "mp4")
    vid_file_path = os.path.join(data_dir, split, vid_file)
    
    """
    if ytid_counter[ytid] > 1:  # remove original and download both
        if os.path.isfile(vid_file_path):
            os.remove(vid_file_path)
        vid_file = "{}.{:05d}.{}".format(ytid, int(float(start)), "mp4")
        vid_file_path = os.path.join(data_dir, split, vid_file)
    else:  # just rename
        vid_file_new = "{}.{:05d}.{}".format(ytid, int(float(start)), "mp4")
        vid_file_new_path = os.path.join(data_dir, split, vid_file_new)
        if os.path.isfile(vid_file_path):
            os.rename(vid_file_path, vid_file_new_path)
            return True
        if os.path.isfile(vid_file_new_path):
            return True
    """

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        # Download video information from YouTube
        try:
            vid_info = ydl.extract_info("https://www.youtube.com/watch?v=" + ytid, download=False)
            url = vid_info["url"]
            ext = vid_info["ext"]
            if not url:
                raise ValueError("Video URL is not available.")
            if ext != "mp4":
                raise ValueError("Video extension is not mp4.")
        except youtube_dl.utils.DownloadError as e:
            return False
        except Exception as e:
            print(e)
            return False
   
        # Video clip download with ffmpeg
        try:
            ret = subprocess.call([
                    "ffmpeg",
                    "-y",                   # Overwrite without asking
                    "-ss", start,           # Starting point (sec)
                    "-i", url,              # Input video path (YouTube video URL)
                    "-t", "10",             # Max duration (sec)
                    "-c:v", "copy",         # Video codec
                    "-c:a", "copy",         # Audio codec
                    "-loglevel", "error",   
                    "-hide_banner",
                    "--", vid_file_path      # Output video path
                ], timeout=timeout)
            if ret != 0:
                if os.path.exists(vid_file_path):
                    os.remove(vid_file_path)
                return False
        except subprocess.TimeoutExpired:
            print("Timeout after {} seconds.".format(timeout))
            return False

        """
        # Convert to video frames (1 frame per second)
        try:
            os.makedirs(os.path.join(vid_dir, ytid), exist_ok=True)
            ret = subprocess.call([
                    "ffmpeg",
                    "-y",                   # Overwrite without asking
                    "-i", vid_file_path,     # Input video path
                    "-vf", "fps=1",         # Frame to extract per second
                    "-s", "256x256",        # Output image size
                    "-loglevel", "error",
                    "-hide_banner",
                    "--", os.path.join(vid_dir, ytid, "{}_%02d.jpg".format(ytid))  # Output image path
                ], timeout=timeout)
            if ret != 0:
                raise RuntimeError("Unable to extract frames from the video.")
        except Exception as e:
            print(e)
            return False

        # Convert to audio (wav file)
        try:
            ret = subprocess.call([
                    "ffmpeg",
                    "-y",                   # Overwrite without asking
                    "-i", vid_file_path,     # Input video path
                    "-ac", "1",             # Number of audio channel (mono)
                    "-ar", "44100",         # Sampling rate (44,100 Hz)
                    "-vn",                  # Save audio only
                    "-loglevel", "error",
                    "-hide_banner",
                    "--", os.path.join(aud_dir, vid_file.replace(".mp4", ".wav"))  # output audio path
                ], timeout=timeout)
            if ret != 0:
                raise RuntimeError("Unable to extract audio from the video.")
        except Exception as e:
            print(e)
            return False
        """
        
        # Remove video file
        if remove_vid and os.path.isfile(vid_file_path):
            os.remove(vid_file_path)
        
        return True


if __name__ == "__main__":
    """
    # Make destination directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(vid_dir, exist_ok=True)
    os.makedirs(aud_dir, exist_ok=True)
    """
    # Make train/test directory
    os.makedirs(os.path.join(data_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "test"), exist_ok=True)

    # Get total rows of csv file
    with open(VGGSound_csv) as vggsound:
        nrows = len(vggsound.readlines())

    # Youtube ID counter
    ytid_list = list()
    csvfile, reader = load_csv(VGGSound_csv)
    for i, row in enumerate(reader):
        if i < row_start:
            continue
        elif i == row_end:
            break
        ytid_list.append(row["ytid"])
    ytid_counter = Counter(ytid_list)
    csvfile.close()

    # Download starts here
    print("Starting VGGSound downloader...")
    success_cnt, fail_cnt = 0, 0
    last_success = row_start
    csvfile, reader = load_csv(VGGSound_csv)
    for i, row in enumerate(reader):
        if i < row_start:
            continue
        elif i == row_end:
            break
        
        t = int(time.time() - start_time)
        timestamp = "{:02d}:{:02d}:{:02d}".format(t // 3600, (t % 3600 // 60), t % 60)
        print("Progress: {}/{}, Time elapsed: {}".format(i, nrows - 1, timestamp))
        if row["ytid"][0] == "#":  # Skip row with comment
            continue
        row["label"] = row["label"].split(",")

        success = download(row, ytid_counter, remove_vid)
        if success:
            last_success = i
            success_cnt += 1
        else:
            fail_cnt += 1
            print("Last successful row was {}.".format(last_success))
        time.sleep(delay)  # Give delay to prevent IP ban
    
    print("Finished downloading (Row: {} ~ {}, Success: {}, Fail: {})"
          .format(row_start, row_end - 1, success_cnt, fail_cnt))
    csvfile.close()
