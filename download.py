import csv
import os
import subprocess
import time

import youtube_dl

vggsound_csv = "VGGSound/vggsound.csv"  # VGGSound csv file path
data_dir = "VGGSound/raw"               # Directory where videos will be saved
timeout = 100       # Time limit for downloading video
row_start = 0       # CSV file row start point (starts from 0)
row_end = -1        # CSV file row end point (exclusive, max row if -1)
delay = 2           # Delay between download (in second) to prevent IP ban


def load_vggsound_csv():
    csvfile = open(vggsound_csv)
    reader = csv.DictReader(csvfile, fieldnames=["ytid", "start", "label", "split"], skipinitialspace=True)
    return csvfile, reader


def get_all_ytid():
    ytid_list = list()
    csvfile, reader = load_vggsound_csv()
    for i, row in enumerate(reader):
        if row["ytid"][0] == "#":  # Skip row with comment
            continue
        if i < row_start:
            continue
        elif i == row_end:
            break
        ytid_list.append(row["ytid"])
    csvfile.close()
    return set(ytid_list)


def get_already_downloaded():
    train_data_dir = os.path.join(data_dir, "train")
    test_data_dir = os.path.join(data_dir, "test")
    downloaded_ytid = os.listdir(train_data_dir) + os.listdir(test_data_dir)
    downloaded_ytid = [vidfile.split(".")[0] for vidfile in downloaded_ytid]
    return set(downloaded_ytid)


def download(row, remove_vid=False):
    ytid = row["ytid"]      # YouTube video id
    start = row["start"]    # Start second
    split = row["split"]    # train or test spilt
    duration = 10
    if split != "train" and split != "test":
        raise ValueError("Unknown value {} for key split.".format(split))

    ydl_opts = {
        "start_time": int(float(start)),
        "end_time": int(float(start)) + duration,
        "format": "mp4[height<=360]",
    }

    vid_file = "{}.{:05d}.{}".format(ytid, int(float(start)), "mp4")
    vid_file_path = os.path.join(data_dir, split, vid_file)

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
                    "--", vid_file_path     # Output video path
                ], timeout=timeout)
            if ret != 0:
                if os.path.exists(vid_file_path):
                    os.remove(vid_file_path)
                return False
        except subprocess.TimeoutExpired:
            print("Timeout after {} seconds.".format(timeout))
            return False
                
        return True


if __name__ == "__main__":
    # Make data destination directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "test"), exist_ok=True)

    # Get total rows of csv file
    with open(vggsound_csv) as vggsound:
        nrows = len(vggsound.readlines())

    # Exclude already downloaded video
    not_downloaded_ytid = get_all_ytid() - get_already_downloaded()

    # Download starts here
    print("Starting VGGSound downloader... (Start: {}, End: {})"
          .format(row_start, nrows - 1 if row_end == -1 else row_end - 1))
    start_time = time.time()
    success_cnt, fail_cnt = 0, 0
    last_success = row_start
    csvfile, reader = load_vggsound_csv()
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
        if row["ytid"] not in not_downloaded_ytid:
            continue
        row["label"] = row["label"].split(",")

        success = download(row)
        if success:
            last_success = i
            success_cnt += 1
        else:
            fail_cnt += 1
            print("Last successful row was {}.".format(last_success))
        time.sleep(delay)  # Give delay to prevent IP ban
    
    print("Finished downloading (Row: {} ~ {}, Success: {}, Fail: {})"
          .format(row_start, nrows - 1 if row_end == -1 else row_end - 1, success_cnt, fail_cnt))
    csvfile.close()
