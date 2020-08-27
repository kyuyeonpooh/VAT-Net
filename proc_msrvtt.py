import csv
import os
import json
import pickle


aud_list = os.listdir("../ssd/MSR-VTT/proc/test/audio")
aud_list = [f.split(".")[0] for f in aud_list]
aud_list = set(aud_list)

aud_list_csv = []
with open("../ssd/MSR-VTT/msrvtt_test.csv") as msrvtt_csv:
    reader = csv.DictReader(msrvtt_csv, fieldnames=["key", "vid_key", "video_id", "sentence"], skipinitialspace=True)
    for i, row in enumerate(reader):
        if i == 0:
            continue
        else:
            if row["video_id"] in aud_list:
                aud_list_csv.append(row)

print(len(aud_list_csv))
with open("../ssd/MSR-VTT/msrvtt_test_new.csv", "w") as msrvtt_csv_new:
    writer = csv.writer(msrvtt_csv_new)
    for i, row in enumerate(aud_list_csv):
        writer.writerow([row["key"], row["vid_key"], row["video_id"], row["sentence"]])

