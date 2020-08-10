from easydict import EasyDict

config = EasyDict()

""" Configuration related to preprocess and loading datasets
"""
config.vggsound = EasyDict()  # Configurations related to VGGSound dataset
config.coco = EasyDict()      # Configurations related to COCO2017 dataset
config.flickr30k = EasyDict() # Configurations related to Flickr30k dataset

""" Configuration of VGGSound downloading source code.
Files using this configurations: download.py
"""
config.vggsound.download = EasyDict()
config.vggsound.download.csv = "VGGSound/vggsound.csv"     # Path to VGGSound csv file
config.vggsound.download.data_dir = "../ssd/VGGSound/raw"  # Directory where videos will be saved

""" Configuration of VGGSound frame and audio extraction source code.
Files using this configurations: extract.py, datasets/VGGSound.py
"""
config.vggsound.extract = EasyDict()
config.vggsound.extract.data_dir = "../ssd/VGGSound/raw"   # Directory where videos are saved
config.vggsound.extract.dest_dir = "../ssd/VGGSound/proc"  # Directory where extracted frames and audio will be saved
config.vggsound.extract.fps = 1                            # FPS configuration for video frame extraction
config.vggsound.extract.img_shape = (256, 256)             # Shape of output image
config.vggsound.extract.audio_sr = 44100                   # Sampling rate of audio (Hz)
config.vggsound.extract.seed = 2020                        # Random seed used for generating validation set
config.vggsound.extract.remove_fail = True                 # Whether to immediately delete video failed on extraction

""" Configuration of model checkpoint and logging while training
File using this configurations: train.py
"""
config.train = EasyDict()
config.train.ckpt_dir = "../ssd/save"
config.train.tensorboard_dir = "../ssd/runs"
