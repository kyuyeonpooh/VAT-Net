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

""" Configuration of COCO2017 and Flickr30k dataset
Files using this configurations: aggregate.py, word2vec.ipynb, datasets/CocoFlickr.py
"""
config.cocoflickr = EasyDict()
config.cocoflickr.coco_train_images = "../ssd/COCO2017/train2017"   # Directory containing images of COCO train set
config.cocoflickr.coco_val_images = "../ssd/COCO2017/val2017"       # Directory containing images of COCO validation set
config.cocoflickr.flickr_images = "../ssd/Flickr30k/images"         # Directory containing images of Flickr30k dataset
config.cocoflickr.coco_train_captions = "../ssd/COCO2017/annotations/captions_train2017.json"  # File containing captions of COCO train set
config.cocoflickr.coco_val_captions = "../ssd/COCO2017/annotations/captions_val2017.json"      # File containing captions of COCO validation set
config.cocoflickr.flickr_captions = "../ssd/Flickr30k/results.csv"                             # File containing captions of Flickr30k dataset

""" Configuration of word2vec and vocabulary files
Files using this configurations: aggregate.py, word2vec.ipynb, datasets/CocoFlickr.py
"""
config.cocoflickr.annotations = "../ssd/CocoFlickr-annotations.pkl"  # File containing combined annotations of COCO and Flickr30k
config.cocoflickr.google_word2vec = "../ssd/GoogleNews-vectors-negative300.bin"  # Raw word2vec of Google-News-300
config.cocoflickr.word2vec = "../ssd/CocoFlickr-word2vec.npy"         # Compressed word2vec of CocoFlickr, referring to Google-News-300
config.cocoflickr.vocabulary = "../ssd/CocoFlickr-vocabulary.pkl"     # File containing dictionary that maps vocabulary to integer
config.cocoflickr.max_words = 16                                      # Max number of words per caption

""" Configuration of model checkpoint and logging while training
File using this configurations: train.py
"""
config.train = EasyDict()
config.train.ckpt_dir = "../ssd/save"          # Directory where model checkpoint will be saved
config.train.tensorboard_dir = "../ssd/runs"   # Directory where tensorboard logs will be saved
