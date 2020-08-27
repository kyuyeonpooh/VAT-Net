from collections import Counter
import re
import pickle

import numpy as np
import nltk
import gensim

from config import config

# Source paths
dataset_ann_path = config.cocoflickr.annotations
google_word2vec_path = config.cocoflickr.google_word2vec

# Destination paths
dataset_vocab_path = config.cocoflickr.vocabulary
dataset_word2vec_path = config.cocoflickr.word2vec

with open(dataset_ann_path, "rb") as dataset_pkl:
    dataset_ann = pickle.load(dataset_pkl)
stopwords = nltk.corpus.stopwords.words("english")
word2vec = gensim.models.KeyedVectors.load_word2vec_format(google_word2vec_path, binary=True)


def collect_sentences(dataset_ann):
    sentences = list()
    for split in ["train", "val"]:
        ann = dataset_ann[split]
        for key in ann.keys():
            sentences += ann[key]["caption"]
    return sentences


def tokenize(sentence):
    sentence = re.sub("[^a-zA-z]", " ", sentence).lower().strip()
    while (" " * 2) in sentence:
        sentence = sentence.replace(" " * 2, " ")
    words = sentence.split()
    return words


def remove_stopwords(words):
    words = filter(lambda word: word not in stopwords, words)
    return list(words)


def make_vocabulary(sentences):
    vocabulary = {"<pad>": 0, "<unk>": 1}
    word_list = list()

    for s in sentences:
        words = tokenize(s)
        words = remove_stopwords(words)
        words = list(filter(lambda word: word in word2vec, words))
        word_list += words

    word_counter = Counter(word_list)
    word_counter = word_counter.most_common()
    for word, _ in word_counter:
        vocabulary[word] = len(vocabulary)

    return vocabulary


sentences = collect_sentences(dataset_ann)
vocabulary = make_vocabulary(sentences)
with open(dataset_vocab_path, "wb") as dataset_vocab_pkl:
    pickle.dump(vocabulary, dataset_vocab_pkl)

n_unique_words = len(vocabulary)
cocoflickr_word2vec = np.zeros((n_unique_words, 300))
for k, v in vocabulary.items():
    if k not in ["<pad>", "<unk>"]:
        cocoflickr_word2vec[v] = word2vec[k]
with open(dataset_word2vec_path, "wb") as word2vec_file:
    np.save(word2vec_file, cocoflickr_word2vec)

print("Number of unique words:", n_unique_words)
