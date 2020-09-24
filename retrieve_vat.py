import numpy as np
import argparse
import torch
import statistics


parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="v2t", choices=["v2t", "t2v", "v2a", "a2v", "a2t", "t2a"],
                    help="Retrieve mode")
args = parser.parse_args()

bundle = torch.load("emb_msrvtt_vat_v4.pt")
(img_embs, aud_embs, txt_embs, captions) = bundle["img_emb"], bundle["aud_emb"], bundle["txt_emb"], bundle["caption"]


def compute_metrics(x):
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) / len(ind)
    metrics['MR'] = np.median(ind) + 1
    print(metrics)


def v2t():
    x = np.dot(img_embs, txt_embs.T)
    #y = np.dot(aud_embs, txt_embs.T)
    #x = 0.8 * x + 0.2 * y
    #x = np.dot(img_embs, txt_embs.T)
    compute_metrics(x)
    

def t2v():
    x = np.dot(txt_embs, img_embs.T)
    y = np.dot(txt_embs, aud_embs.T)
    x = 0.8 * x + 0.2 * y
    #x = np.dot(txt_embs, img_embs.T)
    compute_metrics(x)


def v2a():
    x = np.dot(img_embs, aud_embs.T)
    y = np.dot(txt_embs, aud_embs.T)
    #x = 0.9 * x + 0.1 * y
    compute_metrics(x)
    

def a2v():
    x = np.dot(aud_embs, img_embs.T)
    y = np.dot(aud_embs, txt_embs.T)
    x = 0.7 * x + 0.3 * y
    compute_metrics(x)


def a2t():
    #x = np.dot(aud_embs, txt_embs.T)
    #y = np.dot(img_embs, txt_embs.T)
    #x = 0.5 * x + 0.5 * y
    x = np.dot(aud_embs, txt_embs.T)
    """y = np.dot(aud_embs, img_embs.T)
    y_ind = y.argsort(axis=1)[:, -1]
    y_ind = y_ind.squeeze()
    y = img_embs[y_ind]
    y = np.dot(y, txt_embs.T)
    x = 0.1 * x + 0.9 * y"""
    compute_metrics(x)


def t2a():
    x = np.dot(txt_embs, aud_embs.T)
    """y = np.dot(txt_embs, img_embs.T)
    y_ind = y.argsort(axis=1)[:, -1]
    y_ind = y_ind.squeeze()
    y = img_embs[y_ind]
    y = np.dot(y, aud_embs.T)
    x = 0.6 * x + 0.4 * y"""
    compute_metrics(x)


if __name__ == "__main__":
    if args.mode == "v2t":
        v2t()
    elif args.mode == "t2v":
        t2v()
    elif args.mode == "v2a":
        v2a()
    elif args.mode == "a2v":
        a2v()
    elif args.mode == "a2t":
        a2t()
    elif args.mode == "t2a":
        t2a()
