import argparse
import torch
import statistics


parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="v2t", choices=["v2t", "t2v"],
                    help="Retrieve mode")
args = parser.parse_args()

bundle = torch.load("embeddings/emb_vt_loss_v2_model_change.pt")
(img_embs, txt_embs, captions) = bundle["img_emb"], bundle["txt_emb"], bundle["captions"]
(img_idx, cap2idx, idx2cap) = bundle["img_idx"], bundle["cap2idx"], bundle["idx2cap"]
img_embs = img_embs.view(-1, 256)

img_embs = img_embs[:1000]
txt_embs = txt_embs[:5500]

def v2t():
    rank_list = list()
    
    for i, img_emb in enumerate(img_embs):
        sim_matrix = (img_emb * txt_embs).sum(dim=1)
        idx = torch.argsort(sim_matrix, descending=True)
        print("Query image:", idx2cap[i])
        print("Retrieved text", [captions[j] + " ({:.2f})".format(sim_matrix[j]) for j in idx[:5]])
        
        for k, j in enumerate(idx):
            if i == cap2idx[captions[j]]:
                print(captions[j], k + 1)
                rank_list.append(k + 1)
                break

        a = input()
    print("median rank:", statistics.median(rank_list))
    print("R@1:", rank_list.count(1) / len(rank_list))
    r10= sum([rank_list.count(i) for i in range(1, 11)])
    print("R@10:", r10 / len(rank_list))


def t2v():
    rank_list = list()
    for i, txt_emb in enumerate(txt_embs):
        sim_matrix = (txt_emb * img_embs).sum(dim=1)
        idx = torch.argsort(sim_matrix, descending=True)
        #print("Query text:", captions[i])
        #print("Retrieved image", [captions[j] + " ({:.2f})".format(sim_matrix[j]) for j in idx[:5]])
        #a = input()
        for k, j in enumerate(idx):
            if captions[i] == captions[j]:
                #print(k + 1)
                rank_list.append(k + 1)
                break
    print("median rank:", statistics.median(rank_list))
    print("R@1:", rank_list.count(1) / len(rank_list))
    r10= sum([rank_list.count(i) for i in range(1, 11)])
    print("R@10:", r10 / len(rank_list))

if __name__ == "__main__":
    if args.mode == "v2t":
        v2t()
    else:
        t2v()
