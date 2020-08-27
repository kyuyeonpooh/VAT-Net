import torch

bundle = torch.load("emb_vggsound_vat_v1.pt")
(img_embs, aud_embs, classes) = bundle["img_emb"], bundle["aud_emb"], bundle["class"]


def v2a():
    for i, img_emb in enumerate(img_embs):
        sim_matrix = (img_emb * aud_embs).sum(dim=1)
        idx = torch.argsort(sim_matrix, descending=True)[:10]
        print("Query image:", classes[i])
        print("Retrieved audio", [classes[j] + " ({:.2f})".format(sim_matrix[j]) for j in idx])
        a = input()

def a2v():
    for i, aud_emb in enumerate(aud_embs):
        sim_matrix = (aud_emb * img_embs).sum(dim=1)
        idx = torch.argsort(sim_matrix, descending=True)[:10]
        print("Query audio:", classes[i])
        print("Retrieved image", [classes[j] + " ({:.2f})".format(sim_matrix[j]) for j in idx])
        a = input()


if __name__ == "__main__":
    v2a()