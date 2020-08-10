import torch

bundle = torch.load("Emb_VA_B512_LR1e-03_D1e-07_M0.2_S1_linear.pt")
(img_embs, aud_embs, classes) = bundle["img_emb"], bundle["aud_emb"], bundle["class"]


def v2a():
    for i, img_emb in enumerate(img_embs):
        t = (aud_embs**2).sum(dim=1).sqrt()
        print((t > 1.).sum())
        sim_matrix = (img_emb * aud_embs).sum(dim=1)
        idx = torch.argsort(sim_matrix, descending=True)[:10]
        print("Query image:", classes[i])
        print("Retrieved audio", [classes[j] + str(sim_matrix[j]) for j in idx])
        a = input()


if __name__ == "__main__":
    v2a()