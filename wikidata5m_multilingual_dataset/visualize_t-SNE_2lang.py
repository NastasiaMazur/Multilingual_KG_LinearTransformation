import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# === Load subject and object embeddings ===
subj_en = torch.tensor(np.load("subj_obj_embeddings/subject_embeddings_en.npy")) #[:1000]) #comment out
obj_en = torch.tensor(np.load("subj_obj_embeddings/object_embeddings_en.npy")) #[:1000]) #comment out
subj_de = torch.tensor(np.load("subj_obj_embeddings/subject_embeddings_de.npy")) #[:1000]) #comment out
obj_de = torch.tensor(np.load("subj_obj_embeddings/object_embeddings_de.npy")) #[:1000]) #comment out

# === Load labels ===
with open("subj_obj_embeddings/subject_labels_en.txt", encoding="utf-8") as f:
    labels_subj_en = [line.strip() for line in f]

with open("subj_obj_embeddings/object_labels_en.txt", encoding="utf-8") as f:
    labels_obj_en = [line.strip() for line in f]

with open("subj_obj_embeddings/subject_labels_de.txt", encoding="utf-8") as f:
    labels_subj_de = [line.strip() for line in f]

with open("subj_obj_embeddings/object_labels_de.txt", encoding="utf-8") as f:
    labels_obj_de = [line.strip() for line in f]

# === Combine embeddings ===
all_embeddings = torch.cat([subj_en, obj_en, subj_de, obj_de], dim=0).numpy()

# === t-SNE dimensionality reduction ===
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_result = tsne.fit_transform(all_embeddings)

# === Track ranges for slicing ===
n_subj_en = subj_en.shape[0]
n_obj_en = obj_en.shape[0]
n_subj_de = subj_de.shape[0]
n_obj_de = obj_de.shape[0]

i1 = 0
i2 = i1 + n_subj_en
i3 = i2 + n_obj_en
i4 = i3 + n_subj_de

tsne_subj_en = tsne_result[i1:i2]
tsne_obj_en = tsne_result[i2:i3]
tsne_subj_de = tsne_result[i3:i4]
tsne_obj_de = tsne_result[i4:]

# === Plot ===
plt.figure(figsize=(10, 10))

plt.scatter(tsne_subj_en[:, 0], tsne_subj_en[:, 1], c='orange', s=10, alpha=0.6, label='EN Subjects')
plt.scatter(tsne_obj_en[:, 0], tsne_obj_en[:, 1], c='blue', s=10, alpha=0.6, label='EN Objects')
plt.scatter(tsne_subj_de[:, 0], tsne_subj_de[:, 1], c='green', s=10, alpha=0.6, label='DE Subjects')
plt.scatter(tsne_obj_de[:, 0], tsne_obj_de[:, 1], c='red', s=10, alpha=0.6, label='DE Objects')

plt.title("t-SNE of Subject/Object Embeddings (EN + DE)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
