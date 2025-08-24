import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# === Load subject and object embeddings ===
subj_en = torch.tensor(np.load("subj_obj_embeddings/subject_embeddings_en.npy"))
obj_en = torch.tensor(np.load("subj_obj_embeddings/object_embeddings_en.npy"))
subj_de = torch.tensor(np.load("subj_obj_embeddings/subject_embeddings_de.npy"))
obj_de = torch.tensor(np.load("subj_obj_embeddings/object_embeddings_de.npy"))

# === Load labels ===
with open("subj_obj_embeddings/subject_labels_en.txt", encoding="utf-8") as f:
    labels_subj_en = [line.strip() for line in f]
with open("subj_obj_embeddings/object_labels_en.txt", encoding="utf-8") as f:
    labels_obj_en = [line.strip() for line in f]
with open("subj_obj_embeddings/subject_labels_de.txt", encoding="utf-8") as f:
    labels_subj_de = [line.strip() for line in f]
with open("subj_obj_embeddings/object_labels_de.txt", encoding="utf-8") as f:
    labels_obj_de = [line.strip() for line in f]

# === Combine embeddings for t-SNE ===
all_embeddings = torch.cat([subj_en, obj_en, subj_de, obj_de], dim=0).numpy()
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_result = tsne.fit_transform(all_embeddings)

# === Slice embeddings ===
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
#plt.scatter(tsne_subj_en[:, 0], tsne_subj_en[:, 1], c='#ffcc80', s=10, alpha=0.6, label='EN Subjects')
#plt.scatter(tsne_obj_en[:, 0], tsne_obj_en[:, 1], c='#80b3ff', s=10, alpha=0.6, label='EN Objects')
#plt.scatter(tsne_subj_de[:, 0], tsne_subj_de[:, 1], c='#ff99cc', s=10, alpha=0.6, label='DE Subjects')
#plt.scatter(tsne_obj_de[:, 0], tsne_obj_de[:, 1], c='#99ffcc', s=10, alpha=0.6, label='DE Objects')

plt.scatter(tsne_subj_en[:, 0], tsne_subj_en[:, 1], c='#ffcc80', s=10, alpha=0.6)
plt.scatter(tsne_obj_en[:, 0], tsne_obj_en[:, 1], c='#80b3ff', s=10, alpha=0.6)
plt.scatter(tsne_subj_de[:, 0], tsne_subj_de[:, 1], c='#ff99cc', s=10, alpha=0.6)
plt.scatter(tsne_obj_de[:, 0], tsne_obj_de[:, 1], c='#99ffcc', s=10, alpha=0.6)

# Dummy scatter for large legend markers
plt.scatter([], [], c='#ffcc80', s=90, label='EN Subjects')
plt.scatter([], [], c='#80b3ff', s=90, label='EN Objects')
plt.scatter([], [], c='#ff99cc', s=90, label='DE Subjects')
plt.scatter([], [], c='#99ffcc', s=90, label='DE Objects')

# === Highlight aligned entity pairs ===
# Assumes these indices are aligned in EN and DE

# === Define indices to highlight ===
highlight_subject_indices_en = [48490] # mmake -1 #7124
highlight_object_indices_en = [48490]  # Use different indices if needed

highlight_subject_indices_de = [48491] #7158-1 for ru #8189
highlight_object_indices_de = [48491]  # Use different indices if needed

# === Highlight EN subjects ===
for idx in highlight_subject_indices_en:
    coords = tsne_subj_en[idx]
    label = labels_subj_en[idx]
    plt.scatter(coords[0], coords[1],
                s=100, edgecolors='black', facecolors='red', alpha=0.9,
                label='Highlighted EN Subject' if idx == highlight_subject_indices_en[0] else "")
    #plt.text(coords[0] + 1, coords[1], label, fontsize=9)
    # Shift subject label slightly upward
    plt.text(coords[0] + 2, coords[1] + 3, label, fontsize=9)

# === Highlight EN objects ===
for idx in highlight_object_indices_en:
    coords = tsne_obj_en[idx]
    label = labels_obj_en[idx]
    plt.scatter(coords[0], coords[1],
                s=100, edgecolors='black', facecolors='yellow', alpha=0.9,
                label='Highlighted EN Object' if idx == highlight_object_indices_en[0] else "")
    #plt.text(coords[0] + 1, coords[1], label, fontsize=9)
    # Shift object label slightly downward
    plt.text(coords[0] + 2, coords[1] - 3, label, fontsize=9)

# === Highlight DE subjects ===
for idx in highlight_subject_indices_de:
    coords = tsne_subj_de[idx]
    label = labels_subj_de[idx]
    plt.scatter(coords[0], coords[1],
                s=100, edgecolors='black', facecolors="#26B226", alpha=0.9,
                label='Highlighted DE Subject' if idx == highlight_subject_indices_de[0] else "")
    plt.text(coords[0] + 2, coords[1], label, fontsize=9)

# === Highlight DE objects ===
for idx in highlight_object_indices_de:
    coords = tsne_obj_de[idx]
    label = labels_obj_de[idx]
    plt.scatter(coords[0], coords[1],
                s=100, edgecolors='black', facecolors="#7B008B", alpha=0.9,
                label='Highlighted DE Object' if idx == highlight_object_indices_de[0] else "")
    plt.text(coords[0] + 2, coords[1], label, fontsize=9)

# === Final plot formatting ===
plt.title("t-SNE of Subject/Object Embeddings (EN + DE)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.grid(True)
plt.tight_layout()
plt.show()
