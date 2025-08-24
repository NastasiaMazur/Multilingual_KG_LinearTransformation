import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# === Load subject and object embeddings ===
subj_en2 = torch.tensor(np.load("subj_obj_embeddings/subject_embeddings_en2.npy"))
obj_en2 = torch.tensor(np.load("subj_obj_embeddings/object_embeddings_en2.npy"))
subj_ru = torch.tensor(np.load("subj_obj_embeddings/subject_embeddings_ru.npy"))
obj_ru = torch.tensor(np.load("subj_obj_embeddings/object_embeddings_ru.npy"))

# === Load labels ===
with open("subj_obj_embeddings/subject_labels_en2.txt", encoding="utf-8") as f:
    labels_subj_en2 = [line.strip() for line in f]
with open("subj_obj_embeddings/object_labels_en2.txt", encoding="utf-8") as f:
    labels_obj_en2 = [line.strip() for line in f]
with open("subj_obj_embeddings/subject_labels_ru.txt", encoding="utf-8") as f:
    labels_subj_ru = [line.strip() for line in f]
with open("subj_obj_embeddings/object_labels_ru.txt", encoding="utf-8") as f:
    labels_obj_ru = [line.strip() for line in f]

# === Combine embeddings for t-SNE ===
all_embeddings = torch.cat([subj_en2, obj_en2, subj_ru, obj_ru], dim=0).numpy()
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_result = tsne.fit_transform(all_embeddings)

# === Slice embeddings ===
n_subj_en2 = subj_en2.shape[0]
n_obj_en2 = obj_en2.shape[0]
n_subj_ru = subj_ru.shape[0]
n_obj_ru = obj_ru.shape[0]

i1 = 0
i2 = i1 + n_subj_en2
i3 = i2 + n_obj_en2
i4 = i3 + n_subj_ru

tsne_subj_en2 = tsne_result[i1:i2]
tsne_obj_en2 = tsne_result[i2:i3]
tsne_subj_ru = tsne_result[i3:i4]
tsne_obj_ru = tsne_result[i4:]

# === Plot ===
plt.figure(figsize=(10, 10))
#plt.scatter(tsne_subj_en2[:, 0], tsne_subj_en2[:, 1], c='#ffcc80', s=10, alpha=0.6, label='EN2 Subjects')
#plt.scatter(tsne_obj_en2[:, 0], tsne_obj_en2[:, 1], c='#80b3ff', s=10, alpha=0.6, label='EN2 Objects')
#plt.scatter(tsne_subj_ru[:, 0], tsne_subj_ru[:, 1], c="#fbb3ff", s=10, alpha=0.6, label='RU Subjects')
#plt.scatter(tsne_obj_ru[:, 0], tsne_obj_ru[:, 1], c="#80fdff", s=10, alpha=0.6, label='RU Objects')

plt.scatter(tsne_subj_en2[:, 0], tsne_subj_en2[:, 1], c='#ffcc80', s=10, alpha=0.6)
plt.scatter(tsne_obj_en2[:, 0], tsne_obj_en2[:, 1], c='#80b3ff', s=10, alpha=0.6)
plt.scatter(tsne_subj_ru[:, 0], tsne_subj_ru[:, 1], c='#e3b3ff', s=10, alpha=0.6)
plt.scatter(tsne_obj_ru[:, 0], tsne_obj_ru[:, 1], c='#80fdff', s=10, alpha=0.6)

# Dummy scatter for large legend markers
plt.scatter([], [], c='#ffcc80', s=90, label='EN2 Subjects')
plt.scatter([], [], c='#80b3ff', s=90, label='EN2 Objects')
plt.scatter([], [], c='#e3b3ff', s=90, label='RU Subjects')
plt.scatter([], [], c='#80fdff', s=90, label='RU Objects')

# === Highlight aligned entity pairs ===
# Assumes these indices are aligned in EN and RU

# === Define indices to highlight ===
highlight_subject_indices_en2 = [48490] # mmake -1 #7124
highlight_object_indices_en2 = [48490]  # Use different indices if needed

highlight_subject_indices_ru = [48430] #7158-1 for ru #7157 #6955
highlight_object_indices_ru = [48430]  # Use different indices if needed

# === Highlight EN subjects ===
for idx in highlight_subject_indices_en2:
    coords = tsne_subj_en2[idx]
    label = labels_subj_en2[idx]
    plt.scatter(coords[0], coords[1],
                s=100, edgecolors='black', facecolors='red', alpha=0.9,
                label='Highlighted EN2 Subject' if idx == highlight_subject_indices_en2[0] else "")
    #plt.text(coords[0] + 1, coords[1], label, fontsize=9)
    # Shift subject label slightly upward
    plt.text(coords[0] + 2, coords[1] + 3, label, fontsize=9)

# === Highlight EN objects ===
for idx in highlight_object_indices_en2:
    coords = tsne_obj_en2[idx]
    label = labels_obj_en2[idx]
    plt.scatter(coords[0], coords[1],
                s=100, edgecolors='black', facecolors='yellow', alpha=0.9,
                label='Highlighted EN2 Object' if idx == highlight_object_indices_en2[0] else "")
    #plt.text(coords[0] + 1, coords[1], label, fontsize=9)
    # Shift object label slightly downward
    plt.text(coords[0] + 2, coords[1] - 3, label, fontsize=9)

# === Highlight RU subjects ===
for idx in highlight_subject_indices_ru:
    coords = tsne_subj_ru[idx]
    label = labels_subj_ru[idx]
    plt.scatter(coords[0], coords[1],
                s=100, edgecolors='black', facecolors="#5CF22A", alpha=0.9,
                label='Highlighted RU Subject' if idx == highlight_subject_indices_ru[0] else "")
    plt.text(coords[0] + 2, coords[1], label, fontsize=9)

# === Highlight RU objects ===
for idx in highlight_object_indices_ru:
    coords = tsne_obj_ru[idx]
    label = labels_obj_ru[idx]
    plt.scatter(coords[0], coords[1],
                s=100, edgecolors='black', facecolors="#B638FF", alpha=0.9,
                label='Highlighted RU Object' if idx == highlight_object_indices_ru[0] else "")
    plt.text(coords[0] + 2, coords[1], label, fontsize=9)

# === Final plot formatting ===
plt.title("t-SNE of Subject/Object Embeddings (EN2 + RU)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.grid(True)
plt.tight_layout()
plt.show()
