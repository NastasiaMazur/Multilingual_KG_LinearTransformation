
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# === 1. Load your subject and object embeddings (replace with actual data) ===
# Example shape: (num_entities, embedding_dim)
#subject_embeddings = torch.randn(5000, 768)  # Replace with your actual subject vectors
#object_embeddings = torch.randn(5000, 768)   # Replace with your actual object vectors



#display only 1000 
subject_embeddings = torch.tensor(np.load("subj_obj_embeddings/subject_embeddings_ru.npy")) #[:1000]) #comment out
object_embeddings = torch.tensor(np.load("subj_obj_embeddings/object_embeddings_ru.npy")) #[:1000]) #comment out

#to use pt:
#subject_embeddings = torch.load("subject_embeddings_en.pt")
#object_embeddings = torch.load("object_embeddings_en.pt")

# === . Load subject and object labels ===
with open("subj_obj_embeddings/subject_labels_ru.txt", encoding="utf-8") as f:
    subject_labels = [line.strip() for line in f.readlines()]#[:1000]

with open("subj_obj_embeddings/object_labels_ru.txt", encoding="utf-8") as f:
    object_labels = [line.strip() for line in f.readlines()]#[:1000]


# === 2. (Optional) Choose entities to highlight ===
highlight_subject_idx = 11615 #remember that it start from 0 so better check in .txt
highlight_object_idx = 11615


highlight_subject_label = subject_labels[highlight_subject_idx]
highlight_object_label = object_labels[highlight_object_idx]

# === 4. Combine and reduce embeddings with t-SNE ===
all_embeddings = torch.cat([subject_embeddings, object_embeddings], dim=0).numpy()
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_embeddings = tsne.fit_transform(all_embeddings)

num_subjects = subject_embeddings.shape[0]
subject_tsne = tsne_embeddings[:num_subjects]
object_tsne = tsne_embeddings[num_subjects:]

print("📍 Highlighted subject:", highlight_subject_label)
print("📍 Highlighted object :", highlight_object_label)
print("Subject coords:", subject_tsne[highlight_subject_idx])
print("Object coords :", object_tsne[highlight_object_idx])

# === 5. Plot ===
plt.figure(figsize=(8, 8))
plt.scatter(subject_tsne[:, 0], subject_tsne[:, 1], s=10, c='orange', alpha=0.6, label='Subjects')
plt.scatter(object_tsne[:, 0], object_tsne[:, 1], s=10, c='steelblue', alpha=0.6, label='Objects')

# Highlighted points
plt.scatter(subject_tsne[highlight_subject_idx, 0], subject_tsne[highlight_subject_idx, 1],
            s=100, edgecolors='black', facecolors='red', label='Highlighted Subject')
plt.scatter(object_tsne[highlight_object_idx, 0], object_tsne[highlight_object_idx, 1],
            s=100, edgecolors='black', facecolors='yellow', label='Highlighted Object')

# Add labels to highlighted points
plt.text(subject_tsne[highlight_subject_idx, 0] + 1,
         subject_tsne[highlight_subject_idx, 1],
         highlight_subject_label, fontsize=9)

plt.text(object_tsne[highlight_object_idx, 0] + 1,
         object_tsne[highlight_object_idx, 1],
         highlight_object_label, fontsize=9)

# Plot formatting
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()