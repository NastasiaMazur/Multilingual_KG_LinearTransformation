"""
Export MTransE English entity embeddings to vectors.tsv + metadata.tsv
---------------------------------------------------------------------
• Assumes you trained with model2.py (MTransE).
• Uses the checkpoint prefix  ./test-model-m2.ckpt
• Uses the pickle  ./test-multiG-m2.bin  to map rows → English labels
• Writes both TSVs into ./projector_tsv/
"""

import os, pickle, numpy as np, tensorflow as tf

CKPT_PREFIX   = "./test-model-m2.ckpt"        # <-- change if yours differs
DATA_DUMP     = "./test-multiG-m2.bin"
OUT_DIR       = "./projector_tsv"             # files will appear here
EMB_VAR_SHAPE = None                          # leave None → auto-detect

os.makedirs(OUT_DIR, exist_ok=True)

# --- 1. Load the trained embedding tensor ----------------------------------
reader = tf.train.load_checkpoint(CKPT_PREFIX)
var_map = reader.get_variable_to_shape_map()

# Heuristic: pick the only [num_entities, dim] variable.
candidates = [k for k,shape in var_map.items() if len(shape) == 2]
if not candidates:
    raise RuntimeError("No 2-D variable found in checkpoint!")
emb_key = candidates[0] if len(candidates)==1 else candidates[0]  # pick first
emb_matrix = reader.get_tensor(emb_key)          # numpy array [N, d]
print(f"Loaded embedding '{emb_key}' with shape {emb_matrix.shape}")

# --- 2. Load English labels from test-multiG-m2.bin ------------------------
with open(DATA_DUMP, "rb") as f:
    multiG = pickle.load(f)
row2label = multiG.index2ent[1]   # language 1 == English in your training

if emb_matrix.shape[0] != len(row2label):
    raise RuntimeError("Row count mismatch between embeddings and label list!")

# --- 3. Write vectors.tsv --------------------------------------------------
vec_path = os.path.join(OUT_DIR, "vectors.tsv")
np.savetxt(vec_path, emb_matrix, delimiter="\t", fmt="%.6f")
print(f"Wrote vectors to {vec_path}")

# --- 4. Write metadata.tsv -------------------------------------------------
meta_path = os.path.join(OUT_DIR, "metadata.tsv")
with open(meta_path, "w", encoding="utf-8") as meta_f:
    meta_f.write("Label\n")                 # header row (can add more columns)
    for i in range(emb_matrix.shape[0]):
        meta_f.write(row2label[i] + "\n")
print(f"Wrote metadata to {meta_path}")
print("\nAll done! Download the two TSV files and drop them into "
      "https://projector.tensorflow.org/ .")
