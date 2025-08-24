
# Export EN + DE entity embeddings (MTransE) for TensorBoard / Projector
# --------------------------------------------------------------------
import os, sys, pickle, numpy as np, tensorflow as tf, importlib.util

# ── 1. make src/ importable so Tester, KG, etc., work ----------------------
SRC_DIR = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, SRC_DIR)

def mod_from(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m    = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    sys.modules[name] = m
    return m

KG       = mod_from(os.path.join(SRC_DIR, 'KG.py'),        'KG')
multiG   = mod_from(os.path.join(SRC_DIR, 'multiG.py'),    'multiG')
testerM2 = mod_from(os.path.join(SRC_DIR, 'tester_MTransE2.py'),
                    'tester_MTransE2').Tester

# ── 2. paths ---------------------------------------------------------------
CKPT_PREFIX = './test-model-m2-no-alignment.ckpt'      #  <-- adjust if needed
DATA_DUMP   = './test-multiG-m2-no-alignment-wk3l-en-de.bin'
OUT_DIR     = './projector_tsv_no-alignment'
os.makedirs(OUT_DIR, exist_ok=True)

# ── 3. build Tester exactly as in test_detail_model2.py --------------------
#     so we get vec_e[1] and vec_e[2] already mapped to EN / DE entities
t = testerM2()
t.build(save_path=CKPT_PREFIX, data_save_path=DATA_DUMP)

# vec_e is a dict: {1: [N_en,d], 2: [N_de,d]}
vec_en = t.vec_e[1]
vec_de = t.vec_e[2]
emb    = np.vstack([vec_en, vec_de])
print(f"✓ embeddings: EN {vec_en.shape},  DE {vec_de.shape},  stacked {emb.shape}")

# ── 4. label lists from the same Tester object ----------------------------
labels_en = [t.ent_index2str(i, 1) for i in range(len(vec_en))]
labels_de = [t.ent_index2str(i, 2) for i in range(len(vec_de))]
langs_en  = ['EN'] * len(labels_en)
langs_de  = ['DE'] * len(labels_de)

labels = labels_en + labels_de
langs  = langs_en  + langs_de

assert emb.shape[0] == len(labels), "rows vs. labels mismatch!"

# ── 5. write TSVs ----------------------------------------------------------
vec_path  = os.path.join(OUT_DIR, 'vectors_bi.tsv')
meta_path = os.path.join(OUT_DIR, 'metadata_bi.tsv')

np.savetxt(vec_path, emb, fmt='%.6f', delimiter='\t')

with open(meta_path, 'w', encoding='utf-8') as md:
    md.write('Label\tLang\n')                # header allowed with 2+ columns
    for lbl, lg in zip(labels, langs):
        md.write(f'{lbl}\t{lg}\n')

print("\nDone! Files:")
print("  ", vec_path)
print("  ", meta_path)
print("\nDrag both files into https://projector.tensorflow.org/")
print("or run:  tensorboard --logdir", OUT_DIR)
