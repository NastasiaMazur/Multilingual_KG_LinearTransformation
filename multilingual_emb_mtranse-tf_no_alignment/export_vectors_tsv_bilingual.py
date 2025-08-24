
# Export EN + DE entity embeddings (MTransE) for TensorBoard Projector
# -------------------------------------------------------------------
import os, sys, pickle, numpy as np, tensorflow as tf, importlib.util

# ── 0.  helper to load modules from src/ without changing your repo ─────────
def mod_from(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod

SRC_DIR = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, SRC_DIR)
KG      = mod_from(os.path.join(SRC_DIR, 'KG.py'),      'KG')
multiG  = mod_from(os.path.join(SRC_DIR, 'multiG.py'),  'multiG')

# ---------------------------------------------------------------------------
CKPT_PREFIX = './test-model-m2.ckpt'      #  <-- adjust if needed
DATA_DUMP   = './test-multiG-m2.bin'
OUT_DIR     = './projector_tsv'
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. read checkpoint & pick entity tensor --------------------------------
reader  = tf.train.load_checkpoint(CKPT_PREFIX)
var_map = reader.get_variable_to_shape_map()

cand = [k for k,s in var_map.items()
        if len(s)==2 and ('ent' in k.lower() or s[0] >= 1000)]
if not cand:
    raise RuntimeError('No entity tensor found.')
emb_key = cand[0]
emb     = reader.get_tensor(emb_key)            # shape [N_EN+N_DE, dim]
print('✓ entity tensor:', emb_key, emb.shape)

# ── 2. load pickle & pull EN / DE labels -----------------------------------
with open(DATA_DUMP,'rb') as f:
    obj = pickle.load(f)

kg_en = obj['KG1']                       # English KG
kg_de = obj['KG2']                       # German  KG

labels_en = [kg_en.ent_index2str(i) for i in range(kg_en.num_ents())]
labels_de = [kg_de.ent_index2str(i) for i in range(kg_de.num_ents())]

labels = labels_en + labels_de
langs  = ['EN'] * len(labels_en) + ['DE'] * len(labels_de)

assert len(labels) == emb.shape[0], 'row mismatch with embedding matrix'

# ── 3. write Projector files ----------------------------------------------
vec_path  = os.path.join(OUT_DIR, 'vectors_bi.tsv')   # renamed
meta_path = os.path.join(OUT_DIR, 'metadata_bi.tsv')  # renamed

np.savetxt(vec_path, emb, fmt='%.6f', delimiter='\t')

with open(meta_path, 'w', encoding='utf-8') as md:
    md.write('Label\tLang\n')                   # header because 2 columns
    for lbl, lg in zip(labels, langs):
        md.write(f'{lbl}\t{lg}\n')

print('\nDone!  Files written to', OUT_DIR)
print('Drag both TSVs into https://projector.tensorflow.org/')
print('or run:  tensorboard --logdir', OUT_DIR)
