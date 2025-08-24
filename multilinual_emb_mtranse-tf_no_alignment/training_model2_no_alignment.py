"""
Variation of the original `training_model2.py` that **keeps all variable names and
hyper‑parameters unchanged** except for the minimal tweaks needed to train the two
knowledge graphs **without any alignment step**.

Changes at a glance
-------------------
1. **No call to `load_align`** – the alignment table is never read, so `multiG.align`
   stays empty.
2. **Alignment weight is 0** and **`AM_fold = 0`** when calling `train_MTransE`.
   This disables the AM optimiser while leaving the rest of the training loop
   untouched.

Everything else – batch sizes, random seeds, path variables – is preserved.
"""
from __future__ import absolute_import, division, print_function

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

# pick the same GPU mask you used before
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np  # noqa: F401  (needed by Trainer internals)
import tensorflow as tf  # noqa: F401

from KG import KG
from multiG import multiG
import model2 as model  # noqa: F401  (not referenced directly but left intact)
from trainer2 import Trainer

# -----------------------------------------------------------------------------
# Path and hyper‑parameter definitions (unchanged)
# -----------------------------------------------------------------------------
model_path = './test-model-m2-no-alignment.ckpt'
data_path = 'test-multiG-m2-no-alignment.bin'
kgf1 = 'preprocess/wk3l_60k/structure/en_60k.csv'
kgf2 = 'preprocess/wk3l_60k/structure/de_60k.csv'
alignf = 'preprocess/wk3l_60k/alignment/en_de_60k_train25.csv'  # kept but UNUSED

this_dim = 50

# Allow CLI overrides (same order as before)
if len(sys.argv) > 1:
    this_dim = int(sys.argv[1])
    model_path = sys.argv[2]
    data_path = sys.argv[3]
    kgf1 = sys.argv[4]
    kgf2 = sys.argv[5]
    alignf = sys.argv[6]  # still parsed but we will not load it

# -----------------------------------------------------------------------------
# Load the two monolingual graphs
# -----------------------------------------------------------------------------
KG1, KG2 = KG(), KG()
KG1.load_triples(filename=kgf1, splitter='@@@', line_end='\n')
KG2.load_triples(filename=kgf2, splitter='@@@', line_end='\n')

# Bundle them; **do NOT add alignment pairs**
this_data = multiG(KG1, KG2)
# (original call removed)  this_data.load_align(...)

# -----------------------------------------------------------------------------
# Build the TensorFlow model
# -----------------------------------------------------------------------------
m_train = Trainer()
# a1 is kept at 5.0 here but will be set to 0.0 during training, so it doesn’t matter
a1_build = 5.0
m_train.build(this_data,
              dim=this_dim,
              batch_sizeK=128,
              batch_sizeA=64,
              a1=a1_build,           # value irrelevant for no‑align training
              a2=0.5,
              m1=0.5,
              save_path=model_path,
              multiG_save_path=data_path,
              L1=False)

# -----------------------------------------------------------------------------
# Train **without alignment**
# -----------------------------------------------------------------------------
m_train.train_MTransE(epochs=100,
                      save_every_epoch=100,
                      lr=0.001,
                      a1=0.0,      # <-- alignment weight OFF
                      a2=0.5,
                      m1=0.5,
                      AM_fold=0,   # <-- skip AM batches
                      half_loss_per_epoch=150)
