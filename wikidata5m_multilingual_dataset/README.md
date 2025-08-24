# Wikidata Subsampling & MTransE Prep — README

This repository documents the end‑to‑end pipeline used to sample Wikidata triples, prepare inputs for mTransE, compute cosine similarities (on GPU, but can be switched to CPU), and export multiple dataset formats (descriptions and labels‑only). It also notes how alignment was turned **off** during mTransE training.


---

## TL;DR

```
→ download_wikidata.py
→ compute_relation_stats_60k.py
→ sample_wikidata_triples.py
→ merge_chunks.py

→ convert_for_mTransE_csv.py

→ [train with MTransE‑tf (alignment **off**)]

→ extract_and_save_embeddings_with_labels.py 
→ export_vectors_tsv_bilingual_no_alignment.py
→ visualize_t-SNE*.py   

→ compute_relation_stats_42k.py
→ sample_42k_from_60k.py
→ truncate_descriptions.ipynb  
→ append_cosine.py
→ create_2_formats.py (descriptions & labels‑only)
→ split_dataset.ipynb   

```

---



## 1) Download Wikidata5M dataset (inductive split)

```bash
python download_wikidata.py
```

**Output:** raw triples

---

## 2) Compute relation stats for **60k** target

```bash
python compute_relation_stats_60k.py
```

**Output:** `relation_stats_top200_en_de_ru_60k.tsv`

---

## 3) Sample ~60k triples (optionally in chunks)

**Chunked (e.g., 8 chunks):**
```bash
# Run this 8 times with different --chunk index and language
python sample_wikidata_triples.py --lang ru --chunks 8 --chunk 1
python sample_wikidata_triples.py --lang ru --chunks 8 --chunk 2
...
```

## 4)  Merge created chuncks
```bash
# ...
python merge_chunks.py --lang ru
python merge_chunks.py --lang en
python merge_chunks.py --lang de
```

> Repeat per language as needed: set `--lang` and language‑specific paths (e.g., `en`, `de`, `ru`).

---

## 5) Convert for **mTransE‑tf** (CSV)

```bash
python convert_for_mTransE_csv.py --lang en
python convert_for_mTransE_csv.py --lang de
python convert_for_mTransE_csv.py --lang ru
```

**Output (example):**
```
creates .csv files where each line is a triple in the form subject_label@@@relation_label@@@object_label
```

> This conversion keeps only what mTransE needs. If descriptions are present, they can be kept for later steps but mTransE requires entities and relations.


---

## 6) mTransE-tf training (without cross-lingual alignment)

For training, please refer to the [MTransE-tf](https://github.com/muhaochen/MTransE-tf) repository (this code is adapted from repository and is **not** included here).

If you want to train without cross-lingual alignment, use the files instead:
- `trainer2_no_alignment.py`
- `training_model2_no_alignment.py`

### Exporting embeddings

- **Language-specific extraction (EN/DE/RU):**  
  Use `extract_and_save_embeddings_with_labels.py` to extract and save **subject** and **object** embeddings **with their corresponding labels** from the language-specific triple files (English, German, and Russian).

- **TensorBoard Projector export:**  
  Use `export_vectors_tsv_bilingual_no_alignment.py` to export **EN + DE or EN + RU** entity embeddings (from MTransE) to TSV files suitable for **TensorBoard Projector**.


### t-SNE visualization (subjects vs. objects)

Use these scripts to project and visualize subject/object embeddings with t-SNE and optionally highlight specific entities.

- **Single language:** `visualize_t-SNE.py`  
  Loads `subj_obj_embeddings/{subject,object}_embeddings_<lang>.npy` and corresponding label files  
  `subj_obj_embeddings/{subject,object}_labels_<lang>.txt` (defaults shown for **ru**).  
  Edit `highlight_*_idx` (0-based) in the script to label specific points.

- **Two languages:**  
  - `visualize_t-SNE_2lang_with_dots_en-de.py` (EN + DE)  
  - `visualize_t-SNE_2lang_with_dots_en-ru.py` (EN2 + RU)  
  Each loads 8 files: `{subject,object}_embeddings_*.npy` and `{subject,object}_labels_*.txt` for both languages.

**Run**
```bash
python visualize_t-SNE.py
python visualize_t-SNE_2lang_with_dots_en-de.py
python visualize_t-SNE_2lang_with_dots_en-ru.py
```
---

## 7) Compute relation stats for **42k** target

```bash
python compute_relation_stats_42k.py
```

---

## 8) Sample **42k** from the **60k** pool

```bash
python sample_42k_from_60k.py --lang en
```

**Output:** `wikidata5m_top200_{lang}_42k_descriptions.tsv`

---


## 9) Truncate descriptions to 2 sentences

Use the Jupyter notebook **`truncate_descriptions.ipynb`** to preprocess entity descriptions by truncating them to the **first two sentences**.  

---

## 10) Appending cosine similarities

- **Compute subject–object similarity:**  
  Use `append_cosine.py` to append the **cosine similarity between subject and object embeddings** to each triple.  
  Works with both *label-only* and *description-rich* triple formats.  
  The script automatically skips a header row if present.

  **Example usage:**

  ```bash
  # With descriptions
  python append_cosine.py \
      --ckpt    test-model-m2-no-alignment-wk5m60k-en-ru.ckpt \
      --data    test-multiG-m2-no-alignment-wk5m60k-en-ru.bin \
      --triples wikidata5m_top200_ru_42k_descriptions.tsv \
      --out     triples_42K_ru_with_cos_desc.tsv \
      --lang    ru

  # With labels only
  python append_cosine.py \
      --ckpt    test-model-m2-no-alignment-wk5m60k-en-ru.ckpt \
      --data    test-multiG-m2-no-alignment-wk5m60k-en-ru.bin \
      --triples wikidata5m_top200_ru_42k_labels.tsv \
      --out     triples_42K_ru_with_cos_labels.tsv \
      --lang    ru

---

## 11) Create two datasets in CSV formats

To convert a **cosine-augmented** triples TSV into two CSV files use: 
```bash
# Adjust the input path in the script if needed
python create_2_formats.py
```

- `wikidata5m_multiling_en2_42k_desc.csv` — includes **subject**, **object**, **similarity**, plus **subject_desc** and **object_desc** (all fields quoted).
- `wikidata5m_multiling_en2_42k.csv` — a compact version with only **subject**, **object**, **similarity**.
---

## 12) Splitting the dataset into train/validation/test

Use the Jupyter notebook **`split_dataset.ipynb`** to create reproducible **train / validation / test** splits from your triples file(s).

---


