# Training Multilingual KG Embeddings with MTransE-tf model (without cross-lingual alignment)

For training, please refer to the [MTransE-tf](https://github.com/muhaochen/MTransE-tf) repository (this code is adapted from repository and is **not** included here).

If you want to train without cross-lingual alignment, use the files:
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



---