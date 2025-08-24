
"""
Extract and save subject and object embeddings with their corresponding labels
for English, German, and Russian from language-specific triple files.
"""

import csv, sys, importlib.util, re
from pathlib import Path
import numpy as np

# ── Canonicalize label (normalize for matching) ──
_quote_re = re.compile(r"[\"“”„‟‶‷❝❞«»‹›]")
_ws_re = re.compile(r"\s+")
def canon(label: str) -> str:
    s = _quote_re.sub("", label)
    s = _ws_re.sub(" ", s).strip().lower()
    return s

# ── Dynamic module loader ──
def load_module(py_path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(py_path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    sys.modules[name] = m
    return m

# ── Main extraction function ──
def extract_embeddings(lang, ckpt, data, triples, out_prefix, srcdir="src"):
    SRC = Path(srcdir)
    load_module(SRC / "KG.py", "KG")
    load_module(SRC / "multiG.py", "multiG")
    load_module(SRC / "model2.py", "model2")
    load_module(SRC / "trainer2.py", "trainer2")
    Tester = load_module(SRC / "tester_MTransE2.py", "tester_MTransE2").Tester

    tester = Tester()
    tester.build(save_path=ckpt, data_save_path=data)

    vec = tester.vec_e[1] if lang == "en" else tester.vec_e[2]

    id2idx = {}
    for i in range(len(vec)):
        lbl = tester.ent_index2str(i, 1 if lang == "en" else 2)
        id2idx[lbl] = id2idx[canon(lbl)] = i

    def vec_by_label(lbl):
        c = canon(lbl)
        idx = id2idx.get(lbl) or id2idx.get(c)
        return None if idx is None else vec[idx]

    subj_vecs, obj_vecs = [], []
    subj_labels, obj_labels = [], []

    with Path(triples).open(encoding="utf-8") as fin:
        rdr = csv.reader(fin, delimiter="\t", quoting=csv.QUOTE_NONE)
        first_row = next(rdr)
        if first_row[:2] == ["subject_id", "subject_label"]:
            print(f"⚙️  Skipped header row for {lang}")
        else:
            fin.seek(0)
            rdr = csv.reader(fin, delimiter="\t", quoting=csv.QUOTE_NONE)

        for row in rdr:
            subj_label = row[1] if len(row) >= 6 else row[0]
            obj_label  = row[5] if len(row) >= 6 else row[2]

            vs = vec_by_label(subj_label)
            vo = vec_by_label(obj_label)

            if vs is not None:
                subj_vecs.append(vs)
                subj_labels.append(subj_label)

            if vo is not None:
                obj_vecs.append(vo)
                obj_labels.append(obj_label)

    np.save(f"subject_embeddings_{out_prefix}.npy", np.stack(subj_vecs))
    np.save(f"object_embeddings_{out_prefix}.npy", np.stack(obj_vecs))

    with open(f"subject_labels_{out_prefix}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(subj_labels))
    with open(f"object_labels_{out_prefix}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(obj_labels))

    print(f"✅ {lang.upper()}: saved {len(subj_vecs)} subject vectors → subject_embeddings_{out_prefix}.npy")
    print(f"✅ {lang.upper()}: saved {len(obj_vecs)} object vectors → object_embeddings_{out_prefix}.npy")
    print(f"📝 {lang.upper()}: saved subject labels → subject_labels_{out_prefix}.txt")
    print(f"📝 {lang.upper()}: saved object labels → object_labels_{out_prefix}.txt")

# ── Entry Point ──
def main():
    extract_embeddings(
        lang="en",
        ckpt="test-model-m2-no-alignment-wk5m60k-en-de.ckpt",
        data="test-multiG-m2-no-alignment-wk5m60k-en-de.bin",
        triples="wikidata5m_top200_en_60k_labels.tsv",
        out_prefix="en"
    )

    extract_embeddings(
        lang="de",
        ckpt="test-model-m2-no-alignment-wk5m60k-en-de.ckpt",
        data="test-multiG-m2-no-alignment-wk5m60k-en-de.bin",
        triples="wikidata5m_top200_de_60k_labels.tsv",
        out_prefix="de"
    )

    extract_embeddings(
        lang="ru",
        ckpt="test-model-m2-no-alignment-wk5m60k-en-ru.ckpt",
        data="test-multiG-m2-no-alignment-wk5m60k-en-ru.bin",
        triples="wikidata5m_top200_ru_60k_labels.tsv",
        out_prefix="ru"
    )

    extract_embeddings(
        lang="en",
        ckpt="test-model-m2-no-alignment-wk5m60k-en-ru.ckpt",
        data="test-multiG-m2-no-alignment-wk5m60k-en-ru.bin",
        triples="wikidata5m_top200_en_60k_labels.tsv",
        out_prefix="en2"
    )

if __name__ == "__main__":
    main()
