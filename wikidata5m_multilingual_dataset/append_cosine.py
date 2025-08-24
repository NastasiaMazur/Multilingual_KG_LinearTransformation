
"""

────────────────────────────────
Append cosine(subject, object) to each triple line.
Supports both label-only and description-rich formats.
Skips header row if detected.

Example usage:
python append_cosine.py \
    --ckpt    test-model-m2-no-alignment-wk5m60k-en-ru.ckpt \
    --data    test-multiG-m2-no-alignment-wk5m60k-en-ru.bin \
    --triples wikidata5m_top200_ru_42k_descriptions.tsv \
    --out     triples_42K_ru_with_cos_desc.tsv \
    --lang    ru

    or
    
python append_cosine.py  \
  --ckpt    test-model-m2-no-alignment-wk5m60k-en-ru.ckpt \
  --data    test-multiG-m2-no-alignment-wk5m60k-en-ru.bin \
  --triples wikidata5m_top200_ru_42k_labels.tsv \
  --out     triples_42K_ru_with_cos_labels.tsv \
  --lang    ru
"""

from __future__ import annotations
import argparse, csv, sys, importlib.util, re
from pathlib import Path
import numpy as np, tensorflow as tf

# ── helper to load repo modules from ./src ---------------------------------
def load_module(py_path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(py_path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    sys.modules[name] = m; return m

# ── canonicalise label -----------------------------------------------------
_quote_re = re.compile(r"[\"“”„‟‶‷❝❞«»‹›]")
_ws_re    = re.compile(r"\s+")

def canon(label: str) -> str:
    """lower-case, strip all quotes, normalise whitespace"""
    s = _quote_re.sub("", label)
    s = _ws_re.sub(" ", s).strip().lower()
    return s

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b)+1e-8))

# ── CLI --------------------------------------------------------------------
def cli(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--triples", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--lang", required=True, choices=["en", "de", "ru"])
    p.add_argument("--srcdir", default="src")
    return p.parse_args(argv)

# ── main -------------------------------------------------------------------
def main(argv=None):
    a = cli(argv)

    SRC = Path(a.srcdir)
    load_module(SRC/"KG.py","KG")
    load_module(SRC/"multiG.py","multiG")
    load_module(SRC/"model2.py","model2")
    load_module(SRC/"trainer2.py","trainer2")
    Tester = load_module(SRC/"tester_MTransE2.py","tester_MTransE2").Tester

    tester = Tester()
    tester.build(save_path=a.ckpt, data_save_path=a.data)

    vec_en, vec_de = tester.vec_e[1], tester.vec_e[2]

    # build canonical lookup tables ----------------------------------------
    id2idx_en, id2idx_de = {}, {}
    for i in range(len(vec_en)):
        lbl = tester.ent_index2str(i, 1)
        id2idx_en[lbl] = id2idx_en[canon(lbl)] = i
    for i in range(len(vec_de)):
        lbl = tester.ent_index2str(i, 2)
        id2idx_de[lbl] = id2idx_de[canon(lbl)] = i

    def vec_by_label(lbl: str, lang: str):
        c = canon(lbl)
        if lang == "en":
            idx = id2idx_en.get(lbl) or id2idx_en.get(c)
            return None if idx is None else vec_en[idx]
        idx = id2idx_de.get(lbl) or id2idx_de.get(c)
        return None if idx is None else vec_de[idx]

    missing = 0
    with Path(a.triples).open(encoding="utf-8") as fin, \
         Path(a.out).open("w", encoding="utf-8", newline="") as fout:

        rdr = csv.reader(fin, delimiter="\t", quoting=csv.QUOTE_NONE)
        wtr = csv.writer(fout, delimiter="\t")

        # Detect and skip header row
        first_row = next(rdr)
        if first_row[:2] == ["subject_id", "subject_label"]:
            print("⚙️  Detected and skipped header row.")
        else:
            fin.seek(0)
            rdr = csv.reader(fin, delimiter="\t", quoting=csv.QUOTE_NONE)

        for row in rdr:
            if len(row) == 6:
                subj_label = row[1]
                obj_label = row[5]
            elif len(row) == 8:
                subj_label = row[1]
                obj_label = row[5]
            elif len(row) == 3:  # fallback: subject, relation, object only
                subj_label = row[0]
                obj_label  = row[2]
            else:
                wtr.writerow(row + ["NaN"])
                continue

            vs = vec_by_label(subj_label, a.lang)
            vo = vec_by_label(obj_label,  a.lang)

            if vs is None or vo is None:
                missing += 1
                wtr.writerow(row + ["NaN"])
            else:
                sim = cosine(vs, vo)
                wtr.writerow(row + [f"{sim:.6f}"])

    print(f"✅ wrote {a.out}")
    if missing:
        print(f"⚠️  {missing} triples had missing embeddings (NaN)")


if __name__ == "__main__":
    main()