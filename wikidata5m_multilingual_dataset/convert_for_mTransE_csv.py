"""
Example usage:
python convert_for_mTransE_csv.py --lang en
python convert_for_mTransE_csv.py --lang de
python convert_for_mTransE_csv.py --lang ru
"""

import argparse
from pathlib import Path

def convert_to_triples_csv(lang: str):
    prefix = f"wikidata5m_top200_{lang}_60k"
    labels_path = Path(f"{prefix}_labels.tsv")
    output_csv = Path(f"{prefix}_triples.csv")

    if not labels_path.exists():
        print(f"Labels file not found: {labels_path}")
        return

    with labels_path.open("r", encoding="utf-8") as fin, \
         output_csv.open("w", encoding="utf-8") as fout:
        for line in fin:
            parts = line.strip().split("\t")
            if len(parts) != 6:
                continue
            subj_label, rel_label, obj_label = parts[1], parts[3], parts[5]
            fout.write(f"{subj_label}@@@{rel_label}@@@{obj_label}\n")

    print(f"Saved: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, help="Language code: en, de, ru")
    args = parser.parse_args()

    convert_to_triples_csv(args.lang)