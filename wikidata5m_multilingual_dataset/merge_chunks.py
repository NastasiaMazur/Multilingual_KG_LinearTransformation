"""
Usage example:
python merge_chunks.py --lang ru
python merge_chunks.py --lang en
python merge_chunks.py --lang de
"""

import argparse
import glob
from pathlib import Path

def merge_files(language: str):
    prefix = f"wikidata5m_top200_{language}_60k"
    
    targets = {
        f"{prefix}.tsv": sorted(glob.glob(f"{prefix}_part*.tsv")),
        f"{prefix}_labels.tsv": sorted(glob.glob(f"{prefix}_labels_part*.tsv")),
        f"{prefix}_descriptions.tsv": sorted(glob.glob(f"{prefix}_descriptions_part*.tsv")),
        f"{prefix}_FAILED.tsv": sorted(glob.glob(f"{prefix}_FAILED_part*.tsv")),
    }

    for out_file, parts in targets.items():
        if not parts:
            print(f"⚠️  No files found for {out_file}")
            continue

        print(f"📦 Merging {len(parts)} files into {out_file}")

        with open(out_file, "w", encoding="utf-8") as fout:
            for i, part in enumerate(parts):
                with open(part, "r", encoding="utf-8") as fin:
                    lines = fin.readlines()
                    if i == 0:
                        fout.writelines(lines)
                    else:
                        # Skip header only for descriptions file
                        if "descriptions" in out_file:
                            fout.writelines(lines[1:])
                        else:
                            fout.writelines(lines)

        print(f"✅ Done: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, help="Language code: en, de, ru")
    args = parser.parse_args()

    merge_files(args.lang)
