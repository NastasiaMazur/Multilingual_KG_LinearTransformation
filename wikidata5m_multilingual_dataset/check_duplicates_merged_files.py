"""
Example usage:

python deduplicate_merged_files.py --lang en
python deduplicate_merged_files.py --lang de
python deduplicate_merged_files.py --lang ru
"""

import argparse
from pathlib import Path

def check_duplicates(file_path: Path, has_header=False):
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return

    with file_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    header = lines[0] if has_header else None
    data_lines = lines[1:] if has_header else lines

    total_lines = len(data_lines)
    unique_lines = list(dict.fromkeys(data_lines))  # preserves order
    num_removed = total_lines - len(unique_lines)

    if num_removed > 0:
        print(f"⚠️  {file_path.name}: {num_removed} duplicate lines found out of {total_lines}")
    else:
        print(f"✅ {file_path.name}: No duplicates found")

def main(lang: str):
    base = f"wikidata5m_top200_{lang}_60k"
    files = {
        f"{base}.tsv": False,
        f"{base}_labels.tsv": False,
        f"{base}_descriptions.tsv": True,  # has header
        f"{base}_FAILED.tsv": False,
    }

    for fname, has_header in files.items():
        check_duplicates(Path(fname), has_header)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, help="Language code: en, de, ru")
    args = parser.parse_args()

    main(args.lang)
