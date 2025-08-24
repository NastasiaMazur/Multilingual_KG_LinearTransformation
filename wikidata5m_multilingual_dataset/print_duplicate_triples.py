"""
Deduplicates lines in chunked Wikidata5M TSV files.
Preserves header (if present). Removes all exact duplicate lines.

Example usage:
    python deduplicate_chunk_files.py --lang de --chunk 3
"""

import argparse
from pathlib import Path
from collections import OrderedDict

def deduplicate_file(file_path: Path, has_header=False):
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return

    with file_path.open(encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        print(f"⚠️  {file_path.name}: File is empty.")
        return

    header = lines[0] if has_header else None
    data_lines = lines[1:] if has_header else lines

    seen = OrderedDict()
    for line in data_lines:
        seen[line] = None  # ordered set emulation

    deduplicated_lines = list(seen.keys())
    num_removed = len(data_lines) - len(deduplicated_lines)

    # Write back to file
    with file_path.open("w", encoding="utf-8") as f:
        if has_header:
            f.write(header)
        f.writelines(deduplicated_lines)

    if num_removed == 0:
        print(f"✅ {file_path.name}: No duplicates found.")
    else:
        print(f"🧹 {file_path.name}: Removed {num_removed} duplicates.")

def main(lang: str, chunk_idx: int):
    base = f"wikidata5m_top200_{lang}_60k"
    files = {
        f"{base}_part{chunk_idx}.tsv": {"has_header": False},
        f"{base}_labels_part{chunk_idx}.tsv": {"has_header": False},
        f"{base}_descriptions_part{chunk_idx}.tsv": {"has_header": True},
        f"{base}_FAILED_part{chunk_idx}.tsv": {"has_header": False},
    }

    for fname, opts in files.items():
        deduplicate_file(Path(fname), **opts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, help="Language code: en, de, ru")
    parser.add_argument("--chunk", required=True, type=int, help="Chunk index (0 to N-1)")
    args = parser.parse_args()
    main(args.lang, args.chunk)
 