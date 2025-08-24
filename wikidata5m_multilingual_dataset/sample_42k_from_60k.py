"""
Usage:
python sample_42k_from_60k.py --lang en
"""

import argparse
import csv
import random
from pathlib import Path
from collections import defaultdict

def sample_reduced_dataset(lang: str):
    prefix_60k = f"wikidata5m_top200_{lang}_60k_descriptions"
    prefix_42k = f"wikidata5m_top200_{lang}_42k_descriptions"

    input_path = Path(f"{prefix_60k}.tsv")
    quota_path = Path("relation_stats_top200_en_de_ru_42k.tsv")
    output_path = Path(f"{prefix_42k}.tsv")

    if not input_path.exists() or not quota_path.exists():
        print("❌ Missing input or quota file.")
        return

    # Load quotas
    relation_quotas = {}
    with quota_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            relation_quotas[row["relation"]] = int(row["quota"])

    # Bucket full lines by relation ID (column index 2)
    relation_buckets = defaultdict(list)
    with input_path.open(encoding="utf-8") as f:
        header = next(f)  # save header
        for line in f:
            cols = line.strip().split("\t")
            if len(cols) < 8:
                continue
            rel_id = cols[2]
            if rel_id in relation_quotas:
                relation_buckets[rel_id].append(line)

    # Sample according to quotas
    sampled_lines = []
    for rel, quota in relation_quotas.items():
        lines = relation_buckets.get(rel, [])
        if len(lines) < quota:
            print(f"⚠️ Not enough triples for {rel} — needed {quota}, found {len(lines)}")
        sampled = random.sample(lines, min(len(lines), quota))
        sampled_lines.extend(sampled)

    print(f"✅ Sampled {len(sampled_lines)} lines (target = 42K)")

    # Write to output file
    with output_path.open("w", encoding="utf-8") as f:
        f.write(header)
        f.writelines(sampled_lines)

    print(f"📦 Written to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, help="Language code: en, de, ru")
    args = parser.parse_args()

    sample_reduced_dataset(args.lang)
