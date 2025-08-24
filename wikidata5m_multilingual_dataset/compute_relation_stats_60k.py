"""
compute_relation_stats_60k.py
──────────────────────────────────────────────────────
Creates "relation_stats_top200_en_de_ru_60k.tsv" file with columns:
    relation  count  proportion  quota  en  de  ru

Guarantees:
• Exactly 60 000 triples when quotas are summed.
• Each relation gets at least 50.

"""

import csv, aiohttp, asyncio
from collections import defaultdict
from tqdm import tqdm

# Settings
INPUT_FILE       = "wikidata5m_inductive_train.txt"
TOP_N_RELATIONS  = 1000
TOP_M_FINAL      = 200
TARGET_TRIPLES   = 60_000
MIN_PER_REL      = 50
LANGUAGES        = ["en", "de", "ru"]
OUTPUT_STATS_TSV = "relation_stats_top200_en_de_ru_60k.tsv"


rel_freq    = defaultdict(int)
label_cache = {}
sem         = asyncio.Semaphore(10)


async def fetch_labels(session, entity_id):
    if entity_id in label_cache:
        return label_cache[entity_id]
    url = (
        "https://www.wikidata.org/w/api.php"
        f"?action=wbgetentities&ids={entity_id}&props=labels&languages={'|'.join(LANGUAGES)}&format=json"
    )
    async with sem:
        try:
            async with session.get(url) as resp:
                data   = await resp.json()
                labels = data.get("entities", {}).get(entity_id, {}).get("labels", {})
                result = {lang: labels.get(lang, {}).get("value") for lang in LANGUAGES}
                label_cache[entity_id] = result
                return result
        except Exception:
            label_cache[entity_id] = {lang: None for lang in LANGUAGES}
            return label_cache[entity_id]

# 1. Count relations
print("🔢 Counting relation frequencies…")
with open(INPUT_FILE, encoding="utf-8") as f:
    for line in tqdm(f):
        _, rel, _ = line.rstrip("\n").split("\t")
        rel_freq[rel] += 1

top_relations = sorted(rel_freq.items(), key=lambda kv: kv[1], reverse=True)[:TOP_N_RELATIONS]

# 2. Filter by labels
async def collect_labelled():
    out = []
    async with aiohttp.ClientSession() as sess:
        tasks = {r: asyncio.create_task(fetch_labels(sess, r)) for r, _ in top_relations}
        for r, freq in tqdm(top_relations, desc="🌍 Fetching labels"):
            lbl = await tasks[r]
            if all(lbl[lg] for lg in LANGUAGES):
                out.append({"relation": r, "count": freq, **lbl})
    return out

filtered = asyncio.run(collect_labelled())
filtered = sorted(filtered, key=lambda d: d["count"], reverse=True)[:TOP_M_FINAL]

total_count_kept = sum(d["count"] for d in filtered)

# 3. Initial quotas
for d in filtered:
    prop        = d["count"] / total_count_kept
    d["proportion"] = prop
    d["quota"]       = max(int(round(prop * TARGET_TRIPLES)), MIN_PER_REL)

# 4. Rebalance to EXACT 42 000
quota_sum = sum(d["quota"] for d in filtered)

if quota_sum != TARGET_TRIPLES:
    print(f"⚖️  Rebalancing quotas (initial sum {quota_sum}) …")

    # helper: sort by quota descending each round
    def big_first():
        return sorted(filtered, key=lambda x: x["quota"], reverse=True)

    if quota_sum > TARGET_TRIPLES:  # need to subtract
        surplus = quota_sum - TARGET_TRIPLES
        for d in big_first():
            if surplus == 0:
                break
            take = min(surplus, d["quota"] - MIN_PER_REL)
            d["quota"] -= take
            surplus   -= take
    else:                           # need to add
        deficit = TARGET_TRIPLES - quota_sum
        cyc = 0
        while deficit > 0:
            modified = False
            for d in big_first():
                d["quota"] += 1
                deficit   -= 1
                modified   = True
                if deficit == 0:
                    break
            if not modified:  # should not happen
                break

# Final assert
assert sum(d["quota"] for d in filtered) == TARGET_TRIPLES, "Quota rebalance failed"

# 5. Write TSV
with open(OUTPUT_STATS_TSV, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["relation", "count", "proportion", "quota", "en", "de", "ru"])
    for d in filtered:
        w.writerow([
            d["relation"], d["count"], f"{d['proportion']:.6f}", d["quota"],
            d["en"], d["de"], d["ru"],
        ])

print("Stats written to", OUTPUT_STATS_TSV, "(sum =", sum(d["quota"] for d in filtered), ")")
