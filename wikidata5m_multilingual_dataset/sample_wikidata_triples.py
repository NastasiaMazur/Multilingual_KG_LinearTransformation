
"""
Samples triples from the Wikidata5M dataset based on per-relation quotas and
language-specific label/description availability. Duplicate triples are
excluded: if a (subject, relation, object) has already been written to the
chunk’s files it will be skipped and another candidate will be tried.

Example:
    python sample_wikidata_triples.py --lang ru --chunks 8 --chunk 1

"""
import argparse, asyncio, json, random, csv, sys
from pathlib import Path
from collections import defaultdict
from urllib.parse import quote

import aiohttp
from tqdm import tqdm

# CLI
p = argparse.ArgumentParser()
p.add_argument("--lang", required=True, choices=["en", "de", "ru"])
p.add_argument("--chunks", type=int, default=1)
p.add_argument("--chunk", type=int, default=0)
p.add_argument("--quota-file", default="relation_stats_top200_en_de_ru_60k.tsv")
p.add_argument("--input", default="wikidata5m_inductive_train.txt")
args = p.parse_args()

LANG = args.lang
CHUNKS = args.chunks
CHUNK_IDX = args.chunk
QUOTA_FILE = Path(args.quota_file)
INPUT_FILE = Path(args.input)

OUT_TSV   = Path(f"wikidata5m_top200_{LANG}_60k_part{CHUNK_IDX}.tsv")
LABELS_TSV= Path(f"wikidata5m_top200_{LANG}_60k_labels_part{CHUNK_IDX}.tsv")
DESCR_TSV = Path(f"wikidata5m_top200_{LANG}_60k_descriptions_part{CHUNK_IDX}.tsv")
FAILED_TSV= Path(f"wikidata5m_top200_{LANG}_60k_FAILED_part{CHUNK_IDX}.tsv")

LABEL_CACHE = Path(f"label_cache_{LANG}.json")
DESC_CACHE  = Path(f"desc_cache_{LANG}.json")

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKI_API     = f"https://{LANG}.wikipedia.org/api/rest_v1/page/summary/"

# Cache I/O
def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8").strip()
        return json.loads(text) if text else {}
    except json.JSONDecodeError:
        print(f"⚠️  Warning: {path.name} is not valid JSON. Starting fresh.")
        return {}

def save_json(path: Path, data: dict):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

label_cache: dict[str, str | None] = load_json(LABEL_CACHE)
desc_cache: dict[str, str | None] = load_json(DESC_CACHE)


def save_json(path: Path, data: dict):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

label_cache: dict[str, str | None] = load_json(LABEL_CACHE)
desc_cache : dict[str, str | None] = load_json(DESC_CACHE)

# Load quotas
rel_quota: dict[str, int] = {}
with QUOTA_FILE.open(encoding="utf-8") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        rel_quota[row["relation"]] = int(row["quota"])

all_relations = sorted(rel_quota.keys())
my_relations  = [r for i, r in enumerate(all_relations) if i % CHUNKS == CHUNK_IDX]
print(f"⚙️  Worker {CHUNK_IDX}/{CHUNKS} → {len(my_relations)} relations")

# Collect already-written triples
existing_counts  : dict[str, int]        = defaultdict(int)
existing_triples : set[tuple[str, str, str]] = set()

if OUT_TSV.exists():
    with OUT_TSV.open(encoding="utf-8") as f:
        for line in f:
            s, r, o = line.strip().split("\t")
            existing_counts[r] += 1
            existing_triples.add((s, r, o))

# Bucket triples
buckets: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
with INPUT_FILE.open(encoding="utf-8") as f:
    for line in f:
        s, r, o = line.rstrip("\n").split("\t")
        if r in my_relations:
            buckets[r].append((s, r, o))

if not any(buckets.values()):
    print(f"❌ No triples found for this chunk ({CHUNK_IDX}). Either already done or no matching relations.")
    sys.exit(0)

# Async helpers
BATCH     = 50
MAX_CONCUR= 20
SEM       = asyncio.Semaphore(MAX_CONCUR)

async def fetch_labels(session: aiohttp.ClientSession, ids: list[str]):
    need = [i for i in ids if i not in label_cache]
    if not need:
        return
    for chunk in (need[i:i+BATCH] for i in range(0, len(need), BATCH)):
        params = {
            "action": "wbgetentities", "ids": "|".join(chunk),
            "props": "labels", "languages": LANG, "format": "json"
        }
        try:
            async with SEM, session.get(WIKIDATA_API, params=params, timeout=30) as r:
                data = await r.json(content_type=None)
                for q in chunk:
                    label_cache[q] = data.get("entities", {}).get(q, {}).get("labels", {}).get(LANG, {}).get("value")
        except Exception as e:
            tqdm.write(f"⚠️ [ERROR] Fetching labels failed: {e}")

async def fetch_summary(session: aiohttp.ClientSession, label: str):
    if label in desc_cache:
        return
    url = WIKI_API + quote(label.replace(" ", "_"))
    try:
        async with SEM, session.get(url, timeout=20) as r:
            if r.status == 200:
                data = await r.json()
                extract = (data.get("extract") or "").replace("\n", " ").strip()
                desc_cache[label] = extract if extract.count(".") >= 2 else None
            else:
                desc_cache[label] = None
    except Exception:
        desc_cache[label] = None

# Main per-relation worker
async def process_relation(rel_id: str, triples: list[tuple[str, str, str]],
                           out_tsv, labels_tsv, descr_tsv, fail_tsv):
    already_have = existing_counts.get(rel_id, 0)
    quota        = rel_quota[rel_id]
    remaining    = quota - already_have
    if remaining <= 0:
        tqdm.write(f"✅ Skipping {rel_id}: already reached {already_have}/{quota}")
        return 0

    tqdm.write(f"🔄 Processing {rel_id}: need {remaining} more triples ({already_have}/{quota})")
    random.shuffle(triples)
    triples = triples[:500_000]                               # keep sample manageable
    added   = 0
    tried   = 0

    async with aiohttp.ClientSession(headers={"User-Agent": "KG-fast"}) as session:
        pbar = tqdm(total=remaining, desc=f"🔄 {rel_id}", position=1, leave=True, dynamic_ncols=True)

        for s, r, o in triples:
            if added >= remaining:
                break
            tried += 1

            # Deduplication check
            if (s, r, o) in existing_triples:
                continue

            await fetch_labels(session, [s, r, o])

            if not all([label_cache.get(s), label_cache.get(r), label_cache.get(o)]):
                fail_tsv.write(f"{s}\t{r}\t{o}\tMISSING_LABEL\n"); fail_tsv.flush()
                existing_triples.add((s, r, o))                       # avoid re-trying the same bad triple
                continue

            s_label = label_cache[s]
            o_label = label_cache[o]

            await asyncio.gather(
                fetch_summary(session, s_label),
                fetch_summary(session, o_label)
            )

            if not all([desc_cache.get(s_label), desc_cache.get(o_label)]):
                fail_tsv.write(f"{s}\t{r}\t{o}\tMISSING_DESCRIPTION\n"); fail_tsv.flush()
                existing_triples.add((s, r, o))
                continue

            # Write triple
            out_tsv.write(f"{s}\t{r}\t{o}\n"); out_tsv.flush()
            labels_tsv.write(
                f"{s}\t{s_label}\t{r}\t{label_cache[r]}\t{o}\t{o_label}\n"
            ); labels_tsv.flush()
            descr_tsv.write(
                f"{s}\t{s_label}\t{r}\t{label_cache[r]}"
                f"\t{o}\t{o_label}\t{desc_cache[s_label]}\t{desc_cache[o_label]}\n"
            ); descr_tsv.flush()

            existing_triples.add((s, r, o))
            added += 1
            pbar.update(1)

            # If tried a lot and got nothing, bail out early
            if tried >= 900_000 and added == 0:
                tqdm.write(f"⚠️ {rel_id}: gave up after {tried} attempts, no usable triples")
                break

        pbar.close()

    tqdm.write(f"📦 {rel_id}: added {added} new triples ({already_have}+{added}/{quota})")
    return added

# Main runner
async def runner():
    written_total = 0
    with OUT_TSV.open("a", encoding="utf-8") as out_f, \
         LABELS_TSV.open("a", encoding="utf-8") as lbl_f, \
         DESCR_TSV.open("a", encoding="utf-8") as dsc_f, \
         FAILED_TSV.open("a", encoding="utf-8") as fail_f:

        # header for description file (only once)
        if DESCR_TSV.stat().st_size == 0:
            dsc_f.write(
                "subject_id\tsubject_label\trelation_id\trelation_label\t"
                "object_id\tobject_label\tsubject_description\tobject_description\n"
            ); dsc_f.flush()

        outer = tqdm(my_relations, desc=f"[{LANG}] Processing relations",
                      position=0, dynamic_ncols=True)
        for rid in outer:
            added = await process_relation(rid, buckets[rid],
                                           out_f, lbl_f, dsc_f, fail_f)
            written_total += added

            if written_total % 10 == 0:
                save_json(LABEL_CACHE, label_cache)
                save_json(DESC_CACHE, desc_cache)

        outer.close()

    save_json(LABEL_CACHE, label_cache)
    save_json(DESC_CACHE, desc_cache)
    print(f"✅ Worker {CHUNK_IDX}: wrote {written_total} new triples")

if __name__ == "__main__":
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(runner())
