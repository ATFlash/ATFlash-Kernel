# -*- coding: utf-8 -*-
"""Build a pool of the longest LongBench-v2 "long" items (Llama-tokenized), so that
the Llama-3.2-3B end-to-end runs have at least 8 items at 64K/128K. CPU only,
offline. Writes lb_long_pool.jsonl sorted by token length, longest first (loader
schema + n_tokens), and prints the length distribution."""
import glob
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HOME", "/home/USER/atflash/shared/data/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from transformers import AutoTokenizer

OUT = "/home/USER/atflash/agents/agentB/phasea/lb_long_pool.jsonl"
FIELDS = ["_id", "context", "question", "choice_A", "choice_B", "choice_C",
          "choice_D", "answer", "length", "domain", "difficulty"]


def load_rows():
    try:
        import datasets
        ds = datasets.load_dataset("THUDM/LongBench-v2", split="train")
        return [dict(r) for r in ds]
    except Exception as e:
        print("load_dataset failed, reading parquet directly:", repr(e)[:150])
        import pyarrow.parquet as pq
        pqs = glob.glob(os.path.expanduser(
            "~/atflash/shared/data/hf_cache/hub/datasets--THUDM--LongBench-v2/"
            "snapshots/*/**/*.parquet"), recursive=True)
        print("parquet files:", pqs)
        rows = []
        for p in pqs:
            rows += pq.read_table(p).to_pylist()
        return rows


rows = load_rows()
print("total rows:", len(rows))
longs = [r for r in rows if r.get("length") == "long"]
print("long rows:", len(longs))

tok = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B")
pool = []
for r in longs:
    n = len(tok(r["context"], add_special_tokens=False).input_ids)
    r2 = {k: r.get(k) for k in FIELDS}
    r2["n_tokens"] = n
    pool.append(r2)
pool.sort(key=lambda x: -x["n_tokens"])

ge = lambda t: sum(1 for x in pool if x["n_tokens"] >= t)
print(f"long items reaching: >=32k:{ge(32000)} >=64k:{ge(64000)} "
      f">=128k:{ge(128000)} >=200k:{ge(200000)}")
print("top-16 token lengths:", [x["n_tokens"] for x in pool[:16]])
with open(OUT, "w", encoding="utf-8") as f:
    for x in pool[:32]:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
print("wrote", min(32, len(pool)), "items ->", OUT)
