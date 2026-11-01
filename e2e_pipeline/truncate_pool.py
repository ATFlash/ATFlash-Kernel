# -*- coding: utf-8 -*-
"""Pre-truncate the long pool to about 132K tokens (keeping both ends, cutting the
middle), so that the end-to-end harness starts fast and every KV size shares the
same base content. CPU only, offline."""
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HOME", "/home/USER/atflash/shared/data/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from transformers import AutoTokenizer

SRC = "/home/USER/atflash/agents/agentB/phasea/lb_long_pool.jsonl"
OUT = "/home/USER/atflash/agents/agentB/phasea/lb_long_pool_132k.jsonl"
CAP = 132000
N_KEEP = 10

tok = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B")
rows = [json.loads(l) for l in open(SRC)][:N_KEEP]
out = []
for r in rows:
    ids = tok(r["context"], add_special_tokens=False).input_ids
    half = CAP // 2
    ctx = (tok.decode(ids[:half]) + "\n...[truncated]...\n"
           + tok.decode(ids[-half:]))
    r2 = dict(r)
    r2["context"] = ctx
    r2["n_tokens_orig"] = len(ids)
    r2["n_tokens_trunc"] = len(tok(ctx, add_special_tokens=False).input_ids)
    out.append(r2)
    print(r["_id"], "orig", len(ids), "->trunc", r2["n_tokens_trunc"])
with open(OUT, "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", len(out), "->", OUT)
