# -*- coding: utf-8 -*-
"""Truncate the LongBench-v2 long pool to about 1.01M tokens (keeping both ends,
cutting the middle) with the Qwen2.5-7B-Instruct-1M tokenizer, so that the
256K/512K/1M end-to-end runs share one base content (the harness truncates
further per length). CPU only, offline. Same pairing principle as truncate_pool.py."""
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HOME", "/home/USER/atflash/shared/data/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from transformers import AutoTokenizer

SRC = "/home/USER/atflash/agents/agentB/phasea/lb_long_pool.jsonl"  # full ctx
OUT = "/home/USER/atflash/agents/agentB/phasea/lb_long_pool_1m.jsonl"
CAP = 1_010_000
N_KEEP = 6

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct-1M")
rows = [json.loads(l) for l in open(SRC)][:N_KEEP]
out = []
for r in rows:
    ids = tok(r["context"], add_special_tokens=False).input_ids
    half = CAP // 2
    if len(ids) > CAP:
        ctx = (tok.decode(ids[:half]) + "\n...[truncated]...\n"
               + tok.decode(ids[-half:]))
    else:
        ctx = r["context"]
    r2 = dict(r)
    r2["context"] = ctx
    r2["n_tokens_qwen_orig"] = len(ids)
    r2["n_tokens_qwen_trunc"] = len(tok(ctx, add_special_tokens=False).input_ids)
    out.append(r2)
    print(r["_id"], "qwen_orig", len(ids), "->trunc", r2["n_tokens_qwen_trunc"],
          flush=True)
with open(OUT, "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", len(out), "->", OUT)
