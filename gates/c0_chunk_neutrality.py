# -*- coding: utf-8 -*-
"""Phase C / C0 gate (要件): chunked prefill が中立=単発と一致することを
**harness 経路**(HF use_cache の position 処理込み)で 262k・窓OFF/ON(k=2) 両方で確認。
窓述語は絶対距離 z=n-m 依存ゆえ、chunk 跨ぎの位置オフセット誤りは OFF では出ず ON で
だけ壊れる → 両方一致で初めて chunked 化が中立。単発が可能な 262k で実施。GPU."""
import os
import sys

os.environ.setdefault("HF_HOME", "/home/USER/atflash/shared/data/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, "/home/USER/atflash/agents/agentB/phasea")

import torch
import atflash_e2e_fafi as H

N = 262144
CHUNK = 65536
tok, model = H.setup_model("qwen7b1m", 2.0, 2.0)
g = torch.Generator(device="cuda").manual_seed(0)
ids = torch.randint(0, model.config.vocab_size, (1, N), generator=g,
                    device="cuda", dtype=torch.long)


def get_kv_k(cache, li=0):
    lyr = cache.layers[li] if hasattr(cache, "layers") else None
    if lyr is not None:
        return lyr.keys if hasattr(lyr, "keys") else lyr.key_cache
    return cache.key_cache[li]


def prefill(mode, chunk):
    H.MODE["mode"] = mode
    H.PREFILL_CHUNK["v"] = chunk
    torch.cuda.empty_cache()
    cache = None
    with torch.inference_mode():
        for cs in range(0, N, chunk or N):
            out = model(input_ids=ids[:, cs:min(cs + (chunk or N), N)],
                        use_cache=True, past_key_values=cache, logits_to_keep=1)
            cache = out.past_key_values
    return out.logits[:, -1, :].float().cpu(), get_kv_k(cache).float().cpu()


print(f"model=7B-1M N={N} chunk={CHUNK}")
for mode in ("nowin", "win"):
    ls, ks = prefill(mode, None)      # single forward
    lc, kc = prefill(mode, CHUNK)     # chunked
    dlog = (ls - lc).abs().max().item()
    t1 = int(ls.argmax(-1)) == int(lc.argmax(-1))
    kbit = torch.equal(ks, kc)
    kmax = (ks - kc).abs().max().item()
    tag = "OFF" if mode == "nowin" else "ON(k=2)"
    print(f"[{tag}] last-logits: bit={torch.equal(ls, lc)} max|Δ|={dlog:.3e} "
          f"top1_same={t1} | KV-K: bit={kbit} max|Δ|={kmax:.3e}")
print("C0_CHUNK_NEUTRALITY_DONE")
