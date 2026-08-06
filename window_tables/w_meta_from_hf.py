# Window meta from the HF model ENTITY (spec: no formula reimplementation).
# lam_r = 2*pi / model.model.rotary_emb.inv_freq  (float64)
# w_r   = K_CONVEX * lam_r   (convex k=2; the min(.., N) cap in lbv2
#         make_policies() never changes the drop set for z <= N-1, so the
#         uncapped form is decision-identical on both sides)
# CPU only. Output: w_meta JSON consumed by atf_window_backend + HF reference.
import hashlib, json, os, sys, time

MODEL_ID = "Qwen/Qwen2.5-0.5B"
K_CONVEX = 2.0
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

import math  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402
import transformers  # noqa: E402

ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, trust_remote_code=True,
        attn_implementation="eager")
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, trust_remote_code=True,
        attn_implementation="eager")
inv_freq = model.model.rotary_emb.inv_freq.detach().double()   # ENTITY
lam = (2.0 * math.pi / inv_freq)
w = (K_CONVEX * lam)
sha = hashlib.sha256(inv_freq.numpy().tobytes()).hexdigest()
meta = {"model": MODEL_ID, "k_convex": K_CONVEX,
        "transformers_version": transformers.__version__,
        "inv_freq_sha256": sha,
        "inv_freq": [float(x) for x in inv_freq],
        "lam": [float(x) for x in lam],
        "w": [float(x) for x in w],
        "ts": ts}
path = os.path.join(OUT_DIR, "w_meta_qwen2505b_convexk2.json")
json.dump(meta, open(path, "w"), indent=1)
print(f"[w_meta] half={len(meta['w'])} inv_freq_sha={sha[:16]}.. -> {path}",
      file=sys.stderr)
