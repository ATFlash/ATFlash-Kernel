# Window meta for Qwen2.5-7B-Instruct-1M (base 1e7, standard path / rope_scaling
# null / DCA unused). Uses the model's OWN rotary class from config (entity-
# faithful, NO formula reimplementation) WITHOUT loading 7B weights -> login-safe.
# lam_r = 2*pi / inv_freq ; w_r = K_CONVEX * lam_r  (convex k=2, same as 0.5B).
import hashlib, json, math, os, sys, time
import torch
from transformers import AutoConfig
import transformers

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct-1M"
K_CONVEX = 2.0
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "w_meta_qwen257b1m_convexk2.json")
ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
# ensure standard path: no rope_scaling (YaRN), DCA not engaged for inv_freq
if getattr(cfg, "rope_scaling", None):
    print("[warn] cfg.rope_scaling set; standard path expects null", file=sys.stderr)
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
try:
    rot = Qwen2RotaryEmbedding(config=cfg)
except TypeError:
    rot = Qwen2RotaryEmbedding(config=cfg, device="cpu")
inv_freq = rot.inv_freq.detach().double().cpu()
lam = (2.0 * math.pi / inv_freq)
w = (K_CONVEX * lam)
sha = hashlib.sha256(inv_freq.numpy().tobytes()).hexdigest()
meta = {"model": MODEL_ID, "k_convex": K_CONVEX,
        "transformers_version": transformers.__version__,
        "rope_theta": getattr(cfg, "rope_theta", None),
        "head_dim": getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads),
        "inv_freq_sha256": sha,
        "inv_freq": [float(x) for x in inv_freq],
        "lam": [float(x) for x in lam],
        "w": [float(x) for x in w], "ts": ts}
json.dump(meta, open(OUT, "w"), indent=1)
print(f"[w_meta_7b1m] half={len(meta['w'])} rope_theta={meta['rope_theta']} "
      f"head_dim={meta['head_dim']} inv_freq_sha={sha[:16]}.. -> {OUT}",
      file=sys.stderr)
