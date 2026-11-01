# Window meta for Qwen2.5-7B-Instruct-1M under YaRN factor=2 (extrapolation-pilot
# recipe, the YaRN s=2 setting). Entity-faithful: set rope_scaling=yarn on
# the config and let the model's OWN Qwen2RotaryEmbedding compute the YaRN inv_freq
# (NO formula reimplementation). lam_r = 2*pi/inv_freq_yarn ; w_r = K_CONVEX*lam_r.
# Login-safe (config-only, no 7B weights). Output distinct from the base table.
import hashlib, json, math, os, time
from transformers import AutoConfig
import transformers

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct-1M"
K_CONVEX = 2.0
FACTOR = 2.0
ORIG = 262144
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "w_meta_qwen257b1m_yarn_s2_convexk2.json")
ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
# this model nests rope_theta inside rope_scaling ({'rope_theta':1e7,'rope_type':'default'});
# the YaRN init reads config.rope_theta (top-level), so promote it before overwriting.
_orig_rs = getattr(cfg, "rope_scaling", None) or {}
base_theta = _orig_rs.get("rope_theta") or getattr(cfg, "rope_theta", None) or 10000000.0
cfg.rope_theta = float(base_theta)
# apply YaRN s=2 (replace the default rope_scaling; keep the same base theta)
cfg.rope_scaling = {"rope_type": "yarn", "factor": FACTOR,
                    "original_max_position_embeddings": ORIG}
cfg.max_position_embeddings = int(ORIG * FACTOR)
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
try:
    rot = Qwen2RotaryEmbedding(config=cfg)
except TypeError:
    rot = Qwen2RotaryEmbedding(config=cfg, device="cpu")
inv_freq = rot.inv_freq.detach().double().cpu()
attn_factor = float(getattr(rot, "attention_scaling", 1.0))
lam = (2.0 * math.pi / inv_freq)
w = (K_CONVEX * lam)
sha = hashlib.sha256(inv_freq.numpy().tobytes()).hexdigest()

# sanity vs base (load base table to compare a few freqs)
base = None
bp = os.path.join(os.path.dirname(OUT), "w_meta_qwen257b1m_convexk2.json")
if os.path.exists(bp):
    base = json.load(open(bp))

meta = {"model": MODEL_ID, "k_convex": K_CONVEX, "rope_scaling": cfg.rope_scaling,
        "yarn_factor": FACTOR, "original_max_position_embeddings": ORIG,
        "effective_max_position_embeddings": int(ORIG * FACTOR),
        "attention_scaling": attn_factor,
        "transformers_version": transformers.__version__, "rope_theta": base_theta,
        "head_dim": getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads),
        "inv_freq_sha256": sha,
        "inv_freq": [float(x) for x in inv_freq],
        "lam": [float(x) for x in lam],
        "w": [float(x) for x in w], "ts": ts}
json.dump(meta, open(OUT, "w"), indent=1)
print("[yarn_s2] half=%d attn_scaling=%.6f inv_freq_sha=%s.. -> %s" %
      (len(meta["w"]), attn_factor, sha[:16], OUT))
if base:
    bw, bi = base["w"], base["inv_freq"]
    # YaRN keeps high-freq (short lam) same, stretches low-freq (long lam) windows
    print("  base vs yarn: inv_freq[0]=%.6g/%.6g  inv_freq[-1]=%.6g/%.6g  w[-1]=%.6g/%.6g  (identical=%s)" %
          (bi[0], meta["inv_freq"][0], bi[-1], meta["inv_freq"][-1], bw[-1], meta["w"][-1],
           sha == base.get("inv_freq_sha256")))
