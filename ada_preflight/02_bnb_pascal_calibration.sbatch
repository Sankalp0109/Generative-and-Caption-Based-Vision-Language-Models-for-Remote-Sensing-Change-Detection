#!/bin/bash
#SBATCH --job-name=preflight_bnb
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=preflight_bnb_%j.out

# Stage 0 preflight: bitsandbytes 4-bit NF4 + FP16-vs-FP32 timing, specifically
# to catch whether the Turing-only-proven recipe (see architecture_plan.md,
# "GPU architecture mismatch" section) actually performs the same on a
# GTX 1080 Ti (Pascal, sm_61) node, which is what most jobs will land on.
# Run this a couple of times if needed to land on a 1080 Ti node vs a 2080 Ti
# one, and compare.

echo "Node: $(hostname)"
VENV_PATH="/home2/sankalp0109/src_IS/venv"
if [ ! -x "$VENV_PATH/bin/python3" ]; then
  echo "FATAL: no env found at $VENV_PATH -- run setup_venv.sh first (confirmed missing on this account, see architecture_plan.md 'Software environment' section)."
  exit 1
fi
export PATH="$VENV_PATH/bin:$PATH"  # conda env (created with -p), not a venv -- no bin/activate to source

python3 - <<'PYEOF'
import time, torch

name = torch.cuda.get_device_name(0)
cc = torch.cuda.get_device_capability(0)
print(f"GPU: {name}  compute_capability={cc}")
is_pascal = cc == (6, 1)
print("-> Pascal (1080 Ti class): FP16 compute is expected to be throttled" if is_pascal
      else "-> Not Pascal: FP16 compute should be near-full-rate")

# --- Part A: FP16 vs FP32 forward-pass timing on a ViT-L-14-sized workload ---
import torch.nn as nn
class TinyViTLikeBlock(nn.Module):
    def __init__(self, dim=1024, heads=16, depth=6):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=depth)
    def forward(self, x):
        return self.enc(x)

def timed_forward(dtype, n_iters=20):
    model = TinyViTLikeBlock().to("cuda", dtype=dtype)
    model.eval()
    x = torch.randn(8, 257, 1024, device="cuda", dtype=dtype)  # ~ViT-L/14 patch-token shape
    torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(3):  # warmup
            model(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iters):
            model(x)
        torch.cuda.synchronize()
    return (time.time() - t0) / n_iters

t_fp32 = timed_forward(torch.float32)
t_fp16 = timed_forward(torch.float16)
print(f"\nPart A — ViT-L-14-shaped transformer block forward pass:")
print(f"  FP32: {t_fp32*1000:.1f} ms/iter")
print(f"  FP16: {t_fp16*1000:.1f} ms/iter")
print(f"  FP16 speedup vs FP32: {t_fp32/t_fp16:.2f}x  "
      f"({'as expected, FP16 NOT faster on Pascal' if is_pascal and t_fp16 >= t_fp32*0.9 else 'looks normal' if t_fp16 < t_fp32 else 'UNEXPECTED — re-check'})")

# --- Part B: bitsandbytes 4-bit NF4 load + inference smoke test ---
print(f"\nPart B — bitsandbytes 4-bit NF4:")
try:
    import bitsandbytes as bnb
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    print(f"  bitsandbytes version: {bnb.__version__}")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.float16)
    MODEL = "Qwen/Qwen2-VL-2B-Instruct"  # swap for the exact checkpoint already used in Phase 7 if different
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb_config,
                                                  device_map="cuda:0", trust_remote_code=True)
    print(f"  load time: {time.time()-t0:.1f}s")
    print(f"  peak VRAM after load: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")

    prompt = "Describe this scene in one sentence."
    inputs = tok(prompt, return_tensors="pt").to("cuda:0")
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=40)
    torch.cuda.synchronize()
    print(f"  avg generate() time (text-only smoke test, 40 tokens): {(time.time()-t0)/5:.2f}s")
    print(f"  peak VRAM during generation: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
    print("  RESULT: bnb 4-bit NF4 load+inference WORKS on this GPU.")
except Exception as e:
    print(f"  RESULT: bnb 4-bit NF4 FAILED on this GPU: {type(e).__name__}: {e}")
    print("  If this fails specifically on a 1080 Ti (Pascal), the plan's fallback is to drop")
    print("  4-bit and use fp16-only loading (more VRAM, but Pascal supports it) or 8-bit instead.")
PYEOF
