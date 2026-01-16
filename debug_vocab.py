import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

print("torch:", torch.__version__)
import transformers
print("transformers:", transformers.__version__)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

BASE_MODEL  = "Qwen/Qwen-7B"
DRAFT_MODEL = "Qwen/Qwen1.5-1.8B"  # change to the exact 1.5B-ish model you use

# 1) Load tokenizers
base_tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
draft_tok = AutoTokenizer.from_pretrained(DRAFT_MODEL, trust_remote_code=True)

# 2) Hard checks (most speculative exceptions come from these)
print("base vocab:", base_tok.vocab_size, "draft vocab:", draft_tok.vocab_size)
if base_tok.vocab_size != draft_tok.vocab_size:
    raise RuntimeError("Vocab size mismatch -> speculative decoding will likely fail unless using UAD.")

prompt = "Write one sentence explaining speculative decoding."
base_inputs  = base_tok(prompt, return_tensors="pt")
draft_inputs = draft_tok(prompt, return_tensors="pt")

# If these differ, the assistant is proposing tokens in a different ID space
if not torch.equal(base_inputs["input_ids"], draft_inputs["input_ids"]):
    raise RuntimeError("Tokenizer mismatch: base and draft encode the same prompt to different token IDs.")

# 3) Load models
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
).eval()

draft = AutoModelForCausalLM.from_pretrained(
    DRAFT_MODEL,
    trust_remote_code=True,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
).eval()

# Move inputs to same device as base if needed
if device != "cuda":
    base = base.to(device)
    draft = draft.to(device)

inputs = {k: v.to(device) for k, v in base_inputs.items()}

# 4) Assisted generate smoke test
gen_cfg = GenerationConfig(
    max_new_tokens=64,
    do_sample=False,   # start with greedy; set True if you want sampling
    temperature=1.0,
)

try:
    t0 = time.time()
    out = base.generate(
        **inputs,
        generation_config=gen_cfg,
        assistant_model=draft,
        return_dict_in_generate=True,
        output_scores=False,
    )
    dt = time.time() - t0

    seq = out.sequences[0]
    text = base_tok.decode(seq, skip_special_tokens=True)

    new_tokens = int(seq.shape[-1] - inputs["input_ids"].shape[-1])
    tps = new_tokens / dt if dt > 0 else float("inf")

    print("\n✅ assisted generate OK")
    print("new_tokens:", new_tokens, "wall_time(s):", round(dt, 3), "tokens/sec:", round(tps, 2))
    print("\n--- output ---\n", text)

except Exception as e:
    print("\n❌ assisted generate FAILED")
    print(type(e).__name__, ":", str(e))
    raise
