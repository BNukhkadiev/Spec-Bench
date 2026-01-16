# ======= ONE-CELL SPECULATIVE DECODING DEBUG =======

import time, json, hashlib
import torch, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("cuda available:", torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"

MAIN_ID  = "Qwen/Qwen2.5-7B-Instruct"
DRAFT_ID = "Qwen/Qwen2.5-1.5B-Instruct"

print("\nLoading tokenizers...")
main_tok  = AutoTokenizer.from_pretrained(MAIN_ID, use_fast=True)
draft_tok = AutoTokenizer.from_pretrained(DRAFT_ID, use_fast=True)

def tok_fingerprint(tok):
    vocab = tok.get_vocab()
    h = hashlib.sha256(
        json.dumps(sorted(vocab.items()), ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:12]
    return {
        "vocab_size": tok.vocab_size,
        "hash": h,
        "bos": (tok.bos_token, tok.bos_token_id),
        "eos": (tok.eos_token, tok.eos_token_id),
        "pad": (tok.pad_token, tok.pad_token_id),
        "unk": (tok.unk_token, tok.unk_token_id),
    }

print("\nMAIN tokenizer:", tok_fingerprint(main_tok))
print("DRAFT tokenizer:", tok_fingerprint(draft_tok))

prompt = "Explain speculative decoding in one sentence."

enc_main  = main_tok(prompt, return_tensors="pt")
enc_draft = draft_tok(prompt, return_tensors="pt")

print("\nTokenizer equality checks:")
print("  Same vocab size :", main_tok.vocab_size == draft_tok.vocab_size)
print("  Same input_ids  :", torch.equal(enc_main["input_ids"], enc_draft["input_ids"]))

print("\nLoading models...")
main = AutoModelForCausalLM.from_pretrained(
    MAIN_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
).eval()

drafter = AutoModelForCausalLM.from_pretrained(
    DRAFT_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
).eval()

if device != "cuda":
    main = main.to(device)
    drafter = drafter.to(device)

inputs = {k: v.to(device) for k, v in enc_main.items()}

gen_cfg = GenerationConfig(
    max_new_tokens=64,
    do_sample=False,
    temperature=0.0,
)

# -------- Test 1: vanilla assisted decoding (EXPECTED TO FAIL) --------
print("\n[TEST 1] Vanilla assistant_model generate()")
try:
    _ = main.generate(
        **inputs,
        generation_config=gen_cfg,
        assistant_model=drafter,
    )
    print("❗ Unexpectedly succeeded (tokenizers already compatible)")
except Exception as e:
    print("❌ Expected failure:")
    print(type(e).__name__, ":", e)

# -------- Test 2: FIX 1 — shared tokenizer --------
print("\n[TEST 2] Shared-tokenizer assisted decoding")
try:
    t0 = time.time()
    out = main.generate(
        **inputs,
        generation_config=gen_cfg,
        assistant_model=drafter,
        return_dict_in_generate=True,
    )
    dt = time.time() - t0
    seq = out.sequences[0]
    text = main_tok.decode(seq, skip_special_tokens=True)
    new_tokens = seq.shape[-1] - inputs["input_ids"].shape[-1]
    print("✅ Shared-tokenizer generate OK")
    print("   new_tokens:", new_tokens, "time:", round(dt,3), "tok/s:", round(new_tokens/dt,2))
    print("   output:", text)
except Exception as e:
    print("❌ Shared-tokenizer failed:")
    print(type(e).__name__, ":", e)

# -------- Test 3: FIX 2 — Universal Assisted Decoding --------
print("\n[TEST 3] Universal Assisted Decoding (UAD)")
try:
    t0 = time.time()
    out = main.generate(
        **inputs,
        generation_config=gen_cfg,
        assistant_model=drafter,
        tokenizer=main_tok,
        assistant_tokenizer=draft_tok,
        return_dict_in_generate=True,
    )
    dt = time.time() - t0
    seq = out.sequences[0]
    text = main_tok.decode(seq, skip_special_tokens=True)
    new_tokens = seq.shape[-1] - inputs["input_ids"].shape[-1]
    print("✅ UAD generate OK")
    print("   new_tokens:", new_tokens, "time:", round(dt,3), "tok/s:", round(new_tokens/dt,2))
    print("   output:", text)
except Exception as e:
    print("❌ UAD failed:")
    print(type(e).__name__, ":", e)

print("\n===== DONE =====")
