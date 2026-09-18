"""Load a production decoder and generate from it, to check C5's arithmetic against a real model.

C5 read four decoder configs and priced their attention and parameters. This
loads the weights, quantising when they do not otherwise fit on the GPU it has,
and reports what actually loaded: parameter count, footprint, and tokens per
second. A config can be read wrong; a loaded model cannot be.
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1]
QUANT = sys.argv[2] if len(sys.argv) > 2 else "4bit"
MAX_NEW = int(sys.argv[3]) if len(sys.argv) > 3 else 48
PROMPT = os.environ.get("LLM_PROMPT", "Explain in two sentences why grouped-query attention reduces KV cache size.")

config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
text_config = getattr(config, "text_config", config)
print(
    f"model_type={config.model_type} layers={getattr(text_config, 'num_hidden_layers', '?')} "
    f"hidden={getattr(text_config, 'hidden_size', '?')} kv_heads={getattr(text_config, 'num_key_value_heads', '?')}",
    flush=True,
)

kwargs = {"trust_remote_code": True, "device_map": "auto", "low_cpu_mem_usage": True}
if QUANT in ("4bit", "8bit"):
    from transformers import BitsAndBytesConfig

    kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=QUANT == "4bit",
        load_in_8bit=QUANT == "8bit",
        # NF4 with a second-level quantisation of the scales: the combination
        # that fits a 16B MoE on a 16GB card at all.
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
else:
    kwargs["torch_dtype"] = torch.float16

print(f"loading {MODEL} ({QUANT})", flush=True)
started = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(MODEL, **kwargs).eval()
load_seconds = time.perf_counter() - started
parameters = sum(p.numel() for p in model.parameters())
footprint = model.get_memory_footprint() / 1e9
print(
    f"loaded in {load_seconds:.1f}s  {parameters / 1e9:.2f}B tensor elements  {footprint:.2f} GB on device", flush=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
messages = [{"role": "user", "content": PROMPT}]
try:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
except Exception:
    text = PROMPT
inputs = tokenizer(text, return_tensors="pt").to(model.device)

started = time.perf_counter()
with torch.no_grad():
    generated = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
elapsed = time.perf_counter() - started
new_tokens = int(generated.shape[1] - inputs["input_ids"].shape[1])
completion = tokenizer.decode(generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
summary = {
    "model": MODEL,
    "quantization": QUANT,
    "parameters": parameters,
    "footprint_gb": footprint,
    "load_seconds": load_seconds,
    "new_tokens": new_tokens,
    "tokens_per_second": new_tokens / elapsed,
}
print(f"COMPLETION {completion[:600]!r}", flush=True)
print("SUMMARY " + json.dumps(summary), flush=True)
