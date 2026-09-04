#!/usr/bin/env python3
"""Add GEMMA_QUANTIZE=int8 to upstream's engine.py (Kade-AI, Part 127, Sep 4 2026).

Upstream offers Gemma 3 12B in two shapes: NF4 (4-bit, ~8 GB, the 24 GB
default, "slight quality tradeoff") or bf16 (~24 GB, needs a 48 GB card, or a
slow CPU-streaming path on smaller cards). Her word is "best quality", and no
datacentre with a network volume has 48 GB cards in stock reliably (Sep 4:
L40/L40S/6000 Ada all Low, everywhere). A 32 GB card (RTX 5090, Medium stock
in EU-RO-1) can hold Gemma in 8-bit (~13 GB) fully resident beside the INT8
audio transformer — closer to bf16 quality at NF4 speed. This patch adds that
mode; start.sh picks it by VRAM. If the 8-bit load fails for any reason, the
engine falls back to NF4 and says so, so a render never dies on this.

Applied at image build against upstream 3e3d403. Idempotent. Fails loudly if
the anchors moved (then the pin moved and this needs a look, not a skip).
"""
import re, sys
P = "/app/upstream/src/audio_core/engine.py" if len(sys.argv) < 2 else sys.argv[1]
s = open(P).read()
if "GEMMA_QUANTIZE_INT8_PATCH" in s:
    print("already patched"); sys.exit(0)

def rep(old, new, count=1):
    global s
    assert s.count(old) >= 1, "anchor missing: " + old[:60]
    s = s.replace(old, new, count)

# 1. VRAM strategy: 8-bit Gemma overhead sits between NF4 (11) and bf16 (16).
rep('''        gemma_nf4 = os.environ.get("GEMMA_QUANTIZE", "").lower() == "nf4"
        gemma_overhead_gb = 11.0 if gemma_nf4 else 16.0''',
'''        gemma_nf4 = os.environ.get("GEMMA_QUANTIZE", "").lower() == "nf4"
        gemma_int8 = os.environ.get("GEMMA_QUANTIZE", "").lower() == "int8"  # GEMMA_QUANTIZE_INT8_PATCH
        gemma_overhead_gb = 11.0 if gemma_nf4 else (15.0 if gemma_int8 else 16.0)''')
rep('''            "nf4" if gemma_nf4 else "bf16",
            self.vram_gb,''',
'''            "nf4" if gemma_nf4 else ("int8" if gemma_int8 else "bf16"),
            self.vram_gb,''')

# 2. Loading strategy: int8 rides the NF4 code path with a different config.
rep('''        self._gemma_nf4 = os.environ.get("GEMMA_QUANTIZE", "").lower() == "nf4"
        self._gemma_on_gpu = False

        if self._gemma_nf4:''',
'''        self._gemma_nf4 = os.environ.get("GEMMA_QUANTIZE", "").lower() in ("nf4", "int8")
        self._gemma_int8 = os.environ.get("GEMMA_QUANTIZE", "").lower() == "int8"
        self._gemma_on_gpu = False

        if self._gemma_nf4:''')

# 3. The builder: 8-bit config, NF4 fallback.
rep('''        t0 = time.time()
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        self._nf4_gemma_model = Gemma3ForConditionalGeneration.from_pretrained(
            self.gemma_root,
            quantization_config=quant_config,
            device_map="cuda",
            dtype=torch.bfloat16,
        ).eval()''',
'''        t0 = time.time()
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        want_int8 = getattr(self, "_gemma_int8", False)
        quant_config = BitsAndBytesConfig(load_in_8bit=True) if want_int8 else nf4_config
        try:
            self._nf4_gemma_model = Gemma3ForConditionalGeneration.from_pretrained(
                self.gemma_root,
                quantization_config=quant_config,
                device_map="cuda",
                dtype=torch.bfloat16,
            ).eval()
            logger.info("Gemma quantization mode: %s", "int8" if want_int8 else "nf4")
        except Exception as e:  # noqa: BLE001 — an 8-bit load that will not fit falls back to NF4, out loud
            if not want_int8:
                raise
            logger.warning("Gemma int8 load failed (%s); falling back to NF4", e)
            torch.cuda.empty_cache()
            self._nf4_gemma_model = Gemma3ForConditionalGeneration.from_pretrained(
                self.gemma_root,
                quantization_config=nf4_config,
                device_map="cuda",
                dtype=torch.bfloat16,
            ).eval()''')
open(P, "w").write(s)
print("patched", P)
