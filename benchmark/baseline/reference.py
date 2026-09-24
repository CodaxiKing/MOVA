"""Which bit-exact tiny-VACE reference applies to this machine.

fp32 CPU output is bit-exact on a given CPU, but PyTorch dispatches different SIMD kernels per instruction set
(`torch.backends.cpu.get_cpu_capability()`), which changes the last bits. So there is one reference per
capability, each captured with the SAME frozen script and code (commit 8284436):

- tiny_vace_cpu.json       HP ProBook 640 G8 (Tiger Lake). Capability not recorded at capture time; AVX512 is
                           INFERENCE from the CPU family. Used for every capability without its own file.
- tiny_vace_cpu_avx2.json  Intel i5-10400F (Comet Lake), capability AVX2.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BY_CAPABILITY = {"AVX2": HERE / "tiny_vace_cpu_avx2.json"}
DEFAULT = HERE / "tiny_vace_cpu.json"


def baseline_file() -> Path:
    import torch

    return BY_CAPABILITY.get(torch.backends.cpu.get_cpu_capability(), DEFAULT)


def baseline_sha256() -> str:
    return json.loads(baseline_file().read_text(encoding="utf-8"))["runs"][0]["output"]["sha256"]
