"""Runtime layer: device selection, precision policy, offload/memory policy, runtime lookup and execution.

Multi-GPU / ROCm / no-bf16 machines are simulated with an injected DeviceProbe: this tests MOVA's selection
logic, not the hardware (real GPU execution stays UNVERIFIED until run on one).
"""

import pytest
import torch

from common.errors import (ConfigError, DeviceNotSupportedError, PrecisionNotSupportedError,
                           RuntimeNotAvailableError)
from runtime import DeviceManager, MemoryManager, Precision, PrecisionManager, get_runtime
from runtime.backends.pytorch import PyTorchRuntime
from runtime.device import GIB, DeviceInfo
from runtime.manager import describe_runtimes
from runtime.memory import recommend_offload


class FakeProbe:
    def __init__(self, gpus=(), backend="cuda"):
        self.gpus = list(gpus)  # list of (name, vram_gb, bf16)
        self.backend = backend if gpus else None

    def accelerator_backend(self):
        return self.backend

    def accelerator_count(self):
        return len(self.gpus)

    def accelerator(self, i):
        name, vram, bf16 = self.gpus[i]
        return DeviceInfo(id=f"cuda:{i}", type="cuda", backend=self.backend, name=name, index=i,
                          total_memory_gb=vram, compute_capability="8.6", bf16=bf16, fp16=True)

    def mem_get_info(self, i):
        total = int(self.gpus[i][1] * GIB)
        return total // 2, total

    def cpu(self):
        return DeviceInfo(id="cpu", type="cpu", backend="cpu", name="test-cpu", total_memory_gb=16.0,
                          bf16=True, fp16=True)


def _dm(*gpus, backend="cuda"):
    return DeviceManager(FakeProbe(gpus, backend))


# --- DeviceManager --------------------------------------------------------------------------------

def test_auto_without_gpu_is_cpu():
    dm = _dm()
    assert [d.id for d in dm.list_devices()] == ["cpu"]
    assert dm.resolve("auto").id == "cpu" and dm.resolve(None).id == "cpu"


def test_auto_prefers_first_gpu_and_explicit_index_selects_other():
    dm = _dm(("RTX 3060", 8.0, True), ("RTX 4090", 24.0, True))
    assert dm.resolve("auto").id == "cuda:0"
    assert dm.resolve("cuda").id == "cuda:0"
    assert dm.resolve("CUDA:1").name == "RTX 4090"
    assert dm.resolve("cpu").id == "cpu"
    assert dm.memory_info(dm.resolve("cuda:1")) == {"free_gb": 12.0, "total_gb": 24.0}


@pytest.mark.parametrize("spec", ["cuda:2", "cuda"])
def test_missing_device_raises_with_available_list(spec):
    dm = _dm(("RTX 3060", 8.0, True)) if spec == "cuda:2" else _dm()
    with pytest.raises(DeviceNotSupportedError, match="not present") as e:
        dm.resolve(spec)
    assert "Available on this machine" in str(e.value)


@pytest.mark.parametrize("spec", ["tpu", "mps", "cuda:x", "gpu0", "cpu:1"])
def test_invalid_device_string_raises(spec):
    with pytest.raises(DeviceNotSupportedError):
        _dm().resolve(spec)


def test_rocm_backend_is_reported():
    dev = _dm(("MI210", 64.0, True), backend="rocm").resolve("auto")
    assert dev.backend == "rocm" and dev.type == "cuda"


def test_real_machine_devices_are_consistent_with_torch():
    dm = DeviceManager()
    ids = [d.id for d in dm.list_devices()]
    assert ids[-1] == "cpu"
    assert len(ids) - 1 == (torch.cuda.device_count() if torch.cuda.is_available() else 0)


# --- PrecisionManager -----------------------------------------------------------------------------

def test_precision_defaults_per_device():
    pm = PrecisionManager()
    dm = _dm(("Ampere", 8.0, True), ("Turing", 8.0, False))
    assert pm.default(dm.resolve("cuda:0")) == Precision.BF16
    assert pm.default(dm.resolve("cuda:1")) == Precision.FP16
    assert pm.default(dm.resolve("cpu")) == Precision.FP32


def test_precision_aliases_and_unsupported():
    pm = PrecisionManager()
    turing = _dm(("Turing", 8.0, False)).resolve("cuda:0")
    assert pm.resolve("float16", turing) == Precision.FP16
    with pytest.raises(PrecisionNotSupportedError, match="bf16 is not supported on cuda:0"):
        pm.resolve("bf16", turing)
    with pytest.raises(PrecisionNotSupportedError, match="Unknown precision"):
        pm.resolve("int8", turing)


def test_precision_respects_model_allowed_list():
    pm = PrecisionManager()
    gpu = _dm(("Ampere", 8.0, True)).resolve("cuda:0")
    assert pm.resolve("auto", gpu, allowed=("fp16", "fp32")) == Precision.FP16
    with pytest.raises(PrecisionNotSupportedError, match="not supported by this model"):
        pm.resolve("bf16", gpu, allowed=("fp32",))


# --- MemoryManager --------------------------------------------------------------------------------

@pytest.mark.parametrize("vram,mode", [(4.0, "sequential"), (8.0, "model"), (16.0, "model"), (24.0, "none"),
                                       (None, "sequential")])
def test_recommend_offload_thresholds(vram, mode):
    assert recommend_offload(vram) == mode


def test_offload_resolution_per_device():
    mm = MemoryManager()
    dm = _dm(("RTX 3060", 8.0, True))
    assert mm.resolve_offload("auto", dm.resolve("cuda:0")) == "model"
    assert mm.resolve_offload("auto", dm.resolve("cpu")) == "none"
    assert mm.resolve_offload("sequential", dm.resolve("cuda:0")) == "sequential"
    with pytest.raises(ConfigError, match="needs an accelerator"):
        mm.resolve_offload("model", dm.resolve("cpu"))
    with pytest.raises(ConfigError, match="Unknown offload"):
        mm.resolve_offload("disk", dm.resolve("cuda:0"))


def test_memory_tracking_on_cpu_reports_real_process_memory():
    mm = MemoryManager()
    cpu = DeviceManager().resolve("cpu")
    with mm.track(cpu) as report:
        buf = torch.ones(8 * 1024 * 1024)  # 32 MB
        del buf
    d = report.to_dict()
    assert d["before"]["process_rss_gb"] > 0 and d["after"]["system_total_gb"] > 0
    assert d["peak_allocated_gb"] is None  # not measured on CPU; never reported as 0
    assert d["wall_time_s"] >= 0


# --- Runtime + manager ----------------------------------------------------------------------------

def test_get_runtime_default_and_unknown():
    assert isinstance(get_runtime(), PyTorchRuntime) and get_runtime("PyTorch").name == "pytorch"
    with pytest.raises(RuntimeNotAvailableError, match="Unknown runtime"):
        get_runtime("jax")


@pytest.mark.parametrize("name", ["onnx", "tensorrt"])
def test_unimplemented_runtimes_explain_instead_of_pretending(name):
    with pytest.raises(RuntimeNotAvailableError, match="not implemented"):
        get_runtime(name)
    status = {r["name"]: r["status"] for r in describe_runtimes()}
    assert status[name] == "not implemented" and status["pytorch"] == "available"


def test_context_resolves_auto_per_simulated_hardware():
    rt = PyTorchRuntime(devices=_dm(("RTX 3060", 8.0, True)))
    ctx = rt.context()
    assert (ctx.device.id, ctx.precision, ctx.offload) == ("cuda:0", Precision.BF16, "model")
    assert rt.dtype(ctx) == torch.bfloat16
    cpu = rt.context(device="cpu")
    assert (cpu.precision, cpu.offload) == (Precision.FP32, "none")


def test_runtime_places_and_runs_module_on_cpu():
    rt = get_runtime("pytorch")
    ctx = rt.context(device="cpu", precision="fp32", offload="none")
    lin = rt.place(torch.nn.Linear(4, 2), ctx)
    x = torch.ones(1, 4)
    res = rt.run(lambda: lin(x), ctx)
    assert res.value.shape == (1, 2) and not res.value.requires_grad  # no autograd during inference
    assert res.memory.device == "cpu"
    g1, g2 = rt.generator(7, ctx), rt.generator(7, ctx)
    assert torch.equal(torch.randn(3, generator=g1), torch.randn(3, generator=g2))


def test_runtime_applies_offload_hooks_to_pipelines():
    calls = []

    class FakePipe:
        def enable_model_cpu_offload(self, device):
            calls.append(("model", device))

        def enable_sequential_cpu_offload(self, device):
            calls.append(("sequential", device))

    rt = PyTorchRuntime(devices=_dm(("A", 8.0, True), ("B", 8.0, True)))
    rt.place(FakePipe(), rt.context(device="cuda:1", offload="model"))
    rt.place(FakePipe(), rt.context(device="cuda:0", offload="sequential"))
    assert calls == [("model", "cuda:1"), ("sequential", "cuda:0")]  # offload bound to the chosen GPU
