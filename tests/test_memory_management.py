import types
import importlib
import sys
from pathlib import Path
import pytest

class DummyDevice:
    def __init__(self, type, index=None):
        self.type = type
        self.index = index

class DummyDType:
    def __init__(self, itemsize):
        self.itemsize = itemsize

float16 = DummyDType(2)
bfloat16 = DummyDType(2)
float32 = DummyDType(4)
float8_e4m3fn = DummyDType(1)
float8_e5m2 = DummyDType(1)

class TorchStub:
    def __init__(self):
        self.cuda = types.SimpleNamespace(
            OutOfMemoryError=Exception,
            is_available=lambda: False,
            current_device=lambda: 0,
            memory_stats=lambda dev: {"reserved_bytes.all.current": 2048},
            mem_get_info=lambda dev: (1, 100),
            get_device_name=lambda dev: "cuda",
            get_allocator_backend=lambda: "backend",
            get_device_properties=lambda dev: types.SimpleNamespace(major=8, total_memory=789),
            is_bf16_supported=lambda: False,
        )
        self.xpu = types.SimpleNamespace(
            is_available=lambda: False,
            memory_stats=lambda dev: {"reserved_bytes.all.current": 1024},
            get_device_properties=lambda dev: types.SimpleNamespace(total_memory=1234),
            current_device=lambda: 0,
            get_device_name=lambda dev: "xpu",
        )
        self.version = types.SimpleNamespace(cuda="", __version__="1.0")
        self.backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False),
            cuda=types.SimpleNamespace(
                enable_math_sdp=lambda x: None,
                enable_flash_sdp=lambda x: None,
                enable_mem_efficient_sdp=lambda x: None,
            ),
        )
        self.nn = types.SimpleNamespace(Parameter=lambda *a, **k: None)
        self.float16 = float16
        self.bfloat16 = bfloat16
        self.float32 = float32
        self.float8_e4m3fn = float8_e4m3fn
        self.float8_e5m2 = float8_e5m2

    def device(self, arg=None, index=None):
        if isinstance(arg, str):
            return DummyDevice(arg, index)
        if isinstance(arg, int):
            return DummyDevice("cuda", arg)
        if arg is None:
            return DummyDevice("cpu")
        return arg

    def use_deterministic_algorithms(self, *a, **k):
        pass

    def tensor(self, *a, **k):
        return types.SimpleNamespace(nelement=lambda: 0, element_size=lambda: 0)

    def zeros(self, shape):
        return None

torch_stub = TorchStub()
psutil_stub = types.SimpleNamespace(virtual_memory=lambda: types.SimpleNamespace(total=1024 * 1024 * 1024))
stream_stub = types.SimpleNamespace(should_use_stream=lambda: False)
utils_stub = types.SimpleNamespace(tensor2parameter=lambda x: x)

@pytest.fixture(autouse=True)
def mm(monkeypatch):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "psutil", psutil_stub)
    monkeypatch.setitem(sys.modules, "backend.stream", stream_stub)
    monkeypatch.setitem(sys.modules, "backend.utils", utils_stub)
    import backend.memory_management as mm
    importlib.reload(mm)
    return mm

def test_get_total_memory_cpu(mm, monkeypatch):
    monkeypatch.setattr(mm.psutil, "virtual_memory", lambda: types.SimpleNamespace(total=4096))
    dev = types.SimpleNamespace(type="cpu")
    assert mm.get_total_memory(dev) == 4096


def test_get_total_memory_cuda(mm, monkeypatch):
    monkeypatch.setattr(mm, "directml_enabled", False)
    monkeypatch.setattr(mm, "is_intel_xpu", lambda: False)
    monkeypatch.setattr(mm.torch.cuda, "memory_stats", lambda dev: {"reserved_bytes.all.current": 55})
    monkeypatch.setattr(mm.torch.cuda, "mem_get_info", lambda dev: (1, 256))
    dev = types.SimpleNamespace(type="cuda")
    assert mm.get_total_memory(dev) == 256
    assert mm.get_total_memory(dev, torch_total_too=True) == (256, 55)


def test_state_dict_size(mm):
    t1 = types.SimpleNamespace(nelement=lambda: 2, element_size=lambda: 4, device=mm.cpu)
    t2 = types.SimpleNamespace(nelement=lambda: 3, element_size=lambda: 2, device=mm.cpu)
    sd = {"a": t1, "b": t2}
    assert mm.state_dict_size(sd) == 14
    assert mm.state_dict_size(sd, exclude_device=mm.cpu) == 0


def test_unload_model_clones(mm):
    class DummyModel:
        def __init__(self, ident):
            self.ident = ident

        def is_clone(self, other):
            return self.ident == getattr(other, "ident", None)

    class DummyLoaded:
        def __init__(self, model):
            self.model = model
            self.unloaded = False

        def model_unload(self, avoid_model_moving=True):
            self.unloaded = True

    m1 = DummyModel("x")
    m2 = DummyModel("x")
    l1 = DummyLoaded(m1)
    l2 = DummyLoaded(m2)
    mm.current_loaded_models.clear()
    mm.current_loaded_models.extend([l1, l2])
    mm.unload_model_clones(m1)
    assert l1.unloaded and l2.unloaded
    assert mm.current_loaded_models == []
