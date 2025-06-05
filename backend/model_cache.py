import threading
from queue import Queue
from backend import memory_management


_load_queue = Queue()
_loaded = set()
_status = ""
_lock = threading.Lock()


def _worker():
    global _status
    while True:
        model = _load_queue.get()
        if model is None:
            break
        name = getattr(model, "__class__", type(model)).__name__
        _status = f"Prefetching {name}"
        try:
            memory_management.load_model_gpu(model)
            with _lock:
                _loaded.add(model)
            _status = f"Prefetched {name}"
        except Exception as e:
            _status = f"Prefetch failed: {e}"
        finally:
            _load_queue.task_done()


_thread = threading.Thread(target=_worker, daemon=True)
_thread.start()


def prefetch_model(model):
    with _lock:
        if model in _loaded or model in list(_load_queue.queue):
            return
    _load_queue.put(model)


def release_model(model):
    with _lock:
        if model in _loaded:
            memory_management.unload_model_clones(model)
            _loaded.discard(model)


def get_status():
    return _status
