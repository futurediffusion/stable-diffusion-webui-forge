import time
from threading import Thread

import psutil

try:
    import torch
except Exception:  # Torch may not be installed when running tests
    torch = None


class MemoryMonitor:
    def __init__(self, interval: float = 5.0):
        self.interval = interval
        self.metrics = {}
        self._state = None
        self._running = False
        self._thread = None

    def _collect(self):
        cpu_percent = psutil.cpu_percent()
        vm = psutil.virtual_memory()
        process_mem = psutil.Process().memory_info().rss

        if torch and torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated()
        else:
            gpu_mem = 0

        self.metrics = {
            "cpu_percent": cpu_percent,
            "ram_used": vm.used,
            "ram_total": vm.total,
            "process_used": process_mem,
            "gpu_allocated": gpu_mem,
        }

    def _loop(self):
        while self._running:
            try:
                self._collect()
                if self._state is not None:
                    self._state.value = self.metrics
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self, gr_state):
        """Start monitoring using given gradio.State to expose metrics."""
        self._state = gr_state
        self._running = True
        self._thread = Thread(target=self._loop, daemon=True)
        self._thread.start()


monitor = MemoryMonitor()


def start(interval: float = 5.0):
    """Start the global memory monitor and return a gradio.State for metrics."""
    import gradio as gr  # local import to avoid gradio dependency if unused

    state = gr.State({})
    monitor.interval = interval
    monitor.start(state)
    return state


def get_metrics():
    return monitor.metrics
