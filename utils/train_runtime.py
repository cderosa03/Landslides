"""Bounded raster caches and persistent diagnostics for training processes."""

import faulthandler
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path


def raster_environment(cache_mb):
    """Rasterio's Env takes GDAL_CACHEMAX in bytes, unlike the shell option."""
    import rasterio

    return rasterio.Env(
        GDAL_CACHEMAX=cache_mb * 1024 * 1024,
        GDAL_NUM_THREADS="1",
        VSI_CACHE=False,
        GDAL_DISABLE_READDIR_ON_OPEN="TRUE",
    )


def memory_snapshot(cgroup_root=Path("/sys/fs/cgroup"), proc_root=Path("/proc")):
    """Read container totals separately from process RSS and filesystem cache."""
    result = {"time": datetime.now(timezone.utc).isoformat(), "pid": os.getpid()}
    for name in ("memory.current", "memory.max", "memory.swap.current"):
        try:
            value = (cgroup_root / name).read_text().strip()
            result[name] = int(value) if value != "max" else None
        except (OSError, ValueError):
            pass
    for name in ("memory.stat", "memory.events"):
        try:
            values = dict(line.split() for line in (cgroup_root / name).read_text().splitlines())
            keys = ("anon", "file", "shmem", "slab", "file_dirty", "file_writeback")
            result[name] = {key: int(value) for key, value in values.items()
                            if name == "memory.events" or key in keys}
        except (OSError, ValueError):
            pass
    try:
        for line in (proc_root / "self" / "status").read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, value, _unit = line.split()
                result[key.rstrip(":") + "_bytes"] = int(value) * 1024
    except (OSError, ValueError):
        pass
    return result


class RunDiagnostics:
    """Write memory even while the main thread waits for a DataLoader batch."""

    def __init__(self, directory, interval=10, stall_seconds=300):
        self.directory = Path(directory)
        self.interval = interval
        self.stall_seconds = stall_seconds
        self.stop = threading.Event()

    def _sample(self):
        self.memory_file.write(json.dumps(memory_snapshot()) + "\n")
        self.memory_file.flush()

    def _monitor(self):
        while not self.stop.wait(self.interval):
            try:
                self._sample()
            except OSError:
                logging.getLogger(__name__).exception("Cannot write memory diagnostics")
                return

    def progress(self):
        if self.stall_seconds:
            faulthandler.dump_traceback_later(
                self.stall_seconds, repeat=True, file=self.trace_file
            )

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.memory_file = (self.directory / "memory.jsonl").open("a", encoding="utf-8")
        self.trace_file = (self.directory / "stacks.log").open("a", encoding="utf-8")
        self._sample()
        self.progress()
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        faulthandler.cancel_dump_traceback_later()
        self.stop.set()
        self.thread.join()
        try:
            self._sample()
        finally:
            self.memory_file.close()
            self.trace_file.close()


def atomic_torch_save(value, destination):
    """Keep the previous checkpoint if saving the replacement is interrupted."""
    import torch

    destination = Path(destination)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
