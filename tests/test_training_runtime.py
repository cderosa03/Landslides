import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.train_runtime import RunDiagnostics, atomic_torch_save, memory_snapshot


class RuntimeDiagnosticsTests(unittest.TestCase):
    def test_container_limit_cache_and_process_rss_are_distinct(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "self").mkdir()
            (root / "memory.current").write_text("68719476736")
            (root / "memory.max").write_text("68719476736")
            (root / "memory.stat").write_text("anon 1024\nfile 4096\nshmem 512\n")
            (root / "memory.events").write_text("oom 2\noom_kill 1\n")
            (root / "self/status").write_text("VmRSS:\t100 kB\nVmHWM:\t120 kB\n")
            sample = memory_snapshot(root, root)
            self.assertEqual(sample["memory.max"], 64 * 1024**3)
            self.assertEqual(sample["memory.stat"]["file"], 4096)
            self.assertEqual(sample["VmRSS_bytes"], 102400)
            self.assertEqual(sample["memory.events"]["oom_kill"], 1)

    def test_missing_linux_counters_and_unlimited_memory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "memory.max").write_text("max")
            self.assertIsNone(memory_snapshot(root, root)["memory.max"])

    def test_diagnostics_flush_and_close_when_training_raises(self):
        with tempfile.TemporaryDirectory() as temporary:
            monitor = RunDiagnostics(temporary, stall_seconds=0)
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                with monitor:
                    raise RuntimeError("training failed")
            samples = [json.loads(line) for line in (Path(temporary) / "memory.jsonl").read_text().splitlines()]
            self.assertGreaterEqual(len(samples), 2)
            self.assertFalse(monitor.thread.is_alive())

    def test_failed_checkpoint_write_preserves_previous_checkpoint(self):
        def interrupted_save(value, path):
            Path(path).write_bytes(b"incomplete")
            raise OSError("disk unavailable")

        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "checkpoint.pth"
            target.write_bytes(b"previous checkpoint")
            with patch("torch.save", side_effect=interrupted_save):
                with self.assertRaises(OSError):
                    atomic_torch_save({}, target)
            self.assertEqual(target.read_bytes(), b"previous checkpoint")
            self.assertFalse(target.with_name("checkpoint.pth.tmp").exists())


if __name__ == "__main__":
    unittest.main()
