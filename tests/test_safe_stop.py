from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from humanoidverse.training.safe_stop import request_safe_stop, safe_stop_request_path, safe_stop_status_path
from humanoidverse.training.workspace import Workspace


class SafeStopProtocolTest(unittest.TestCase):
    def test_request_safe_stop_writes_atomic_request(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            work_dir = Path(temporary_dir)
            request = request_safe_stop(work_dir, reason="test stop")

            written = json.loads(safe_stop_request_path(work_dir).read_text())
            self.assertEqual(written, request)
            self.assertEqual(written["reason"], "test stop")
            self.assertTrue(written["request_id"])

    def _workspace(self, work_dir: Path, *, checkpoint_global_time: int = 100) -> Workspace:
        workspace = Workspace.__new__(Workspace)
        workspace.cfg = SimpleNamespace(save_on_exit=True, distributed_sync=False)
        workspace.distributed_world_size = 1
        workspace._write_shared_artifacts = True
        workspace.work_dir = work_dir
        workspace._safe_stop_requested_by_signal = False
        workspace._safe_stop_reason = None
        workspace._safe_stop_request_id = None
        workspace._checkpoint_local_time = checkpoint_global_time
        workspace._checkpoint_global_time = checkpoint_global_time
        workspace.save = Mock()
        return workspace

    def test_request_saves_exact_complete_boundary_and_confirms(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            work_dir = Path(temporary_dir)
            request = request_safe_stop(work_dir, reason="overnight stop")
            workspace = self._workspace(work_dir)
            replay_buffer = {"train": object()}

            should_stop, last_saved = workspace._save_and_stop_if_requested(
                local_time=123,
                global_time=456,
                optimizer_steps=7,
                replay_buffer=replay_buffer,
                last_saved_global_time=100,
            )

            self.assertTrue(should_stop)
            self.assertEqual(last_saved, 456)
            workspace.save.assert_called_once_with(
                local_time=123,
                global_time=456,
                optimizer_steps=7,
                replay_buffer=replay_buffer,
            )
            self.assertFalse(safe_stop_request_path(work_dir).exists())
            status = json.loads(safe_stop_status_path(work_dir).read_text())
            self.assertEqual(status["status"], "saved_and_stopped")
            self.assertEqual(status["request_id"], request["request_id"])
            self.assertEqual(status["local_time"], 123)
            self.assertEqual(status["global_time"], 456)
            self.assertEqual(status["optimizer_steps"], 7)
            self.assertFalse(status["checkpoint_reused"])

    def test_request_reuses_checkpoint_at_same_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            work_dir = Path(temporary_dir)
            request_safe_stop(work_dir)
            workspace = self._workspace(work_dir, checkpoint_global_time=456)

            should_stop, last_saved = workspace._save_and_stop_if_requested(
                local_time=456,
                global_time=456,
                optimizer_steps=8,
                replay_buffer={"train": object()},
                last_saved_global_time=456,
            )

            self.assertTrue(should_stop)
            self.assertEqual(last_saved, 456)
            workspace.save.assert_not_called()
            status = json.loads(safe_stop_status_path(work_dir).read_text())
            self.assertTrue(status["checkpoint_reused"])

    def test_disabled_safe_stop_ignores_request(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            work_dir = Path(temporary_dir)
            request_safe_stop(work_dir)
            workspace = self._workspace(work_dir)
            workspace.cfg.save_on_exit = False

            should_stop, last_saved = workspace._save_and_stop_if_requested(
                local_time=123,
                global_time=456,
                optimizer_steps=7,
                replay_buffer={},
                last_saved_global_time=100,
            )

            self.assertFalse(should_stop)
            self.assertEqual(last_saved, 100)
            workspace.save.assert_not_called()
            self.assertTrue(safe_stop_request_path(work_dir).exists())


if __name__ == "__main__":
    unittest.main()
