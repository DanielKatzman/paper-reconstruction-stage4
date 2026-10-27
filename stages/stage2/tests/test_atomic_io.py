from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from paper_reconstruction_stage2.atomic_io import atomic_write_json


class AtomicIoTests(unittest.TestCase):
    def test_transient_windows_permission_error_is_retried(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "state.json"
            real_replace = __import__("os").replace
            attempts = 0

            def flaky_replace(source, destination):
                nonlocal attempts
                attempts += 1
                if attempts < 3:
                    raise PermissionError(5, "simulated transient scanner lock")
                return real_replace(source, destination)

            with mock.patch(
                "paper_reconstruction_stage2.atomic_io.os.replace",
                side_effect=flaky_replace,
            ), mock.patch("paper_reconstruction_stage2.atomic_io.time.sleep"):
                atomic_write_json(target, {"complete": 31617})

            self.assertEqual(attempts, 3)
            self.assertEqual(json.loads(target.read_text()), {"complete": 31617})
            self.assertEqual(list(target.parent.glob("*.tmp")), [])
