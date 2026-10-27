from __future__ import annotations

import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def _replace_with_retry(
    temporary: Path,
    target: Path,
    *,
    attempts: int = 20,
    initial_delay_seconds: float = 0.05,
    maximum_delay_seconds: float = 0.75,
) -> None:
    """Replace a file atomically, tolerating transient Windows file locks."""

    delay = initial_delay_seconds
    for attempt in range(attempts):
        try:
            os.replace(temporary, target)
            return
        except PermissionError:
            if attempt + 1 >= attempts:
                raise
            time.sleep(delay)
            delay = min(delay * 1.6, maximum_delay_seconds)


@contextmanager
def atomic_output_path(path: Path, *, preserve_suffix: bool = False) -> Iterator[Path]:
    """Yield a unique same-directory path and atomically commit it on success."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix if preserve_suffix else ".tmp"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.stem}.", suffix=suffix, dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        yield temporary
        _replace_with_retry(temporary, target)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except PermissionError:
            # A scanner may still hold the abandoned temporary file briefly. It
            # is hidden, uniquely named, and never considered a commit marker.
            pass


def atomic_write_json(path: Path, value: Any) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with atomic_output_path(path) as temporary:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
