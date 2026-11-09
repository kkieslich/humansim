from __future__ import annotations

import fcntl
import hashlib
import os
import random
import tempfile
import urllib.request
from abc import ABC
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .. import LocalPlanner

_FETCH_TIMEOUT_SECONDS = 60.0


def policy_cache_dir(policy_name: str) -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "arena_humansim" / policy_name


@contextmanager
def cache_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def fetch_to_disk(url: str, dest: Path, timeout: float = _FETCH_TIMEOUT_SECONDS, sha256: str | None = None) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Per-call unique tmp path keeps concurrent workers from truncating each other's
    # in-flight downloads of the same checkpoint. Atomic os.replace at the end means
    # dest.exists() is only True for fully-written files.
    fd, tmp_name = tempfile.mkstemp(prefix=dest.name + ".", suffix=".part", dir=str(dest.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f, urllib.request.urlopen(url, timeout=timeout) as resp:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
        if sha256 is not None and hashlib.sha256(tmp.read_bytes()).hexdigest() != sha256:
            raise ValueError(f"{url} did not match sha256 {sha256}")
        os.replace(tmp, dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


class RobotPolicy(LocalPlanner, ABC):
    bypasses_kinematic_constraints: bool = True
    needs_global_subgoal: bool = True
    supports_pool: bool = False

    def _seed_rngs(self, seed: int) -> None:
        random.seed(seed)
        try:
            import numpy as np

            np.random.seed(seed & 0xFFFFFFFF)
        except ImportError:
            pass
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.use_deterministic_algorithms(True, warn_only=True)
        except ImportError:
            pass
