"""Tiny progress printer: prints every Nth message so logs stay readable."""
import time

_last = [0.0]


def progress(msg: str, every_seconds: float = 2.0):
    now = time.time()
    if now - _last[0] >= every_seconds:
        print(msg, flush=True)
        _last[0] = now
