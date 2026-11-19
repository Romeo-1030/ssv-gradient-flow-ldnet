"""Atomic JSON writes that tolerate brief Windows reader/antivirus locks."""
import json
import os
from pathlib import Path
import time
import uuid


def save_json(path, value, timeout=5.):
    path = Path(path)
    # Separate writers must not share a temporary filename.
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    contents = json.dumps(value, indent=2)+'\n'
    deadline = time.monotonic()+timeout
    delay = .01
    try:
        while True:
            try:
                temporary.write_text(contents, encoding='utf-8')
                break
            except PermissionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(delay)
                delay = min(.2, delay*2)
        while True:
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(delay)
                delay = min(.2, delay*2)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except PermissionError:
            pass
