"""Wait within the original 45 seconds for complete owned Chromium metadata.

This polls the same process; it never starts a second browser or grants a retry.
"""
from pathlib import Path
import json
import re
import time

UUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"


def port_lines(raw):
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError:
        return None
    if (len(lines) != 2 or re.fullmatch(r"[0-9]{1,5}", lines[0]) is None
            or not 1 <= int(lines[0]) <= 65535
            or re.fullmatch("/devtools/browser/" + UUID, lines[1]) is None):
        return None
    return lines


def complete_port(proc, path, deadline, diagnostic):
    path, diagnostic = Path(path), Path(diagnostic)
    while True:
        now, status = time.monotonic(), proc.poll()
        try:
            raw, error = path.read_bytes(), None
        except FileNotFoundError:
            raw, error = b"", "FileNotFoundError"
        except OSError as exc:
            with diagnostic.open("a") as stream:
                stream.write(json.dumps(dict(monotonic=now, deadline=deadline,
                    process_returncode=status, read_error=type(exc).__name__, errno=exc.errno)) + "\n")
                stream.flush()
            raise
        lines = port_lines(raw)
        post_status, post_now = proc.poll(), time.monotonic()
        with diagnostic.open("a") as stream:
            stream.write(json.dumps(dict(monotonic=now, deadline=deadline,
                process_returncode=status, read_error=error, raw_hex=raw.hex(),
                raw_ascii=raw.decode("ascii", errors="backslashreplace"),
                post_read_monotonic=post_now, post_read_process_returncode=post_status)) + "\n")
            stream.flush()
        # Reads and diagnostic writes may stall. Qualify only after the final
        # process/time sample, including when complete bytes arrived too late.
        status, now = proc.poll(), time.monotonic()
        if status is not None or now > deadline:
            raise RuntimeError("Owned Chromium startup failed before complete DevToolsActivePort")
        if lines is not None:
            return lines
        if now >= deadline:
            raise RuntimeError("Owned Chromium original45s startup deadline without complete DevToolsActivePort")
        time.sleep(min(.1, max(0, deadline - now)))
