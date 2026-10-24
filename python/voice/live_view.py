"""Live view of a file in the browser that highlights each change. Only watches the file on disk.

uv run python -m voice.live_view path/to/file [--port 8765]   ->   http://localhost:8765
"""

import argparse
import difflib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = Path(__file__).with_name("live_view.html")


class FileWatcher:
    """Polls a file; keeps the version at start (or reset), the one before the last change, and the current one."""

    def __init__(self, path: Path, interval: float = 0.1):
        self.path = path
        self.interval = interval
        self.baseline = self.previous = self.current = path.read_text()
        self.version = 0
        self.changed_at: float | None = None
        self._changed = threading.Condition()

    def run(self) -> None:
        while True:
            time.sleep(self.interval)
            try:
                text = self.path.read_text()
            except FileNotFoundError:  # mid atomic write
                continue
            if text != self.current:
                with self._changed:
                    self.previous, self.current = self.current, text
                    self.version += 1
                    self.changed_at = time.time()
                    self._changed.notify_all()

    def reset(self) -> None:
        with self._changed:
            self.baseline = self.previous = self.current
            self.version += 1
            self._changed.notify_all()

    def wait(self, version: int, timeout: float) -> bool:
        with self._changed:
            return self._changed.wait_for(lambda: self.version != version, timeout)

    def snapshot(self) -> dict:
        return {
            "version": self.version,
            "path": str(self.path),
            "changed_at": self.changed_at,
            "last": diff_rows(self.previous, self.current),
            "total": diff_rows(self.baseline, self.current),
        }


def diff_rows(before: str, after: str) -> list[list]:
    """The whole file as rows [kind, old line, new line, text]; kind is same, add or del."""
    old, new = before.splitlines(), after.splitlines()
    rows = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if op == "equal":
            rows += [["same", i1 + k + 1, j1 + k + 1, new[j1 + k]] for k in range(i2 - i1)]
            continue
        rows += [["del", i + 1, None, old[i]] for i in range(i1, i2)]
        rows += [["add", None, j + 1, new[j]] for j in range(j1, j2)]
    return rows


def make_handler(watcher: FileWatcher):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                self._send(200, "text/html; charset=utf-8", PAGE.read_bytes())
            elif self.path == "/events":
                self._stream()
            else:
                self._send(404, "text/plain", b"not found")

        def do_POST(self):
            if self.path == "/reset":
                watcher.reset()
                self._send(204, "text/plain", b"")
            else:
                self._send(404, "text/plain", b"not found")

        def _stream(self):  # server-sent events: a snapshot per change
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            version = None
            try:
                while True:
                    if version is None or watcher.wait(version, timeout=15):
                        snapshot = watcher.snapshot()
                        version = snapshot["version"]
                        self.wfile.write(f"data: {json.dumps(snapshot)}\n\n".encode())
                    else:
                        self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _send(self, status: int, content_type: str, body: bytes):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Show a file in the browser and highlight changes live.")
    parser.add_argument("path")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    watcher = FileWatcher(Path(args.path))
    threading.Thread(target=watcher.run, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(watcher))
    server.daemon_threads = True
    print(f"Live view of {args.path} on http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
