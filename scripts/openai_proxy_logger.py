from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def _utc_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def _redact_headers(headers: dict[str, str]) -> dict[str, str]:
    out = dict(headers)
    for k in list(out.keys()):
        lk = k.lower()
        if lk in {"authorization", "proxy-authorization"}:
            out[k] = "<redacted>"
        if lk in {"x-api-key"}:
            out[k] = "<redacted>"
    return out


def _try_decode_json(data: bytes) -> Any:
    try:
        # Best-effort: if this looks like gzipped content, decompress for logging.
        if len(data) >= 2 and data[0] == 0x1F and data[1] == 0x8B:
            data = gzip.decompress(data)
        return json.loads(data.decode("utf-8"))
    except Exception:
        return None


class _State:
    def __init__(self, *, upstream: str, log_dir: Path):
        self.upstream = upstream.rstrip("/")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._counter = 0

    def next_id(self) -> int:
        with self._lock:
            self._counter += 1
            return self._counter


STATE: _State | None = None


class ProxyHandler(BaseHTTPRequestHandler):
    server_version = "OpenAIProxyLogger/0.1"

    def _read_body(self) -> bytes:
        n = int(self.headers.get("content-length", "0") or "0")
        return self.rfile.read(n) if n > 0 else b""

    def _forward(self, body: bytes) -> tuple[int, dict[str, str], bytes]:
        assert STATE is not None
        url = STATE.upstream + self.path

        # Copy headers, but remove hop-by-hop and host.
        req_headers: dict[str, str] = {}
        for k, v in self.headers.items():
            lk = k.lower()
            if lk in {"host", "content-length", "connection", "keep-alive", "proxy-connection"}:
                continue
            req_headers[k] = v

        req = urllib.request.Request(url=url, data=body if body else None, method=self.command)
        for k, v in req_headers.items():
            req.add_header(k, v)

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                status = int(getattr(resp, "status", 200))
                headers = {k: v for k, v in resp.headers.items()}
                data = resp.read()
                return status, headers, data
        except urllib.error.HTTPError as e:
            data = e.read() if hasattr(e, "read") else b""
            headers = {k: v for k, v in getattr(e, "headers", {}).items()} if getattr(e, "headers", None) else {}
            return int(e.code), headers, data

    def _write_log(
        self,
        *,
        req_body: bytes,
        resp_status: int,
        resp_headers: dict[str, str],
        resp_body: bytes,
        elapsed_s: float,
    ) -> None:
        assert STATE is not None
        req_headers = {k: v for k, v in self.headers.items()}
        log_obj: dict[str, Any] = {
            "ts_utc": _utc_ts(),
            "request": {
                "id": STATE.next_id(),
                "method": self.command,
                "path": self.path,
                "upstream": STATE.upstream,
                "headers": _redact_headers(req_headers),
                "json": _try_decode_json(req_body),
                "raw_bytes_len": len(req_body),
            },
            "response": {
                "status": resp_status,
                "headers": _redact_headers(resp_headers),
                "json": _try_decode_json(resp_body),
                "raw_bytes_len": len(resp_body),
            },
            "timing": {"elapsed_s": elapsed_s},
        }

        # Filename: counter + timestamp + method
        rid = log_obj["request"]["id"]
        fname = f"{rid:06d}_{log_obj['ts_utc']}_{self.command}.json"
        (STATE.log_dir / fname).write_text(
            json.dumps(log_obj, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _handle(self) -> None:
        assert STATE is not None
        start = time.time()
        req_body = self._read_body()
        status, headers, data = self._forward(req_body)
        elapsed = time.time() - start

        # Log before responding
        try:
            self._write_log(
                req_body=req_body,
                resp_status=status,
                resp_headers=headers,
                resp_body=data,
                elapsed_s=elapsed,
            )
        except Exception:
            # Never break the proxy due to logging issues
            pass

        self.send_response(status)
        for k, v in headers.items():
            lk = k.lower()
            # We'll set Content-Length ourselves below. Keep Content-Encoding so clients can decompress.
            if lk in {"transfer-encoding", "content-length", "connection"}:
                continue
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802
        self._handle()

    def do_GET(self) -> None:  # noqa: N802
        self._handle()

    def do_PUT(self) -> None:  # noqa: N802
        self._handle()

    def do_DELETE(self) -> None:  # noqa: N802
        self._handle()

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep stdout clean; logs go to files.
        return


def main() -> None:
    p = argparse.ArgumentParser(description="OpenAI-compatible proxy that logs requests/responses to disk")
    p.add_argument("--listen-host", default="127.0.0.1")
    p.add_argument("--listen-port", type=int, default=8801)
    p.add_argument("--upstream", required=True, help="Upstream OpenAI-compatible base URL (e.g. https://openrouter.ai/api/v1)")
    p.add_argument("--log-dir", required=True, help="Directory to write JSON logs into")
    args = p.parse_args()

    global STATE  # noqa: PLW0603
    STATE = _State(upstream=str(args.upstream), log_dir=Path(args.log_dir))

    # Helpful header if needed:
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")

    server = ThreadingHTTPServer((args.listen_host, int(args.listen_port)), ProxyHandler)
    sys.stderr.write(
        f"[openai_proxy_logger] listening on http://{args.listen_host}:{args.listen_port} -> {STATE.upstream}\n"
    )
    sys.stderr.flush()
    server.serve_forever()


if __name__ == "__main__":
    main()


