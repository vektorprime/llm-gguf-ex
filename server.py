from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
import time
import traceback
import urllib.parse
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from gguf_explorer.gguf import GgufError, GgufFile, write_sample_gguf
from gguf_explorer.optimize_q8_0 import OptimizationProgress, OptimizationSettings, optimize_q8_0_file


ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"


class AppState:
    def __init__(self) -> None:
        self.reader: GgufFile | None = None
        self.reference_reader: GgufFile | None = None
        self.lock = threading.RLock()

    def open(self, path: str) -> GgufFile:
        next_reader = GgufFile(path)
        with self.lock:
            if self.reader is not None:
                self.reader.close()
            self.reader = next_reader
        return next_reader

    def open_reference(self, path: str) -> GgufFile:
        next_reader = GgufFile(path)
        with self.lock:
            if self.reference_reader is not None:
                self.reference_reader.close()
            self.reference_reader = next_reader
        return next_reader

    def clear_reference(self) -> None:
        with self.lock:
            if self.reference_reader is not None:
                self.reference_reader.close()
            self.reference_reader = None

    def require_reader(self) -> GgufFile:
        with self.lock:
            if self.reader is None:
                raise GgufError("No GGUF file is open")
            return self.reader


STATE = AppState()
OPTIMIZATION_LOCK = threading.RLock()
OPTIMIZATION_JOB: dict[str, Any] | None = None
ACTIVE_OPTIMIZATION_STATUSES = {"queued", "preparing", "running"}


class ExplorerHandler(SimpleHTTPRequestHandler):
    server_version = "GGUFExplorer/0.1"

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self.handle_api_get(parsed)
            return
        self.serve_static(parsed.path)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self.handle_api_post(parsed)
            return
        self.send_error(404)

    def handle_api_get(self, parsed: urllib.parse.ParseResult) -> None:
        try:
            query = urllib.parse.parse_qs(parsed.query)
            if parsed.path == "/api/state":
                self.send_json(state_payload())
                return
            if parsed.path == "/api/tensor":
                reader = STATE.require_reader()
                name = one(query, "name")
                self.send_json(reader.tensor_detail(name, reference=STATE.reference_reader))
                return
            if parsed.path == "/api/tensor/consecutive_duplicates":
                reader = STATE.require_reader()
                name = one(query, "name")
                self.send_json(reader.count_consecutive_duplicates(name))
                return
            if parsed.path == "/api/values":
                reader = STATE.require_reader()
                name = one(query, "name")
                start = int(one(query, "start", "0"))
                count = int(one(query, "count", "64"))
                mode = one(query, "mode", "dequantized")
                self.send_json(
                    reader.sample_tensor(
                        name,
                        start=start,
                        count=count,
                        mode=mode,
                        reference=STATE.reference_reader,
                    )
                )
                return
            if parsed.path == "/api/models":
                directory_hint = one(query, "dir", "")
                self.send_json(discovered_models_payload(directory_hint))
                return
            if parsed.path == "/api/optimize/q8_0/status":
                self.send_json({"optimization_job": optimization_job_snapshot()})
                return
            if parsed.path == "/api/sample":
                sample_path = write_sample_gguf(ROOT / "samples" / "tiny-bf16-q8_0.gguf")
                reader = STATE.open(str(sample_path))
                self.send_json(state_payload(reader))
                return
            self.send_error(404)
        except Exception as exc:
            self.send_api_error(exc)

    def handle_api_post(self, parsed: urllib.parse.ParseResult) -> None:
        try:
            if parsed.path == "/api/open":
                body = self.read_json_body()
                path = str(body.get("path", "")).strip()
                if not path:
                    raise GgufError("Path is required")
                reader = STATE.open(path)
                self.send_json(state_payload(reader))
                return
            if parsed.path == "/api/reference/open":
                body = self.read_json_body()
                path = str(body.get("path", "")).strip()
                if not path:
                    raise GgufError("Reference path is required")
                STATE.open_reference(path)
                self.send_json(state_payload())
                return
            if parsed.path == "/api/reference/clear":
                STATE.clear_reference()
                self.send_json(state_payload())
                return
            if parsed.path == "/api/optimize/q8_0":
                reader = STATE.require_reader()
                if STATE.reference_reader is None:
                    raise GgufError("Load a reference GGUF before optimizing quantized tensors")
                body = self.read_json_body()
                passes = int(body.get("passes", 8))
                workers_raw = body.get("workers")
                workers = int(workers_raw) if workers_raw not in (None, "") else None
                chunk_blocks = int(body.get("chunk_blocks", 8192))
                parallelism = str(body.get("parallelism", "process")).strip().lower()
                if parallelism not in {"process", "thread", "none"}:
                    parallelism = "process"
                output_path = str(body.get("output_path", "")).strip() or None
                settings = OptimizationSettings(
                    passes=passes,
                    workers=workers,
                    chunk_blocks=chunk_blocks,
                    parallelism=parallelism,  # type: ignore[arg-type]
                )
                job = start_optimization_job(
                    reader.path,
                    STATE.reference_reader.path,
                    output_path,
                    settings,
                )
                self.send_json({"optimization_job": job}, status=202)
                return
            self.send_error(404)
        except Exception as exc:
            self.send_api_error(exc)

    def read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def serve_static(self, route_path: str) -> None:
        if route_path in {"", "/"}:
            file_path = STATIC / "index.html"
        else:
            relative = route_path.lstrip("/")
            file_path = (STATIC / relative).resolve()
            if not file_path.is_relative_to(STATIC.resolve()):
                self.send_error(403)
                return

        if not file_path.exists() or not file_path.is_file():
            self.send_error(404)
            return

        content_type, _ = mimetypes.guess_type(str(file_path))
        content = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, payload: Any, status: int = 200) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def send_api_error(self, exc: Exception) -> None:
        if not isinstance(exc, GgufError):
            traceback.print_exc()
        self.send_json({"error": str(exc)}, status=400 if isinstance(exc, GgufError) else 500)

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def start_optimization_job(
    target_path: str,
    reference_path: str,
    output_path: str | None,
    settings: OptimizationSettings,
) -> dict[str, Any]:
    global OPTIMIZATION_JOB
    now = time.time()
    job_id = uuid.uuid4().hex
    with OPTIMIZATION_LOCK:
        if OPTIMIZATION_JOB and OPTIMIZATION_JOB.get("status") in ACTIVE_OPTIMIZATION_STATUSES:
            raise GgufError("A quantization optimization job is already running")
        OPTIMIZATION_JOB = {
            "id": job_id,
            "active": True,
            "status": "queued",
            "error": "",
            "created_at": now,
            "updated_at": now,
            "progress": {
                "status": "queued",
                "message": "Queued quantization optimization",
                "total_blocks": 0,
                "processed_blocks": 0,
                "progress_percent": 0,
                "changed_blocks": 0,
                "previous_sse": 0,
                "optimized_sse": 0,
                "improvement": 0,
                "workers": settings.workers,
                "parallelism": settings.parallelism,
                "passes": settings.passes,
                "chunk_blocks": settings.chunk_blocks,
                "output_path": output_path or "",
                "current_tensor": None,
            },
            "result": None,
            "state_payload": None,
        }
        snapshot = _job_snapshot_locked()

    thread = threading.Thread(
        target=_run_optimization_job,
        args=(job_id, target_path, reference_path, output_path, settings),
        daemon=True,
    )
    thread.start()
    return snapshot


def _run_optimization_job(
    job_id: str,
    target_path: str,
    reference_path: str,
    output_path: str | None,
    settings: OptimizationSettings,
) -> None:
    def on_progress(progress: OptimizationProgress) -> None:
        with OPTIMIZATION_LOCK:
            if not OPTIMIZATION_JOB or OPTIMIZATION_JOB.get("id") != job_id:
                return
            OPTIMIZATION_JOB["status"] = progress.status
            OPTIMIZATION_JOB["progress"] = progress.to_json()
            OPTIMIZATION_JOB["updated_at"] = time.time()

    try:
        result = optimize_q8_0_file(
            target_path,
            reference_path,
            output_path,
            settings=settings,
            progress_callback=on_progress,
        )
        optimized_reader = STATE.open(result.output_path)
        payload = state_payload(optimized_reader)
        payload["optimization"] = result.to_json()
        with OPTIMIZATION_LOCK:
            if not OPTIMIZATION_JOB or OPTIMIZATION_JOB.get("id") != job_id:
                return
            OPTIMIZATION_JOB["active"] = False
            OPTIMIZATION_JOB["status"] = "complete"
            OPTIMIZATION_JOB["result"] = result.to_json()
            OPTIMIZATION_JOB["state_payload"] = payload
            OPTIMIZATION_JOB["updated_at"] = time.time()
    except Exception as exc:
        traceback.print_exc()
        with OPTIMIZATION_LOCK:
            if not OPTIMIZATION_JOB or OPTIMIZATION_JOB.get("id") != job_id:
                return
            OPTIMIZATION_JOB["active"] = False
            OPTIMIZATION_JOB["status"] = "error"
            OPTIMIZATION_JOB["error"] = str(exc)
            progress = dict(OPTIMIZATION_JOB.get("progress") or {})
            progress["status"] = "error"
            progress["message"] = str(exc)
            OPTIMIZATION_JOB["progress"] = progress
            OPTIMIZATION_JOB["updated_at"] = time.time()


def optimization_job_snapshot() -> dict[str, Any]:
    with OPTIMIZATION_LOCK:
        return _job_snapshot_locked()


def _job_snapshot_locked() -> dict[str, Any]:
    if OPTIMIZATION_JOB is None:
        return {"active": False, "status": "idle"}
    snapshot = dict(OPTIMIZATION_JOB)
    snapshot["progress"] = dict(OPTIMIZATION_JOB.get("progress") or {})
    if OPTIMIZATION_JOB.get("result") is not None:
        snapshot["result"] = dict(OPTIMIZATION_JOB["result"])
    if OPTIMIZATION_JOB.get("state_payload") is not None:
        snapshot["state_payload"] = OPTIMIZATION_JOB["state_payload"]
    return snapshot


def state_payload(reader: GgufFile | None = None) -> dict[str, Any]:
    with STATE.lock:
        active_reader = reader if reader is not None else STATE.reader
        payload: dict[str, Any]
        if active_reader is None:
            payload = {"open": False}
        else:
            payload = {
                "open": True,
                "file": active_reader.summary(),
                "metadata": active_reader.metadata_json(),
                "tree": active_reader.tree(),
            }
        payload["reference"] = reference_payload(STATE.reference_reader)
        return payload


def reference_payload(reader: GgufFile | None) -> dict[str, Any]:
    if reader is None:
        return {"open": False}
    return {"open": True, "file": reader.summary()}


def discovered_models_payload(directory_hint: str = "") -> dict[str, Any]:
    directory = model_scan_directory(directory_hint)
    models = [model_file_payload(path) for path in sorted(directory.glob("*.gguf"), key=lambda item: item.name.lower())]
    return {
        "directory": str(directory),
        "count": len(models),
        "models": models,
    }


def model_scan_directory(directory_hint: str) -> Path:
    raw = directory_hint.strip()
    if not raw:
        return ROOT
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    candidate = candidate.resolve()
    if candidate.is_file():
        return candidate.parent
    if candidate.exists() and candidate.is_dir():
        return candidate
    if candidate.suffix.lower() == ".gguf" and candidate.parent.exists():
        return candidate.parent
    raise GgufError(f"Folder does not exist: {candidate}")


def model_file_payload(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "path": str(path),
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "modified": path.stat().st_mtime,
    }
    reader: GgufFile | None = None
    try:
        reader = GgufFile(path)
        summary = reader.summary()
        payload.update(
            {
                "model_name": summary.get("model_name"),
                "architecture": summary.get("architecture"),
                "tensor_count": summary.get("tensor_count"),
                "type_counts": summary.get("type_counts", {}),
                "primary_type": primary_tensor_type(summary.get("type_counts", {})),
                "version": summary.get("version"),
                "error": None,
            }
        )
    except Exception as exc:
        payload["error"] = str(exc)
    finally:
        if reader is not None:
            reader.close()
    return payload


def primary_tensor_type(type_counts: Any) -> str | None:
    if not isinstance(type_counts, dict) or not type_counts:
        return None
    quantized = {name: count for name, count in type_counts.items() if name != "F32"}
    source = quantized or type_counts
    return max(source.items(), key=lambda item: item[1])[0]


def one(query: dict[str, list[str]], key: str, default: str | None = None) -> str:
    values = query.get(key)
    if not values:
        if default is None:
            raise GgufError(f"Missing query parameter: {key}")
        return default
    return values[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local GGUF visual explorer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--open", dest="open_path", default=None, help="GGUF file to open at startup")
    parser.add_argument("--reference", dest="reference_path", default=None, help="Reference GGUF for value diffs")
    args = parser.parse_args()

    if args.open_path:
        STATE.open(args.open_path)
    if args.reference_path:
        STATE.open_reference(args.reference_path)

    server = ThreadingHTTPServer((args.host, args.port), ExplorerHandler)
    print(f"GGUF Explorer listening at http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if STATE.reader is not None:
            STATE.reader.close()
        STATE.clear_reference()
        server.server_close()


if __name__ == "__main__":
    main()
