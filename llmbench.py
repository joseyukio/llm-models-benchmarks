#!/usr/bin/env python3
"""llmbench - local LLM speed, VRAM and context benchmark harness.

Runs GGUF models through a llama-server-compatible backend, measures prompt
processing / generation throughput, maps context-vs-VRAM limits, and writes a
Markdown report. Standard library only.

See README.md for usage and config.toml for machine-specific settings.
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import json
import os
import re
import socket
import statistics
import struct
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.toml"

# Same filler used for earlier manual benchmarks (~1.2k tokens), so new numbers
# stay comparable with historical ones.
FILLER = (
    "The quick brown fox jumps over the lazy dog near the riverbank while the "
    "morning sun rises slowly over the distant hills. "
    * 50
)
SHORT_PROMPT = "Write one sentence about the ocean."
SANITY_PROMPT = (
    "A train travels 240 km in 2.5 hours, then 180 km in 1.5 hours. "
    "What is its average speed for the whole trip? Answer with the number and units."
)

VERSION = "0.1.0"
MIB = 1024.0 * 1024.0


# --------------------------------------------------------------------------- #
# utilities
# --------------------------------------------------------------------------- #

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def expand_path(p: str | Path) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(str(p))))


def resolve_path(p: str | Path, base: Path = ROOT) -> Path:
    q = expand_path(p)
    if not q.is_absolute():
        q = base / q
    return q.resolve()


def slugify(name: str) -> str:
    s = re.sub(r"\.gguf$", "", name, flags=re.I)
    s = re.sub(r"-\d{5}-of-\d{5}$", "", s)
    s = re.sub(r"[^a-z0-9._]+", "-", s.lower())
    return s.strip("-._") or "model"


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def rel_to_root(path: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def fmt_num(v, digits: int = 1, dash: str = "-") -> str:
    if isinstance(v, (int, float)):
        return f"{v:,.{digits}f}"
    return dash


def fmt_gib(mib) -> str:
    if isinstance(mib, (int, float)):
        return f"{mib / 1024.0:.1f} GB"
    return "-"


def ctx_label(ctx: int, k: str | None = None) -> str:
    base = f"{ctx // 1024}k" if ctx % 1024 == 0 else str(ctx)
    return f"{base} {k}" if k else base


def branch_name() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(ROOT), "branch", "--show-current"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        ).strip()
        return out or "main"
    except Exception:
        return "main"


# --------------------------------------------------------------------------- #
# system / GPU
# --------------------------------------------------------------------------- #

def gpu_info() -> dict:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=10,
        )
        line = out.strip().splitlines()[0]
        name, driver, total, used = [x.strip() for x in line.split(",")]
        return {
            "name": name,
            "driver": driver,
            "memory_total_mib": int(float(total)),
            "memory_used_mib": int(float(used)),
        }
    except Exception as exc:
        return {"name": "unknown", "driver": "unknown",
                "memory_total_mib": 0, "memory_used_mib": 0, "error": str(exc)}


def gpu_mem_used_mib() -> int:
    return int(gpu_info().get("memory_used_mib") or 0)


def gpu_mem_samples(n: int = 3, delay: float = 0.35) -> int:
    vals = []
    for i in range(max(1, n)):
        vals.append(gpu_mem_used_mib())
        if i < n - 1:
            time.sleep(delay)
    return max(vals)


def system_info() -> dict:
    cpu = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    mem_gib = 0.0
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                mem_gib = round(int(line.split()[1]) / 1024 / 1024, 1)
                break
    except Exception:
        pass
    os_name = "unknown"
    kernel = "unknown"
    try:
        os_name = next(
            ln.split("=", 1)[1].strip('"')
            for ln in Path("/etc/os-release").read_text().splitlines()
            if ln.startswith("PRETTY_NAME=")
        )
    except Exception:
        pass
    try:
        kernel = os.uname().release
    except Exception:
        pass
    return {
        "host": socket.gethostname(),
        "cpu": cpu,
        "ram_gib": mem_gib,
        "os": os_name,
        "kernel": kernel,
    }


# --------------------------------------------------------------------------- #
# GGUF metadata (architecture, native context length)
# --------------------------------------------------------------------------- #

_GGUF_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
             6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_GGUF_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _read_string(f) -> str:
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", "replace")


def _read_scalar(f, t: int):
    fmt = _GGUF_FMT[t]
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


def _skip_value(f, t: int) -> None:
    if t == 8:  # string
        (n,) = struct.unpack("<Q", f.read(8))
        f.seek(n, 1)
    elif t == 9:  # array
        (et,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        if et == 8:
            for _ in range(n):
                (ln,) = struct.unpack("<Q", f.read(8))
                f.seek(ln, 1)
        elif et == 9:
            for _ in range(n):
                _skip_value(f, 9)
        else:
            f.seek(_GGUF_SIZE[et] * n, 1)
    else:
        f.seek(_GGUF_SIZE[t], 1)


def read_gguf_meta(path: Path) -> dict:
    """Read only the handful of metadata keys we need. Stops early, before the
    tokenizer arrays, so this is fast even for large models."""
    meta = {"arch": None, "native_ctx": None, "block_count": None,
            "file_type": None, "gguf_version": None, "error": None}
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                raise ValueError("not a GGUF file")
            (version,) = struct.unpack("<I", f.read(4))
            _n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
            meta["gguf_version"] = version
            arch = None
            have_ctx = False
            have_block = False
            for _ in range(n_kv):
                key = _read_string(f)
                (vt,) = struct.unpack("<I", f.read(4))
                wanted = (
                    key.endswith(".context_length")
                    or key.endswith(".block_count")
                    or key in {"general.architecture", "general.file_type"}
                )
                if wanted and vt != 9:
                    val = _read_string(f) if vt == 8 else _read_scalar(f, vt)
                    if key == "general.architecture":
                        arch = str(val)
                        meta["arch"] = arch
                    elif key == "general.file_type":
                        meta["file_type"] = val
                    elif key.endswith(".block_count") and isinstance(val, int):
                        meta["block_count"] = int(val)
                        have_block = True
                    elif key.endswith(".context_length") and isinstance(val, int):
                        if val > 0:
                            meta["native_ctx"] = val
                            have_ctx = True
                else:
                    _skip_value(f, vt)
                if arch and have_ctx and have_block:
                    break
    except Exception as exc:
        meta["error"] = str(exc)
    return meta


_QUANT_RE = re.compile(
    r"(PQ2_0|PTQ1_0|IQ\d+_[A-Z0-9]+|Q\d+_K_[SML]|Q\d+_K|Q\d+_\d_\d|Q\d+_\d|BF16|F16|F32)",
    re.I,
)


def quant_from_name(name: str) -> str:
    m = _QUANT_RE.search(name)
    return m.group(1).upper() if m else "?"


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #

@dataclass
class Model:
    path: Path
    slug: str
    name: str
    size_bytes: int
    quant: str
    arch: str | None
    native_ctx: int | None
    block_count: int | None = None
    meta_error: str | None = None

    @property
    def size_gib(self) -> float:
        return self.size_bytes / (1024 ** 3)

    def as_dict(self) -> dict:
        return {
            "slug": self.slug,
            "name": self.name,
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "size_gib": round(self.size_gib, 2),
            "quant": self.quant,
            "arch": self.arch,
            "native_ctx": self.native_ctx,
            "block_count": self.block_count,
        }


def make_model(path: Path) -> Model:
    meta = read_gguf_meta(path)
    return Model(
        path=path.resolve(),
        slug=slugify(path.stem),
        name=path.stem,
        size_bytes=path.stat().st_size,
        quant=quant_from_name(path.name),
        arch=meta["arch"],
        native_ctx=meta["native_ctx"],
        block_count=meta["block_count"],
        meta_error=meta["error"],
    )


def discover_models(cfg: dict, include_mmproj: bool = False) -> list[Model]:
    roots = [resolve_path(r) for r in cfg.get("general", {}).get("model_roots", [])]
    models: list[Model] = []
    seen_paths: set[Path] = set()
    seen_slugs: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            print(f"warning: model root does not exist: {root}", file=sys.stderr)
            continue
        for path in sorted(root.rglob("*.gguf")):
            if not include_mmproj and path.name.lower().startswith("mmproj"):
                continue
            rp = path.resolve()
            if rp in seen_paths:
                continue
            seen_paths.add(rp)
            model = make_model(path)
            if model.slug in seen_slugs:
                model.slug = f"{slugify(model.path.parent.name)}-{model.slug}"
            base = model.slug
            n = 2
            while model.slug in seen_slugs:
                model.slug = f"{base}-{n}"
                n += 1
            seen_slugs[model.slug] = model.path
            models.append(model)
    return models


def find_model(cfg: dict, query: str) -> Model:
    p = expand_path(query)
    if p.is_file() and p.suffix.lower() == ".gguf":
        return make_model(p)
    models = discover_models(cfg)
    exact = [m for m in models
             if query in (m.slug, m.name, m.path.name, str(m.path))]
    if len(exact) == 1:
        return exact[0]
    q = query.lower()
    partial = [m for m in models
               if q in m.slug.lower() or q in m.name.lower()
               or q in str(m.path).lower()]
    if len(partial) == 1:
        return partial[0]
    if not partial:
        raise SystemExit(f"no model matches {query!r}; run `llmbench list`")
    listing = "\n".join(f"  {m.slug}" for m in partial)
    raise SystemExit(f"ambiguous model {query!r}; matches:\n{listing}")


def fit_estimate(model: Model, gpu: dict, baseline_mib: int,
                 overhead_mib: float = 900.0) -> tuple[bool, float, float]:
    avail = max(0.0, gpu.get("memory_total_mib", 0) - baseline_mib)
    need = model.size_bytes / MIB + overhead_mib
    return need <= avail, need, avail


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #

@dataclass
class Backend:
    name: str
    binary: Path
    ld_library_path: Path | None
    flags: list[str]


def load_config(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def backend_env(backend: Backend) -> dict:
    env = os.environ.copy()
    if backend.ld_library_path:
        prev = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = str(backend.ld_library_path) + ((":" + prev) if prev else "")
    return env


def resolve_backend(cfg: dict, name: str) -> Backend:
    table = cfg.get("backends", {}).get(name)
    if table is None:
        known = ", ".join(cfg.get("backends", {})) or "(none)"
        raise SystemExit(f"unknown backend {name!r}; configured: {known}")
    if "binary" in table:
        binary = resolve_path(table["binary"])
    else:
        pattern = os.path.expanduser(str(table.get("binary_glob", "")))
        matches = sorted(Path(x) for x in glob.glob(pattern) if Path(x).is_file())
        if not matches:
            raise SystemExit(f"backend {name!r}: no binary matched {pattern}")
        binary = max(matches, key=lambda p: p.stat().st_mtime)
    if not binary.exists():
        raise SystemExit(f"backend {name!r}: binary not found: {binary}")
    ld = table.get("ld_library_path")
    return Backend(
        name=name,
        binary=binary,
        ld_library_path=resolve_path(ld) if ld else None,
        flags=list(table.get("flags", [])),
    )


def select_backend_name(cfg: dict, model: Model, override: str | None = None) -> str:
    if override:
        return override
    for name, table in cfg.get("backends", {}).items():
        for pat in table.get("match", []):
            if fnmatch.fnmatch(model.path.name, pat):
                return name
    default = cfg.get("general", {}).get("default_backend")
    if default:
        return default
    backends = list(cfg.get("backends", {}))
    if not backends:
        raise SystemExit("no backends configured")
    return backends[0]


_VERSION_CACHE: dict[str, str] = {}


def backend_version(backend: Backend) -> str:
    key = str(backend.binary)
    if key in _VERSION_CACHE:
        return _VERSION_CACHE[key]
    try:
        out = subprocess.run(
            [str(backend.binary), "--version"],
            capture_output=True, text=True, env=backend_env(backend), timeout=30,
        )
        text = (out.stdout or out.stderr).strip()
        version = text.splitlines()[0] if text else "unknown"
    except Exception as exc:
        version = f"unknown ({exc})"
    _VERSION_CACHE[key] = version
    return version


# --------------------------------------------------------------------------- #
# llama-server process
# --------------------------------------------------------------------------- #

class ServerError(RuntimeError):
    def __init__(self, message: str, *, kind: str = "load-error",
                 log_path: Path | None = None, detail: str = ""):
        super().__init__(message)
        self.kind = kind
        self.log_path = log_path
        self.detail = detail


def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def classify_failure(text: str) -> str:
    low = text.lower()
    if "out of memory" in low or "failed to allocate" in low or "cudamalloc failed" in low:
        return "OOM"
    if "libcudart" in low or "libcublas" in low or "error while loading shared libraries" in low:
        return "missing-runtime"
    if "unknown model architecture" in low or "unsupported model" in low:
        return "unsupported-model"
    if "failed to create context" in low:
        return "context-failed"
    if "error loading model" in low or "failed to load model" in low:
        return "load-error"
    return "server-exit"


def layer_summary(model: Model, backend: Backend, info: dict) -> dict:
    """Which layers ended up on the GPU.

    The log value is authoritative when the backend prints it (Prism/stock
    llama.cpp). LM Studio builds suppress loader logs, so fall back to
    block_count vs the requested -ngl."""
    off = info.get("layers_offloaded")
    tot = info.get("layers_total")
    ngl = None
    flags = list(backend.flags)
    if "-ngl" in flags:
        try:
            ngl = int(flags[flags.index("-ngl") + 1])
        except (ValueError, IndexError):
            ngl = None
    if off is not None and tot:
        full = off >= tot
    elif model.block_count is not None and ngl is not None:
        full = ngl >= model.block_count + 1
    else:
        full = None
    return {"offloaded": off, "total": tot, "block_count": model.block_count,
            "ngl": ngl, "full_gpu": full}


class Server:
    """A llama-server subprocess bound to a random localhost port."""

    def __init__(self, cfg: dict, model: Model, backend: Backend,
                 ctx: int, k: str, v: str, extra_flags: list[str] | None = None):
        self.cfg = cfg
        self.model = model
        self.backend = backend
        self.ctx = ctx
        self.k = k
        self.v = v
        self.extra_flags = list(extra_flags or [])
        self.port = pick_free_port()
        results_dir = resolve_path(cfg.get("general", {}).get("results_dir", "results"))
        log_dir = results_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_path = log_dir / f"{stamp}-{model.slug}-c{ctx}-{k}-{self.port}.log"
        self.proc: subprocess.Popen | None = None
        self.log_fh = None
        self.ready_s: float | None = None
        self._stopped = False

    def _cmd(self) -> list[str]:
        return [
            str(self.backend.binary),
            "-m", str(self.model.path),
            "--host", "127.0.0.1", "--port", str(self.port),
            *self.backend.flags,
            "-c", str(self.ctx),
            "-ctk", self.k,
            "-ctv", self.v,
            *self.extra_flags,
        ]

    def _close_log(self) -> None:
        if self.log_fh is not None:
            try:
                self.log_fh.flush()
                self.log_fh.close()
            except Exception:
                pass
            self.log_fh = None

    def _log_text(self) -> str:
        self._close_log()
        try:
            return self.log_path.read_text(errors="replace")
        except Exception:
            return ""

    def _failure(self, message: str) -> ServerError:
        text = self._log_text()
        detail = "\n".join(text.strip().splitlines()[-8:])
        return ServerError(message, kind=classify_failure(text),
                           log_path=self.log_path, detail=detail)

    def start(self) -> "Server":
        timeout = float(self.cfg.get("general", {}).get("load_timeout_s", 240))
        self.log_fh = open(self.log_path, "wb")
        self.proc = subprocess.Popen(
            self._cmd(),
            stdout=self.log_fh,
            stderr=subprocess.STDOUT,
            env=backend_env(self.backend),
            start_new_session=True,
            cwd=str(ROOT),
        )
        t0 = time.time()
        health = f"http://127.0.0.1:{self.port}/health"
        while time.time() - t0 < timeout:
            rc = self.proc.poll()
            if rc is not None:
                self._close_log()
                raise self._failure(f"server exited with code {rc} before becoming ready")
            try:
                with urllib.request.urlopen(health, timeout=1.5) as resp:
                    if resp.status == 200:
                        self.ready_s = time.time() - t0
                        time.sleep(0.25)
                        return self
            except Exception:
                pass
            time.sleep(0.4)
        self.stop()
        text = self._log_text()
        raise ServerError(
            f"timed out after {timeout:.0f}s waiting for server",
            kind="timeout", log_path=self.log_path,
            detail="\n".join(text.strip().splitlines()[-8:]),
        )

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            except Exception:
                pass
        self._close_log()
        if not self._stopped:
            cooldown = float(self.cfg.get("general", {}).get("cooldown_s", 1.5))
            time.sleep(cooldown)
            self._stopped = True

    def sample_vram(self) -> int:
        return gpu_mem_samples(3, 0.3)

    def log_info(self) -> dict:
        try:
            text = self.log_path.read_text(errors="replace")
        except Exception:
            text = ""
        info = {"layers_offloaded": None, "layers_total": None,
                "n_ctx": None, "buffers_mib": {}}
        m = re.search(r"offloaded\s+(\d+)/(\d+)\s+layers to GPU", text)
        if m:
            info["layers_offloaded"] = int(m.group(1))
            info["layers_total"] = int(m.group(2))
        m = re.search(r"n_ctx(?:_slot)?\s*=\s*(\d+)", text)
        if m:
            info["n_ctx"] = int(m.group(1))
        for label in ("model", "KV", "compute"):
            m = re.search(rf"CUDA0 {label} buffer size\s*=\s*([\d.]+) MiB", text, re.I)
            if m:
                info["buffers_mib"][label.lower()] = float(m.group(1))
        return info


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #

def post_json(url: str, payload: dict, timeout: float = 900) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:800]
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def completion(port: int, payload: dict, timeout: float = 900) -> dict:
    return post_json(f"http://127.0.0.1:{port}/completion", payload, timeout)


def chat_completion(port: int, payload: dict, timeout: float = 900) -> dict:
    return post_json(f"http://127.0.0.1:{port}/v1/chat/completions", payload, timeout)


def completion_payload(prompt: str, n_predict: int, ignore_eos: bool = True) -> dict:
    return {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "top_p": 1.0,
        "cache_prompt": False,
        "stream": False,
        "ignore_eos": ignore_eos,
    }


def normalize_response(resp: dict) -> dict:
    t = resp.get("timings") or {}
    return {
        "prompt_n": t.get("prompt_n"),
        "prompt_ms": t.get("prompt_ms"),
        "prompt_per_second": t.get("prompt_per_second"),
        "predicted_n": t.get("predicted_n"),
        "predicted_ms": t.get("predicted_ms"),
        "predicted_per_second": t.get("predicted_per_second"),
        "stop_type": resp.get("stop_type"),
    }


def stats(values: list) -> dict:
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return {"median": None, "min": None, "max": None,
                "stdev": None, "n": 0}
    return {
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
        "stdev": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "n": len(vals),
    }


# --------------------------------------------------------------------------- #
# benchmarks
# --------------------------------------------------------------------------- #

def run_bench(cfg: dict, model: Model, backend: Backend,
              ctx: int, kv: str, reps: int, gen_tokens: int) -> dict:
    results_dir = resolve_path(cfg.get("general", {}).get("results_dir", "results"))
    baseline = gpu_mem_used_mib()
    gpu = gpu_info()
    sysinfo = system_info()
    server = Server(cfg, model, backend, ctx, kv, kv)

    print(f"  backend {backend.name}: {backend.binary}")
    print(f"  context {ctx} / KV {kv} / {reps} reps / {gen_tokens} generated tokens")
    server.start()
    print(f"  server ready in {server.ready_s:.1f}s on port {server.port}")
    try:
        warmup_short = normalize_response(
            completion(server.port, completion_payload(SHORT_PROMPT, 16)))
        warmup_prefill = normalize_response(
            completion(server.port, completion_payload(FILLER, 1)))
        print(f"  warm-up (discarded): prefill "
              f"{fmt_num(warmup_prefill['prompt_per_second'])} t/s")
        loaded_mib = server.sample_vram()
        runs = []
        for i in range(reps):
            rec = normalize_response(
                completion(server.port, completion_payload(FILLER, gen_tokens)))
            runs.append(rec)
            print(f"  rep {i + 1}/{reps}: prefill "
                  f"{fmt_num(rec['prompt_per_second'])} t/s, generation "
                  f"{fmt_num(rec['predicted_per_second'])} t/s "
                  f"({rec['predicted_n']} tok)")
        info = server.log_info()
    finally:
        server.stop()

    pp = stats([r["prompt_per_second"] for r in runs])
    tg = stats([r["predicted_per_second"] for r in runs])
    prompt_n = stats([r["prompt_n"] for r in runs])
    layers = layer_summary(model, backend, info)

    warnings: list[str] = []
    if layers["full_gpu"] is False:
        warnings.append(
            f"possible partial GPU offload (ngl {layers['ngl']}, "
            f"blocks {layers['block_count']}, log "
            f"{layers['offloaded']}/{layers['total']})")
    if baseline > 2000:
        warnings.append(f"high desktop VRAM baseline ({baseline} MiB); "
                        "close other GPU apps for comparable numbers")
    if any((r["predicted_n"] or 0) < gen_tokens for r in runs):
        warnings.append("generation stopped before the requested token count")
    for label, st in (("prefill", pp), ("generation", tg)):
        if st["median"] and st["stdev"] and st["stdev"] / st["median"] > 0.10:
            warnings.append(f"{label} variability >10% across reps")

    result = {
        "schema": 1,
        "tool_version": VERSION,
        "run_at": now_iso(),
        "model": model.as_dict(),
        "backend": {
            "name": backend.name,
            "binary": str(backend.binary),
            "version": backend_version(backend),
        },
        "env": {"gpu": gpu, "system": sysinfo},
        "config": {
            "ctx": ctx, "k": kv, "v": kv, "reps": reps,
            "gen_tokens": gen_tokens, "ignore_eos": True,
            "prompt_tokens": prompt_n["median"],
        },
        "vram": {
            "baseline_mib": baseline,
            "loaded_mib": loaded_mib,
            "delta_mib": loaded_mib - baseline,
        },
        "layers": layers,
        "buffers_mib": info["buffers_mib"],
        "server": {"ready_s": round(server.ready_s or 0.0, 2),
                   "log": rel_to_root(server.log_path)},
        "warmup": {"short": warmup_short, "prefill": warmup_prefill},
        "runs": runs,
        "summary": {"prefill": pp, "generation": tg},
        "warnings": warnings,
    }
    out = results_dir / model.slug / "speed.json"
    write_json(out, result)
    if layers["offloaded"] is not None:
        layer_txt = f"{layers['offloaded']}/{layers['total']}"
    elif layers["full_gpu"]:
        layer_txt = f"all (ngl {layers['ngl']})"
    else:
        layer_txt = f"unknown (ngl {layers['ngl']})"
    print(f"  VRAM {loaded_mib} MiB (baseline {baseline}, +{loaded_mib - baseline})"
          f"  layers {layer_txt}")
    print(f"  saved {rel_to_root(out)}")
    return result


def run_context(cfg: dict, model: Model, backend: Backend,
                ladder: list[dict]) -> dict:
    results_dir = resolve_path(cfg.get("general", {}).get("results_dir", "results"))
    baseline = gpu_mem_used_mib()
    gpu = gpu_info()
    sysinfo = system_info()
    records: list[dict] = []

    for i, rung in enumerate(ladder, 1):
        ctx = int(rung["ctx"])
        k = str(rung.get("k", "f16"))
        v = str(rung.get("v", k))
        label = ctx_label(ctx, k)
        if model.native_ctx and ctx > model.native_ctx:
            rec = {"ctx": ctx, "k": k, "v": v, "label": label, "ok": None,
                   "skipped": f"exceeds native context ({model.native_ctx})"}
            print(f"  [{i}/{len(ladder)}] {label}: skipped (native {model.native_ctx})")
            records.append(rec)
            continue

        print(f"  [{i}/{len(ladder)}] {label}: loading ...")
        server = Server(cfg, model, backend, ctx, k, v)
        try:
            server.start()
            total = server.sample_vram()
            info = server.log_info()
            rec = {
                "ctx": ctx, "k": k, "v": v, "label": label, "ok": True,
                "ready_s": round(server.ready_s or 0.0, 2),
                "total_mib": total,
                "delta_mib": total - baseline,
                "layers": layer_summary(model, backend, info),
                "buffers_mib": info["buffers_mib"],
                "log": rel_to_root(server.log_path),
            }
            print(f"      OK  {total} MiB total (+{total - baseline})")
        except ServerError as exc:
            rec = {
                "ctx": ctx, "k": k, "v": v, "label": label, "ok": False,
                "error": exc.kind,
                "detail": exc.detail.splitlines()[-1] if exc.detail else "",
                "log": rel_to_root(exc.log_path) if exc.log_path else None,
            }
            print(f"      FAIL {exc.kind}"
                  + (f" - {rec['detail']}" if rec["detail"] else ""))
        finally:
            server.stop()
        records.append(rec)

    result = {
        "schema": 1,
        "tool_version": VERSION,
        "run_at": now_iso(),
        "model": model.as_dict(),
        "backend": {
            "name": backend.name,
            "binary": str(backend.binary),
            "version": backend_version(backend),
        },
        "env": {"gpu": gpu, "system": sysinfo},
        "baseline_mib": baseline,
        "native_ctx": model.native_ctx,
        "results": records,
    }
    out = results_dir / model.slug / "context.json"
    write_json(out, result)
    print(f"  saved {rel_to_root(out)}")
    return result


def run_sanity(cfg: dict, model: Model, backend: Backend, ctx: int) -> dict:
    results_dir = resolve_path(cfg.get("general", {}).get("results_dir", "results"))
    server = Server(cfg, model, backend, ctx, "f16", "f16", extra_flags=["--jinja"])
    server.start()
    try:
        payload = {
            "messages": [{"role": "user", "content": SANITY_PROMPT}],
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.05,
            "max_tokens": 300,
            "stream": False,
        }
        resp = chat_completion(server.port, payload)
        msg = ((resp.get("choices") or [{}])[0].get("message") or {})
        content = (msg.get("content") or "").strip()
        reasoning = (msg.get("reasoning_content") or "").strip()
        timings = normalize_response(resp)
    finally:
        server.stop()

    ok = bool(content or reasoning)
    result = {
        "schema": 1,
        "tool_version": VERSION,
        "run_at": now_iso(),
        "model": model.as_dict(),
        "backend": {
            "name": backend.name,
            "binary": str(backend.binary),
            "version": backend_version(backend),
        },
        "ctx": ctx,
        "ok": ok,
        "reasoning": reasoning,
        "content": content,
        "timings": timings,
    }
    out = results_dir / model.slug / "sanity.json"
    write_json(out, result)
    if reasoning:
        print("  reasoning: " + reasoning[:300].replace("\n", " "))
    print("  answer: " + (content[:300].replace("\n", " ") or "(empty)"))
    print(f"  {'OK' if ok else 'EMPTY'} - {timings['predicted_n']} tok @ "
          f"{fmt_num(timings['predicted_per_second'])} t/s")
    print(f"  saved {rel_to_root(out)}")
    return result


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #

def cmd_list(args, cfg: dict) -> None:
    models = discover_models(cfg, include_mmproj=args.mmproj)
    if not models:
        raise SystemExit("no GGUF models found; check model_roots in config.toml")
    gpu = gpu_info()
    baseline = gpu_mem_used_mib()
    overhead = float(cfg.get("general", {}).get("fit_overhead_mib", 900))
    rows = []
    for m in models:
        fits, need, avail = fit_estimate(m, gpu, baseline, overhead)
        rows.append({
            "model": m,
            "backend": select_backend_name(cfg, m),
            "fits": fits,
            "need_mib": need,
            "avail_mib": avail,
        })

    if args.json:
        data = []
        for row in rows:
            d = row["model"].as_dict()
            d.update({"backend": row["backend"], "fits": row["fits"],
                      "est_need_mib": round(row["need_mib"])})
            data.append(d)
        print(json.dumps(data, indent=2))
        return

    print(f"GPU: {gpu['name']}  {gpu['memory_total_mib']:,} MiB  "
          f"baseline {baseline:,} MiB  free {max(0, gpu['memory_total_mib'] - baseline):,} MiB")
    if baseline > 2000:
        print("warning: high baseline - a model may already be loaded; numbers will be skewed")
    print()
    print(f"{'SLUG':<40} {'SIZE':>8} {'CTX':>7} {'QUANT':>9} "
          f"{'BACKEND':<10} {'FITS':<20} FILE")
    for row in rows:
        m = row["model"]
        ctx = (f"{m.native_ctx // 1024}k" if m.native_ctx and m.native_ctx % 1024 == 0
               else (str(m.native_ctx) if m.native_ctx else "?"))
        fit = f"yes ~{row['need_mib']:.0f} MiB" if row["fits"] else \
              f"NO (needs {row['need_mib']:.0f})"
        print(f"{m.slug:<40} {m.size_gib:>7.1f}G {ctx:>7} {m.quant:>9} "
              f"{row['backend']:<10} {fit:<20} {m.path.name}")
    print(f"\n{len(models)} model(s). Sizes are weight files; FIT includes "
          f"~{overhead:.0f} MiB runtime overhead and current baseline.")


def cmd_bench(args, cfg: dict) -> None:
    model = find_model(cfg, args.model)
    backend = resolve_backend(cfg, select_backend_name(cfg, model, args.backend))
    speed = cfg.get("speed", {})
    ctx = args.ctx if args.ctx is not None else int(speed.get("ctx", 8192))
    kv = args.kv or str(speed.get("kv", "f16"))
    reps = args.reps if args.reps is not None else int(speed.get("reps", 3))
    gen = args.gen if args.gen is not None else int(speed.get("gen_tokens", 128))
    print(f"=== {model.name} ===")
    run_bench(cfg, model, backend, ctx, kv, reps, gen)


def cmd_context(args, cfg: dict) -> None:
    model = find_model(cfg, args.model)
    backend = resolve_backend(cfg, select_backend_name(cfg, model, args.backend))
    ladder = cfg.get("context", {}).get("ladder", [])
    if not ladder:
        raise SystemExit("no context ladder configured")
    print(f"=== {model.name}: context ladder ===")
    run_context(cfg, model, backend, ladder)


def cmd_sanity(args, cfg: dict) -> None:
    model = find_model(cfg, args.model)
    backend = resolve_backend(cfg, select_backend_name(cfg, model, args.backend))
    ctx = args.ctx if args.ctx is not None else int(cfg.get("speed", {}).get("ctx", 8192))
    print(f"=== {model.name}: chat sanity ===")
    run_sanity(cfg, model, backend, ctx)


def cmd_suite(args, cfg: dict) -> None:
    if args.models:
        models = [find_model(cfg, q) for q in args.models]
    elif args.all:
        models = discover_models(cfg)
    else:
        raise SystemExit("suite: pass one or more model names, or --all")

    gpu = gpu_info()
    baseline = gpu_mem_used_mib()
    overhead = float(cfg.get("general", {}).get("fit_overhead_mib", 900))
    speed = cfg.get("speed", {})
    ladder = cfg.get("context", {}).get("ladder", [])
    results_dir = resolve_path(cfg.get("general", {}).get("results_dir", "results"))

    for model in models:
        print(f"\n=== {model.name} ===")
        if args.all and not args.include_oversized:
            fits, need, avail = fit_estimate(model, gpu, baseline, overhead)
            if not fits:
                print(f"  skip: needs ~{need:.0f} MiB, only ~{avail:.0f} MiB free "
                      f"(use --include-oversized to try anyway)")
                continue
        backend = resolve_backend(cfg, select_backend_name(cfg, model, args.backend))
        outdir = results_dir / model.slug
        if args.bench:
            if args.force or not (outdir / "speed.json").exists():
                run_bench(cfg, model, backend,
                          int(speed.get("ctx", 8192)), str(speed.get("kv", "f16")),
                          int(speed.get("reps", 3)), int(speed.get("gen_tokens", 128)))
            else:
                print("  speed: already present (use --force to redo)")
        if args.context:
            if args.force or not (outdir / "context.json").exists():
                run_context(cfg, model, backend, ladder)
            else:
                print("  context: already present (use --force to redo)")
    print("\ndone. run `llmbench report` to regenerate the Markdown report.")


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #

def fmt_stat(st: dict | None) -> str:
    if not st or st.get("median") is None:
        return "-"
    if st.get("n", 0) > 1 and st.get("min") is not None and st.get("max") is not None:
        return f"{st['median']:,.1f} ({st['min']:,.1f}-{st['max']:,.1f})"
    return f"{st['median']:,.1f}"


def strip_leading_heading(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines.pop(0)
    return "\n".join(lines)


def collect_entries(cfg: dict) -> list[dict]:
    results_dir = resolve_path(cfg.get("general", {}).get("results_dir", "results"))
    notes_dir = resolve_path(cfg.get("general", {}).get("notes_dir", "notes"))
    entries = []
    if not results_dir.exists():
        return entries
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir() or d.name == "logs":
            continue
        speed = read_json(d / "speed.json")
        context = read_json(d / "context.json")
        sanity = read_json(d / "sanity.json")
        if not (speed or context):
            continue
        model = (speed or context).get("model", {})
        note_path = notes_dir / f"{d.name}.md"
        notes = note_path.read_text().strip() if note_path.exists() else ""
        entries.append({"slug": d.name, "model": model, "speed": speed,
                        "context": context, "sanity": sanity, "notes": notes})
    return entries


def _kv_rank(k: str) -> int:
    return {"f16": 3, "q8_0": 2, "q4_0": 1}.get(k, 0)


def build_report(entries: list[dict], cfg: dict) -> str:
    lines: list[str] = []
    add = lines.append
    speed_entries = [e for e in entries if e.get("speed")]
    ctx_entries = [e for e in entries if e.get("context")]
    first = (speed_entries or ctx_entries)[0]
    payload = first.get("speed") or first.get("context") or {}
    env = payload.get("env", {})
    gpu = env.get("gpu", {})
    sysinfo = env.get("system", {})
    now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")

    add("# Local LLM benchmark report")
    add("")
    add(f"Generated **{now}** on `{sysinfo.get('host', '?')}` - "
        f"{sysinfo.get('os', '?')}, kernel `{sysinfo.get('kernel', '?')}`.")
    add("")
    add("## Environment")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| GPU | {gpu.get('name', '?')} ({gpu.get('memory_total_mib', 0):,} MiB) |")
    add(f"| Driver | {gpu.get('driver', '?')} |")
    add(f"| CPU | {sysinfo.get('cpu', '?')} |")
    add(f"| RAM | {sysinfo.get('ram_gib', '?')} GiB |")
    versions = []
    for e in entries:
        b = (e.get("speed") or e.get("context") or {}).get("backend")
        if b:
            s = f"{b.get('name')} - {b.get('version')}"
            if s not in versions:
                versions.append(s)
    if versions:
        add(f"| Backend | {'; '.join(versions)} |")
    add("")

    if speed_entries:
        add("## At a glance")
        add("")
        best_tg = max(speed_entries,
                      key=lambda e: (e["speed"]["summary"]["generation"]["median"] or 0))
        best_pp = max(speed_entries,
                      key=lambda e: (e["speed"]["summary"]["prefill"]["median"] or 0))
        add(f"- Fastest generation: **{best_tg['model'].get('name')}** - "
            f"{fmt_num(best_tg['speed']['summary']['generation']['median'])} t/s")
        add(f"- Fastest prefill: **{best_pp['model'].get('name')}** - "
            f"{fmt_num(best_pp['speed']['summary']['prefill']['median'])} t/s")
        best_ctx = None
        for e in ctx_entries:
            for r in e["context"].get("results", []):
                if r.get("ok") is True:
                    key = (r["ctx"], _kv_rank(r["k"]))
                    if best_ctx is None or key > best_ctx[0]:
                        best_ctx = (key, e, r)
        if best_ctx:
            _, e, r = best_ctx
            add(f"- Largest fully-GPU context: **{e['model'].get('name')}** - "
                f"{r['label']} at {fmt_gib(r['total_mib'])} total")
        add(f"- Models in report: {len(entries)} "
            f"({len(speed_entries)} speed, {len(ctx_entries)} context)")
        add("")

    if speed_entries:
        add("## Speed")
        add("")
        add("Sorted by generation speed. Values are medians; min-max across "
            "repetitions in parentheses. VRAM is total used after load, desktop included.")
        add("")
        add("| # | Model | Quant | Size | VRAM | Layers | Prefill t/s | Generation t/s | Backend |")
        add("|---:|---|---:|---:|---:|---:|---:|---:|---|")
        ordered = sorted(speed_entries,
                         key=lambda e: (e["speed"]["summary"]["generation"]["median"] or 0),
                         reverse=True)
        for i, e in enumerate(ordered, 1):
            s = e["speed"]
            m = s.get("model", {})
            layers = s.get("layers") or {}
            off, tot = layers.get("offloaded"), layers.get("total")
            if off is not None:
                lay = f"{off}/{tot}"
            elif layers.get("full_gpu"):
                lay = f"all (ngl {layers.get('ngl', '?')})"
            else:
                lay = "?"
            size = m.get("size_gib")
            size_s = f"{size:.1f} GB" if isinstance(size, (int, float)) else "-"
            add(f"| {i} | {m.get('name', e['slug'])} | {m.get('quant', '?')} | "
                f"{size_s} | {fmt_gib(s.get('vram', {}).get('loaded_mib'))} | {lay} | "
                f"{fmt_stat(s['summary']['prefill'])} | "
                f"{fmt_stat(s['summary']['generation'])} | {s['backend']['name']} |")
        add("")

    if ctx_entries:
        add("## Context / VRAM map")
        add("")
        add("Cell = total GPU memory after a successful load (desktop baseline included), "
            "or failure mode. `-` = skipped (beyond native context), `.` = not tested.")
        add("")
        cols = sorted({(r["ctx"], r["k"]) for e in ctx_entries
                       for r in e["context"].get("results", [])},
                      key=lambda x: (x[0], -_kv_rank(x[1])))
        header = " | ".join(ctx_label(c, k) for c, k in cols)
        add(f"| Model | {header} |")
        add("|---|" + "---:|" * len(cols))
        for e in ctx_entries:
            recs = {(r["ctx"], r["k"]): r for r in e["context"].get("results", [])}
            cells = []
            for c in cols:
                r = recs.get(c)
                if r is None:
                    cells.append(".")
                elif r.get("ok") is True:
                    cells.append(f"✓ {r['total_mib'] / 1024:.1f}G")
                elif r.get("ok") is None:
                    cells.append("-")
                else:
                    cells.append(f"✗ {r.get('error', 'fail')}")
            add(f"| {e['model'].get('name', e['slug'])} | " + " | ".join(cells) + " |")
        add("")

    fails = [(e, r) for e in ctx_entries
             for r in e["context"].get("results", []) if r.get("ok") is False]
    if fails:
        add("## Context load failures")
        add("")
        add("| Model | Rung | Error | Log |")
        add("|---|---|---|---|")
        for e, r in fails:
            log = r.get("log") or ""
            add(f"| {e['model'].get('name', e['slug'])} | {r['label']} | "
                f"{r.get('error', 'fail')} | `{log}` |")
        add("")

    warned = [(e, w) for e in speed_entries for w in e["speed"].get("warnings", [])]
    if warned:
        add("## Warnings")
        add("")
        for e, w in warned:
            add(f"- **{e['model'].get('name', e['slug'])}**: {w}")
        add("")

    noted = [e for e in entries if e.get("notes")]
    if noted:
        add("## Model notes")
        add("")
        for e in noted:
            add(f"### {e['model'].get('name', e['slug'])}")
            add("")
            add(f"`{e['model'].get('path', '')}`")
            add("")
            add(strip_leading_heading(e["notes"]))
            add("")

    add("## Methodology")
    add("")
    reps = 3
    prompt_tokens = 1201
    gen_tokens = 128
    if speed_entries:
        c = speed_entries[0]["speed"].get("config", {})
        reps = c.get("reps", reps)
        prompt_tokens = c.get("prompt_tokens") or prompt_tokens
        gen_tokens = c.get("gen_tokens", gen_tokens)
    add("- Backend: `llama-server`, one slot, `-ngl 99`, `-fa on`, "
        "`-b 2048 -ub 512`, `--parallel 1`, no speculative decoding.")
    add(f"- Speed: warm-up requests discarded, then {reps} repetitions with "
        f"~{int(prompt_tokens)} prompt tokens and exactly {gen_tokens} generated "
        "tokens (`ignore_eos`), temperature 0, `cache_prompt: false`. Timings are "
        "server-reported; median with min-max.")
    add("- VRAM: total used reported by `nvidia-smi` after load, including the "
        "desktop baseline.")
    add("- Context: successful server startup at each ladder rung; KV cache type as "
        "labeled. Failures are load failures, usually OOM.")
    add("- Fully-GPU only: models that require CPU offload are out of scope. "
        "Expect roughly +/-5% run-to-run variation.")
    add("")
    add("## Raw data")
    add("")
    add("- `results/<slug>/speed.json`, `results/<slug>/context.json`, "
        "`results/<slug>/sanity.json`")
    add("- Server logs: `results/logs/` (gitignored)")
    add("")

    return "\n".join(lines)


def cmd_report(args, cfg: dict) -> None:
    entries = collect_entries(cfg)
    if not entries:
        raise SystemExit("no results yet; run `llmbench bench <model>` or `llmbench suite --all`")
    report = build_report(entries, cfg)
    results_dir = resolve_path(cfg.get("general", {}).get("results_dir", "results"))
    out = resolve_path(args.out) if args.out else results_dir / "report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(f"wrote {rel_to_root(out)} ({len(entries)} model(s))")
    if args.stdout:
        print()
        print(report)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="llmbench",
        description="Benchmark local GGUF models: speed, VRAM, context.",
    )
    parser.add_argument("--version", action="version", version=f"llmbench {VERSION}")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", help="list discovered GGUF models")
    p.add_argument("--json", action="store_true")
    p.add_argument("--mmproj", action="store_true", help="include mmproj files")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("bench", help="speed + VRAM at one context size")
    p.add_argument("model")
    p.add_argument("--backend")
    p.add_argument("--ctx", type=int)
    p.add_argument("--kv", help="KV cache type, e.g. f16, q8_0, q4_0")
    p.add_argument("--reps", type=int)
    p.add_argument("--gen", type=int, help="generated tokens per rep")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("context", help="context/VRAM ladder")
    p.add_argument("model")
    p.add_argument("--backend")
    p.set_defaults(func=cmd_context)

    p = sub.add_parser("sanity", help="chat smoke test")
    p.add_argument("model")
    p.add_argument("--backend")
    p.add_argument("--ctx", type=int)
    p.set_defaults(func=cmd_sanity)

    p = sub.add_parser("suite", help="bench + context for several models")
    p.add_argument("models", nargs="*")
    p.add_argument("--all", action="store_true")
    p.add_argument("--backend")
    p.add_argument("--bench", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--context", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--force", action="store_true")
    p.add_argument("--include-oversized", action="store_true")
    p.set_defaults(func=cmd_suite)

    p = sub.add_parser("report", help="build the Markdown report")
    p.add_argument("--out")
    p.add_argument("--stdout", action="store_true")
    p.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    cfg = load_config(Path(args.config))
    try:
        args.func(args, cfg)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except ServerError as exc:
        print(f"error: {exc} ({exc.kind})", file=sys.stderr)
        if exc.log_path:
            print(f"log: {rel_to_root(exc.log_path)}", file=sys.stderr)
        if exc.detail:
            print(exc.detail, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())


