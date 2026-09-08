"""Event-driven memory probe for reproducing Forge startup/generation failures.

The probe launches Forge as a child process and emits a memory snapshot only on
meaningful Forge log messages or when a memory threshold is crossed.  It is a
diagnostic helper; it does not import or modify Forge internals.
"""

from __future__ import annotations

import argparse
import hashlib
import ctypes
import json
import os
import queue
import re
import subprocess
import sys
import threading
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import psutil


GIB = 1024**3
MIB = 1024**2


class MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


FORGE_EVENTS = re.compile(
    r"Comfy-Kitchen|Total VRAM|Pinned Memory|Running on local URL|Uvicorn running on|Loading Model:|"
    r"Using MixedPrecision|Diffusion Model:|Model loaded in|Requested to load|"
    r"loaded completely|loaded partially|Unloaded partially|Moving model\(s\)|"
    r"patches:|Sampling|100%|Traceback|Error|Exception|WEIGHTLIFE|LORA",
    re.IGNORECASE,
)


def system_memory() -> dict[str, float]:
    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    committed = status.ullTotalPageFile - status.ullAvailPageFile
    return {
        "commit_gib": committed / GIB,
        "commit_limit_gib": status.ullTotalPageFile / GIB,
        "commit_pct": 100.0 * committed / status.ullTotalPageFile,
        "ram_used_gib": (status.ullTotalPhys - status.ullAvailPhys) / GIB,
        "ram_avail_gib": status.ullAvailPhys / GIB,
        "pagefile_used_gib": psutil.swap_memory().used / GIB,
    }


def gpu_memory() -> tuple[float | None, float | None]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        used, total = result.stdout.splitlines()[0].split(",")
        return float(used.strip()) / 1024, float(total.strip()) / 1024
    except Exception:
        return None, None


class Probe:
    def __init__(self, process: subprocess.Popen[str], output: Path):
        self.process = process
        self.ps_process = psutil.Process(process.pid)
        self.output = output.open("w", encoding="utf-8", buffering=1)
        self.lock = threading.Lock()
        self.started = time.perf_counter()
        self.last_gpu = (None, None)
        self.last_gpu_at = 0.0

    def process_memory(self) -> dict[str, object]:
        processes = [self.ps_process]
        try:
            processes.extend(self.ps_process.children(recursive=True))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        private = 0
        peak_private = 0
        working_set = 0
        peak_working_set = 0
        page_faults = 0
        pids = []
        for process in {item.pid: item for item in processes}.values():
            try:
                memory = process.memory_info()
            except psutil.NoSuchProcess:
                continue
            pids.append(process.pid)
            private += memory.private
            peak_private += memory.peak_pagefile
            working_set += memory.rss
            peak_working_set += memory.peak_wset
            page_faults += memory.num_page_faults

        return {
            "process_pids": sorted(pids),
            "private_gib": private / GIB,
            "process_peak_private_gib": peak_private / GIB,
            "working_set_gib": working_set / GIB,
            "peak_working_set_gib": peak_working_set / GIB,
            "page_faults": page_faults,
        }

    def close(self) -> None:
        self.output.close()

    def _gpu(self) -> tuple[float | None, float | None]:
        now = time.perf_counter()
        if now - self.last_gpu_at >= 0.5:
            self.last_gpu = gpu_memory()
            self.last_gpu_at = now
        return self.last_gpu

    def snapshot(self, event: str, detail: str = "") -> dict[str, object]:
        with self.lock:
            record: dict[str, object] = {
                "elapsed_s": round(time.perf_counter() - self.started, 3),
                "event": event,
                "detail": detail.strip(),
                "pid": self.process.pid,
            }
            try:
                process_memory = self.process_memory()
                record.update({k: round(v, 3) if isinstance(v, float) else v for k, v in process_memory.items()})
            except (psutil.NoSuchProcess, psutil.AccessDenied, MemoryError) as exc:
                record['process_memory_error'] = type(exc).__name__
            try:
                record.update({k: round(v, 3) for k, v in system_memory().items()})
            except Exception as exc:
                record["system_memory_error"] = repr(exc)
            if "commit_gib" in record and "private_gib" in record:
                record["non_forge_commit_gib"] = round(
                    float(record["commit_gib"]) - float(record["private_gib"]), 3
                )
            gpu_used, gpu_total = self._gpu()
            if gpu_used is not None:
                record["gpu_used_gib"] = round(gpu_used, 3)
                record["gpu_total_gib"] = round(gpu_total, 3)

            line = json.dumps(record, ensure_ascii=False)
            print(f"[MEMEVENT] {line}", flush=True)
            self.output.write(line + "\n")
            return record


def read_output(process: subprocess.Popen[str], output_queue: queue.Queue[str]) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        output_queue.put(line.rstrip())


def send_txt2img(url: str, payload: dict[str, object], result_queue: queue.Queue[object]) -> None:
    request = urllib.request.Request(
        f"{url.rstrip('/')}/sdapi/v1/txt2img",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            body = response.read()
            images = json.loads(body).get('images', [])
            result_queue.put(("ok", response.status, len(body), [hashlib.sha256(x.encode()).hexdigest() for x in images]))
    except urllib.error.HTTPError as exc:
        result_queue.put(('error', repr(exc), exc.read().decode('utf-8', errors='replace')))
    except Exception as exc:
        result_queue.put(("error", repr(exc)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--output", default="event-memory-probe.jsonl")
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--request-delay", type=float, default=1.0)
    parser.add_argument("--no-request", action="store_true")
    parser.add_argument("--lora-sequence", action="store_true", help="Full-size generation followed by offline LoRA switches")
    parser.add_argument("--skip-full-size", action="store_true")
    parser.add_argument("--internal-events", action="store_true")
    parser.add_argument("--sparse-fixtures", action="store_true", help="Use small derived adapters instead of the full diamel adapter")
    parser.add_argument("forge_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def prepare_lora_fixture(source, destination, multiplier='-0.5'):
    from safetensors import safe_open
    from safetensors.torch import save_file

    with safe_open(source, framework='pt', backend='pread') as handle:
        keys = handle.keys()
        prefixes = sorted(k.removesuffix('.lora_up.weight') for k in keys if k.endswith('.lora_up.weight'))[:3]
        tensors = {}
        for prefix in prefixes:
            for suffix in ('.lora_up.weight', '.lora_down.weight', '.alpha'):
                key = prefix + suffix
                if key in keys:
                    tensor = handle.get_tensor(key)
                    tensors[key] = tensor * float(multiplier) if suffix == '.lora_up.weight' else tensor
        save_file(tensors, destination)


def main() -> int:
    args = parse_args()
    forge_args = args.forge_args
    if forge_args[:1] == ["--"]:
        forge_args = forge_args[1:]

    root = Path(__file__).resolve().parents[1]
    python = root / "venv" / "Scripts" / "python.exe"
    fixture_dir = None
    prompts = ['diagnostic test image']
    if args.lora_sequence:
        fixture_dir = tempfile.TemporaryDirectory(prefix='forge-lora-regression-')
        source = root / 'models/Lora/diamel-v2.2-krea2-000012.safetensors'
        subprocess.run([
            str(python), '-c',
            'from scripts.event_memory_probe import prepare_lora_fixture; import sys; prepare_lora_fixture(*sys.argv[1:])',
            str(source), str(Path(fixture_dir.name) / 'forge_regression_bar.safetensors'),
        ], cwd=root, check=True)
        forge_args += ['--lora-dir', str(source.parent), '--lora-dirs', fixture_dir.name]
        foo = 'diagnostic test image <lora:diamel-v2.2-krea2-000012:1>'
        if args.sparse_fixtures:
            subprocess.run([
                str(python), '-c',
                'from scripts.event_memory_probe import prepare_lora_fixture; import sys; prepare_lora_fixture(*sys.argv[1:])',
                str(source), str(Path(fixture_dir.name) / 'forge_regression_foo.safetensors'), '1.0',
            ], cwd=root, check=True)
            foo = 'diagnostic test image <lora:forge_regression_foo:1>'
        prompts = ['diagnostic test image', foo, 'diagnostic test image <lora:forge_regression_bar:1>', 'diagnostic test image', foo]
    entrypoint = 'scripts/diagnostics/weight_lifetime_launch.py' if args.internal_events else 'launch.py'
    command = [str(python), entrypoint, *forge_args]
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    process = subprocess.Popen(
        command,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=creationflags,
    )

    line_queue: queue.Queue[str] = queue.Queue()
    result_queue: queue.Queue[object] = queue.Queue()
    reader = threading.Thread(target=read_output, args=(process, line_queue), daemon=True)
    reader.start()
    probe = Probe(process, root / args.output)
    probe.snapshot("child-start", " ".join(command))

    commit_thresholds = iter((50, 60, 70, 80, 85, 90, 92, 94, 96, 98, 99))
    private_thresholds = iter(range(2, 33, 2))
    next_commit = next(commit_thresholds, None)
    next_private = next(private_thresholds, None)
    server_ready_at: float | None = None
    request_started = False
    request_finished = False
    last_poll = 0.0
    request_index = 0
    peak_private = 0.0
    peak_commit = 0.0

    try:
        while process.poll() is None:
            now = time.perf_counter()
            while True:
                try:
                    line = line_queue.get_nowait()
                except queue.Empty:
                    break
                if FORGE_EVENTS.search(line):
                    probe.snapshot("forge-log", line)
                if "Running on local URL:" in line or "Uvicorn running on" in line:
                    server_ready_at = now

            if now - last_poll >= 0.1:
                last_poll = now
                try:
                    system = system_memory()
                    private_gib = float(probe.process_memory()["private_gib"])
                    peak_private = max(peak_private, private_gib)
                    peak_commit = max(peak_commit, system['commit_gib'])
                except (psutil.NoSuchProcess, psutil.AccessDenied, MemoryError) as exc:
                    probe.snapshot('poll-error', type(exc).__name__)
                    continue

                while next_commit is not None and system["commit_pct"] >= next_commit:
                    probe.snapshot("commit-threshold", f">={next_commit}%")
                    next_commit = next(commit_thresholds, None)
                while next_private is not None and private_gib >= next_private:
                    probe.snapshot("private-threshold", f">={next_private} GiB")
                    next_private = next(private_thresholds, None)

            if (
                server_ready_at is not None
                and not args.no_request
                and not request_started
                and now - server_ready_at >= args.request_delay
            ):
                payload = {
                    "prompt": prompts[request_index],
                    "negative_prompt": "",
                    "steps": 12 if args.lora_sequence and not args.skip_full_size and request_index == 0 else 1,
                    "cfg_scale": 1,
                    "width": 1408 if args.lora_sequence and not args.skip_full_size and request_index == 0 else 512,
                    "height": 1024 if args.lora_sequence and not args.skip_full_size and request_index == 0 else 512,
                    "seed": 12345,
                    "sampler_name": "Euler a",
                    "scheduler": "Krea2",
                    "batch_size": 1,
                    "n_iter": 1,
                    "save_images": False,
                    "send_images": args.lora_sequence,
                    # API encoder supports PNG/JPEG/WebP, not every UI format.
                    # These overrides live only in the disposable child process.
                    "override_settings": {"samples_format": "png"} if args.lora_sequence else {},
                    "override_settings_restore_afterwards": False,
                }
                probe.snapshot("api-request-start", json.dumps(payload))
                sender = threading.Thread(
                    target=send_txt2img,
                    args=(args.url, payload, result_queue),
                    daemon=True,
                )
                sender.start()
                request_started = True

            if request_started and not request_finished:
                try:
                    result = result_queue.get_nowait()
                except queue.Empty:
                    pass
                else:
                    probe.snapshot("api-request-finish", repr(result))
                    request_finished = True
                    if result[0] == "ok":
                        request_index += 1
                        if request_index >= len(prompts):
                            break
                        request_started = False
                        request_finished = False
                    else:
                        break

            if server_ready_at is None and now - probe.started > args.startup_timeout:
                probe.snapshot("startup-timeout")
                break
            time.sleep(0.02)
    finally:
        probe.snapshot('poll-peaks', f'private_gib={peak_private:.3f} commit_gib={peak_commit:.3f}')
        return_code = process.poll()
        probe.snapshot("child-exit", f"returncode={return_code}")
        try:
            descendants = psutil.Process(process.pid).children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            descendants = []
        for descendant in reversed(descendants):
            try:
                descendant.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        _, alive = psutil.wait_procs(descendants, timeout=10)
        for descendant in alive:
            try:
                descendant.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        if return_code is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            probe.snapshot("child-terminated", f"returncode={process.returncode}")
        probe.close()
        if fixture_dir is not None:
            fixture_dir.cleanup()

    return 0 if request_index == len(prompts) else (process.returncode or 1)


if __name__ == "__main__":
    raise SystemExit(main())
