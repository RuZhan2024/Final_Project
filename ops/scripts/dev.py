#!/usr/bin/env python3
"""Cross-platform developer entrypoint for install/start/stop/smoke flows."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = ROOT / "applications" / "frontend"
STATE_DIR = ROOT / ".make"
STATE_PATH = STATE_DIR / "dev-state.json"
LEGACY_WINDOWS_STATE_PATH = STATE_DIR / "dev-windows.json"
IS_WINDOWS = os.name == "nt"
DEFAULT_VENV = ".venv-win" if IS_WINDOWS else ".venv"
DEFAULT_BACKEND_HOST = "127.0.0.1"
DEFAULT_BACKEND_PORT = 8000
DEFAULT_FRONTEND_HOST = "127.0.0.1"
DEFAULT_FRONTEND_PORT = 3000


class DevError(RuntimeError):
    """User-facing failure with a concise message."""


def run(
    cmd: list[str],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    kwargs: dict[str, Any] = {
        "cwd": str(cwd),
        "env": env,
        "text": True,
    }
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    proc = subprocess.run(cmd, **kwargs)
    if check and proc.returncode != 0:
        quoted = " ".join(cmd)
        detail = ""
        if capture:
            detail = f"\nstdout:\n{proc.stdout or ''}\nstderr:\n{proc.stderr or ''}"
        raise DevError(f"command failed ({proc.returncode}): {quoted}{detail}")
    return proc


def command_path(name: str) -> str | None:
    if IS_WINDOWS and not name.lower().endswith(".cmd"):
        return shutil.which(f"{name}.cmd") or shutil.which(name)
    return shutil.which(name)


def command_version(cmd: list[str]) -> str | None:
    try:
        proc = run(cmd, check=False, capture=True)
    except OSError:
        return None
    text = (proc.stdout or proc.stderr or "").strip()
    if proc.returncode != 0 and not text:
        return None
    return text.splitlines()[0] if text else "available"


def venv_dir(args: argparse.Namespace | None = None) -> Path:
    override = getattr(args, "venv_dir", None) if args is not None else None
    name = override or os.environ.get("FD_VENV_DIR") or DEFAULT_VENV
    return ROOT / name


def venv_python(args: argparse.Namespace | None = None) -> Path:
    base = venv_dir(args)
    return base / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def local_node_bin() -> Path | None:
    if IS_WINDOWS:
        node_dir = ROOT / ".tools" / "node-v22-win-x64"
        return node_dir if (node_dir / "npm.cmd").exists() else None
    return None


def npm_command() -> str:
    node_bin = local_node_bin()
    if node_bin is not None:
        npm = node_bin / "npm.cmd"
        if npm.exists():
            return str(npm)
    npm = command_path("npm")
    if not npm:
        raise DevError("missing npm. Install Node.js 22.x, then rerun bootstrap.")
    return npm


def node_command() -> str | None:
    node_bin = local_node_bin()
    if node_bin is not None:
        node = node_bin / ("node.exe" if IS_WINDOWS else "node")
        if node.exists():
            return str(node)
    return command_path("node")


def env_with_local_node() -> dict[str, str]:
    env = os.environ.copy()
    node_bin = local_node_bin()
    if node_bin is not None:
        env["PATH"] = f"{node_bin}{os.pathsep}{env.get('PATH', '')}"
    return env


def ensure_python_compatible(python: str) -> None:
    code = "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
    proc = run([python, "-c", code], check=False, capture=True)
    if proc.returncode != 0:
        raise DevError(f"{python} must be Python 3.10 or newer.")


def resolve_bootstrap_python(requested: str | None = None) -> list[str]:
    candidates: list[list[str]] = []
    if requested:
        candidates.append([requested])
    if os.environ.get("PY_BIN"):
        candidates.append([os.environ["PY_BIN"]])
    candidates.append([sys.executable])
    if IS_WINDOWS:
        candidates.extend([["py", "-3.10"], ["py", "-3"], ["python"], ["python3"]])
    else:
        candidates.extend([["python3.10"], ["python3"], ["python"]])

    for parts in candidates:
        exe = shutil.which(parts[0]) if len(parts) == 1 else shutil.which(parts[0])
        if not exe:
            continue
        cmd = [exe, *parts[1:]]
        proc = run(
            cmd + ["-c", "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"],
            check=False,
            capture=True,
        )
        if proc.returncode == 0:
            return cmd
    raise DevError("could not find Python 3.10+. Install Python 3.10/3.11 and rerun bootstrap.")


def port_is_free(host: str, port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def http_ok(url: str, timeout_s: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            return 200 <= int(resp.status) < 400
    except (OSError, urllib.error.URLError):
        return False


def http_json(url: str, timeout_s: float = 5.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_for_url(url: str, attempts: int) -> bool:
    for _ in range(attempts):
        if http_ok(url):
            return True
        time.sleep(1)
    return False


def windows_listening_pids(port: int) -> list[int]:
    ps = (
        f"Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue | "
        "Select-Object -ExpandProperty OwningProcess -Unique"
    )
    proc = run(["powershell", "-NoProfile", "-Command", ps], check=False, capture=True)
    pids: list[int] = []
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids


def unix_listening_pids(port: int) -> list[int]:
    lsof = shutil.which("lsof")
    if not lsof:
        return []
    proc = run([lsof, f"-tiTCP:{port}", "-sTCP:LISTEN"], check=False, capture=True)
    pids: list[int] = []
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids


def listening_pids(port: int) -> list[int]:
    return windows_listening_pids(port) if IS_WINDOWS else unix_listening_pids(port)


def windows_command_line(pid: int) -> str:
    ps = f"(Get-CimInstance Win32_Process -Filter \"ProcessId={pid}\" -ErrorAction SilentlyContinue).CommandLine"
    proc = run(["powershell", "-NoProfile", "-Command", ps], check=False, capture=True)
    return (proc.stdout or "").strip()


def unix_command_line(pid: int) -> str:
    proc = run(["ps", "-p", str(pid), "-o", "command="], check=False, capture=True)
    return (proc.stdout or "").strip()


def process_command_line(pid: int) -> str:
    return windows_command_line(pid) if IS_WINDOWS else unix_command_line(pid)


def is_project_process(pid: int) -> bool:
    cmd = process_command_line(pid)
    if not cmd:
        return False
    root_text = str(ROOT)
    markers = [
        root_text,
        "applications.backend.app:app",
        "react-scripts",
        "fall_detection_frontend",
        "fall_detection_backend",
    ]
    cmd_lower = cmd.lower()
    return any(marker.lower() in cmd_lower for marker in markers)


def stop_pid(pid: int) -> bool:
    if pid <= 0 or not is_project_process(pid):
        return False
    try:
        if IS_WINDOWS:
            run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False, capture=True)
        else:
            os.kill(pid, signal.SIGTERM)
        return True
    except OSError:
        return False


def read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_state(state: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def runtime_env(args: argparse.Namespace | None = None) -> dict[str, str]:
    env = env_with_local_node()
    env["PYTHONPATH"] = f"{ROOT / 'ml' / 'src'}{os.pathsep}{ROOT}"
    env.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
    if args is not None:
        env["REACT_APP_API_BASE"] = f"http://{args.backend_host}:{args.backend_port}"
    return env


def print_doctor_report(report: dict[str, Any]) -> None:
    print("[doctor] platform:", report["platform"])
    print("[doctor] python:", report["python"])
    print("[doctor] venv:", report["venv"])
    print("[doctor] node:", report["node"])
    print("[doctor] npm:", report["npm"])
    print("[doctor] git:", report["git"])
    print("[doctor] docker:", report["docker"])
    print("[doctor] runtime assets:")
    for key, value in report["runtime_assets"].items():
        print(f"  - {key}: {value}")
    print("[doctor] raw data:")
    for key, value in report["raw_data"].items():
        print(f"  - {key}: {value}")
    print("[doctor] ports:")
    for key, value in report["ports"].items():
        print(f"  - {key}: {value}")


def build_doctor_report(args: argparse.Namespace) -> dict[str, Any]:
    node = node_command()
    try:
        npm = npm_command()
    except DevError:
        npm = None
    replay_root = ROOT / "ops" / "deploy_assets" / "replay_clips"
    replay_count = len(list(replay_root.rglob("*.mp4"))) if replay_root.exists() else 0
    raw_root = ROOT / "data" / "raw"
    raw_data = {name: (raw_root / name).exists() for name in ("caucafall", "le2i", "urfall", "muvim")}
    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "python": {
            "current": sys.version.split()[0],
            "executable": sys.executable,
            "compatible": sys.version_info >= (3, 10),
        },
        "venv": {
            "path": str(venv_dir(args)),
            "python": str(venv_python(args)),
            "exists": venv_python(args).exists(),
        },
        "node": command_version([node, "--version"]) if node else None,
        "npm": command_version([npm, "--version"]) if npm else None,
        "git": command_version([command_path("git") or "git", "--version"]),
        "docker": command_version([command_path("docker") or "docker", "--version"]),
        "runtime_assets": {
            "manifest": (ROOT / "ops" / "deploy_assets" / "manifest.json").exists(),
            "checkpoint": (ROOT / "ops" / "deploy_assets" / "checkpoints" / "caucafall_tcn_best.pt").exists(),
            "ops_profile": (ROOT / "ops" / "configs" / "ops" / "tcn_caucafall.yaml").exists(),
            "replay_clips": replay_count,
        },
        "raw_data": raw_data,
        "ports": {
            "backend_8000_free": port_is_free(DEFAULT_BACKEND_HOST, DEFAULT_BACKEND_PORT),
            "frontend_3000_free": port_is_free(DEFAULT_FRONTEND_HOST, DEFAULT_FRONTEND_PORT),
            "backend_8000_pids": listening_pids(DEFAULT_BACKEND_PORT),
            "frontend_3000_pids": listening_pids(DEFAULT_FRONTEND_PORT),
        },
    }


def cmd_doctor(args: argparse.Namespace) -> int:
    report = build_doctor_report(args)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_doctor_report(report)
    if args.strict:
        required = [
            bool(report["python"]["compatible"]),
            bool(report["node"]),
            bool(report["npm"]),
            bool(report["runtime_assets"]["manifest"]),
            bool(report["runtime_assets"]["checkpoint"]),
            bool(report["runtime_assets"]["ops_profile"]),
            int(report["runtime_assets"]["replay_clips"]) == 24,
        ]
        return 0 if all(required) else 1
    return 0


def cmd_bootstrap(args: argparse.Namespace) -> int:
    py = resolve_bootstrap_python(args.python)
    vpy = venv_python(args)

    if not vpy.exists():
        print(f"[bootstrap] creating {venv_dir(args).name}")
        run(py + ["-m", "venv", str(venv_dir(args))])

    print("[bootstrap] upgrading packaging tools")
    run([str(vpy), "-m", "pip", "install", "--upgrade", "pip", "setuptools<82", "wheel"])

    dep_probe = run(
        [str(vpy), "-c", "import fastapi, uvicorn, yaml, numpy, torch"],
        check=False,
        capture=True,
    )
    if dep_probe.returncode != 0:
        print("[bootstrap] installing backend/runtime dependencies")
        install = run([str(vpy), "-m", "pip", "install", "-r", "requirements.txt"], check=False)
        if install.returncode != 0 and IS_WINDOWS:
            print("[bootstrap] full requirements failed; falling back to requirements_server.txt")
            run([str(vpy), "-m", "pip", "install", "-r", "requirements_server.txt"])
        run([str(vpy), "-m", "pip", "install", "-e", ".", "--no-build-isolation"])

    node = node_command()
    if not node:
        raise DevError("missing Node.js. Install Node.js 22.x, then rerun bootstrap.")
    node_major = command_version([node, "-p", "process.versions.node.split('.')[0]"])
    if node_major and node_major.strip() != "22":
        print(f"[warn] detected Node.js major {node_major}; Node.js 22.x is recommended.")

    node_modules = FRONTEND_DIR / "node_modules"
    if not node_modules.exists():
        npm = npm_command()
        npm_install_cmd = "ci" if (FRONTEND_DIR / "package-lock.json").exists() else "install"
        print(f"[bootstrap] installing frontend dependencies with npm {npm_install_cmd}")
        run([npm, npm_install_cmd], cwd=FRONTEND_DIR, env=env_with_local_node())

    print("[bootstrap] syncing frontend MediaPipe assets")
    run([npm_command(), "run", "sync-mediapipe-assets"], cwd=FRONTEND_DIR, env=env_with_local_node())

    doctor_args = argparse.Namespace(**vars(args))
    doctor_args.json = False
    doctor_args.strict = False
    cmd_doctor(doctor_args)

    if args.skip_start:
        print("[bootstrap] done. Start later with: python ops/scripts/dev.py start")
        return 0
    start_args = argparse.Namespace(**vars(args))
    return cmd_start(start_args)


def ensure_start_requirements(args: argparse.Namespace) -> None:
    if not venv_python(args).exists():
        raise DevError("missing virtual environment. Run bootstrap first.")
    if not (FRONTEND_DIR / "node_modules").exists():
        raise DevError("missing frontend node_modules. Run bootstrap first.")
    if not port_is_free(args.backend_host, args.backend_port):
        raise DevError(f"backend port already in use: {args.backend_host}:{args.backend_port}")
    if not port_is_free(args.frontend_host, args.frontend_port):
        raise DevError(f"frontend port already in use: {args.frontend_host}:{args.frontend_port}")


def popen_kwargs(stdout_path: Path, stderr_path: Path, cwd: Path, env: dict[str, str]) -> dict[str, Any]:
    stdout = stdout_path.open("w", encoding="utf-8")
    stderr = stderr_path.open("w", encoding="utf-8")
    kwargs: dict[str, Any] = {
        "cwd": str(cwd),
        "env": env,
        "stdout": stdout,
        "stderr": stderr,
    }
    if IS_WINDOWS:
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    else:
        kwargs["start_new_session"] = True
    return kwargs


def cmd_start(args: argparse.Namespace) -> int:
    ensure_start_requirements(args)
    temp = Path(tempfile.gettempdir())
    backend_out = temp / "fall_detection_backend.log"
    backend_err = temp / "fall_detection_backend.err.log"
    frontend_out = temp / "fall_detection_frontend.log"
    frontend_err = temp / "fall_detection_frontend.err.log"
    health_url = f"http://{args.backend_host}:{args.backend_port}/api/health"
    frontend_url = f"http://{args.frontend_host}:{args.frontend_port}"
    env = runtime_env(args)

    print(f"[dev] starting backend on {args.backend_host}:{args.backend_port}")
    backend = subprocess.Popen(
        [
            str(venv_python(args)),
            "-m",
            "uvicorn",
            "applications.backend.app:app",
            "--host",
            args.backend_host,
            "--port",
            str(args.backend_port),
        ],
        **popen_kwargs(backend_out, backend_err, ROOT, env),
    )
    frontend: subprocess.Popen[str] | None = None

    try:
        if not wait_for_url(health_url, args.backend_attempts):
            raise DevError(f"backend failed health check: {health_url}. Logs: {backend_out} / {backend_err}")
        print(f"[dev] backend healthy: {health_url}")

        frontend_env = env.copy()
        frontend_env.update(
            {
                "HOST": args.frontend_host,
                "PORT": str(args.frontend_port),
                "BROWSER": args.browser,
                "REACT_APP_API_BASE": f"http://{args.backend_host}:{args.backend_port}",
            }
        )
        print(f"[dev] starting frontend on {args.frontend_host}:{args.frontend_port}")
        frontend = subprocess.Popen(
            [npm_command(), "start"],
            **popen_kwargs(frontend_out, frontend_err, FRONTEND_DIR, frontend_env),
        )

        if args.detached:
            if not wait_for_url(frontend_url, args.frontend_attempts):
                raise DevError(f"frontend failed readiness check: {frontend_url}. Logs: {frontend_out} / {frontend_err}")
            backend_pids = listening_pids(args.backend_port)
            frontend_pids = listening_pids(args.frontend_port)
            state = {
                "backend_pid": backend_pids[0] if backend_pids else backend.pid,
                "frontend_pid": frontend_pids[0] if frontend_pids else frontend.pid,
                "backend_launcher_pid": backend.pid,
                "frontend_launcher_pid": frontend.pid,
                "backend_url": health_url,
                "frontend_url": frontend_url,
                "backend_stdout": str(backend_out),
                "backend_stderr": str(backend_err),
                "frontend_stdout": str(frontend_out),
                "frontend_stderr": str(frontend_err),
            }
            write_state(state)
            print(f"[dev] frontend ready: {frontend_url}")
            print(f"[dev] detached state: {STATE_PATH}")
            return 0

        state = {
            "backend_pid": backend.pid,
            "frontend_pid": frontend.pid,
            "backend_url": health_url,
            "frontend_url": frontend_url,
            "backend_stdout": str(backend_out),
            "backend_stderr": str(backend_err),
            "frontend_stdout": str(frontend_out),
            "frontend_stderr": str(frontend_err),
        }
        write_state(state)
        try:
            return frontend.wait()
        finally:
            if backend.poll() is None:
                backend.terminate()
            if STATE_PATH.exists():
                STATE_PATH.unlink()
    except Exception:
        if frontend is not None and frontend.poll() is None:
            frontend.terminate()
        if backend.poll() is None:
            backend.terminate()
        raise


def cmd_stop(args: argparse.Namespace) -> int:
    stopped: set[int] = set()
    for path in (STATE_PATH, LEGACY_WINDOWS_STATE_PATH):
        state = read_state(path)
        for key in ("backend_pid", "frontend_pid", "backend_launcher_pid", "frontend_launcher_pid"):
            value = state.get(key)
            if isinstance(value, int) and stop_pid(value):
                stopped.add(value)
        if path.exists():
            path.unlink()

    for port in args.ports:
        for pid in listening_pids(int(port)):
            if stop_pid(pid):
                stopped.add(pid)

    if stopped:
        print("[dev] stopped project processes:", ", ".join(str(pid) for pid in sorted(stopped)))
    else:
        print("[dev] no project-owned dev processes found")
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    backend = f"http://{args.backend_host}:{args.backend_port}"
    frontend = f"http://{args.frontend_host}:{args.frontend_port}"
    if not http_ok(f"{backend}/api/health"):
        raise DevError(f"backend health failed: {backend}/api/health")
    settings = http_json(f"{backend}/api/settings")
    specs_response = http_json(f"{backend}/api/deploy/specs")
    clips_response = http_json(f"{backend}/api/replay/clips")
    frontend_ok = http_ok(frontend)
    specs = specs_response.get("specs", [])
    clips = clips_response.get("clips", [])
    keys = [spec.get("spec_key") for spec in specs]
    result = {
        "backend_ok": True,
        "settings_db_available": settings.get("db_available"),
        "active_model": settings.get("active_model_code"),
        "active_dataset": (settings.get("system") or {}).get("active_dataset_code"),
        "active_op": settings.get("active_op_code"),
        "deploy_specs_count": len(specs),
        "deploy_spec_keys": keys,
        "replay_clips_count": len(clips),
        "frontend_index_ok": frontend_ok,
    }
    print(json.dumps(result, indent=2))
    expected = (
        result["active_model"] == "TCN"
        and result["active_dataset"] == "caucafall"
        and result["active_op"] == "OP-2"
        and keys == ["caucafall_tcn"]
        and len(clips) == 24
        and frontend_ok
    )
    if not expected:
        raise DevError("smoke check failed")
    return 0


def add_common_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend-host", default=DEFAULT_BACKEND_HOST)
    parser.add_argument("--backend-port", type=int, default=DEFAULT_BACKEND_PORT)
    parser.add_argument("--frontend-host", default=DEFAULT_FRONTEND_HOST)
    parser.add_argument("--frontend-port", type=int, default=DEFAULT_FRONTEND_PORT)
    parser.add_argument("--venv-dir", default=os.environ.get("FD_VENV_DIR") or DEFAULT_VENV)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Portable dev workflow entrypoint")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Inspect local toolchain, assets, data, and ports")
    add_common_runtime_args(doctor)
    doctor.add_argument("--json", action="store_true")
    doctor.add_argument("--strict", action="store_true")
    doctor.set_defaults(func=cmd_doctor)

    bootstrap = sub.add_parser("bootstrap", help="Install dependencies and optionally start the app")
    add_common_runtime_args(bootstrap)
    bootstrap.add_argument("--python", default=None)
    bootstrap.add_argument("--skip-start", action="store_true")
    bootstrap.add_argument("--detached", action="store_true")
    bootstrap.add_argument("--browser", default="none")
    bootstrap.add_argument("--backend-attempts", type=int, default=30)
    bootstrap.add_argument("--frontend-attempts", type=int, default=60)
    bootstrap.set_defaults(func=cmd_bootstrap)

    start = sub.add_parser("start", help="Start backend and frontend")
    add_common_runtime_args(start)
    start.add_argument("--detached", action="store_true")
    start.add_argument("--browser", default="none")
    start.add_argument("--backend-attempts", type=int, default=30)
    start.add_argument("--frontend-attempts", type=int, default=60)
    start.set_defaults(func=cmd_start)

    stop = sub.add_parser("stop", help="Stop project-owned backend/frontend processes")
    stop.add_argument("--ports", nargs="*", type=int, default=[DEFAULT_BACKEND_PORT, DEFAULT_FRONTEND_PORT])
    stop.set_defaults(func=cmd_stop)

    smoke = sub.add_parser("smoke", help="Validate local backend/frontend runtime")
    add_common_runtime_args(smoke)
    smoke.set_defaults(func=cmd_smoke)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except DevError as exc:
        print(f"[err] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
