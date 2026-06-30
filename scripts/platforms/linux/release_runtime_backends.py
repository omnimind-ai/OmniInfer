#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
import stat
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimePackage:
    id: str
    runtime_dir_name: str
    source_dir: str
    copy_mode: str
    launcher_name: str | None
    runtime_mode: str
    priority: int


@dataclass(frozen=True)
class BackendTemplate:
    id: str
    family: str
    runtime_dir_name: str
    fallback_runtime_dir_names: tuple[str, ...]
    launcher_name: str | None
    runtime_mode: str
    python_modules: tuple[str, ...] = ()


BACKEND_PRIORITY = {
    "llama.cpp-linux-cuda": 0,
    "llama.cpp-linux-rocm": 0,
    "llama.cpp-linux-vulkan": 0,
    "omniinfer-native-linux": 0,
    "llama.cpp-linux-openvino": 0,
    "ik_llama.cpp-linux-cuda": 0,
    "llama.cpp-linux": 1,
    "llama.cpp-linux-s390x": 1,
    "ik_llama.cpp-linux": 1,
    "vllm-linux-cuda": 2,
}

LINUX_TEMPLATES = (
    BackendTemplate("llama.cpp-linux", "llama.cpp", "llama.cpp-linux", (), "llama-server", "external_server"),
    BackendTemplate("llama.cpp-linux-cuda", "llama.cpp", "llama.cpp-linux-cuda", (), "llama-server", "external_server"),
    BackendTemplate(
        "llama.cpp-linux-rocm",
        "llama.cpp",
        "llama.cpp-linux-rocm",
        ("llama.cpp-linux-ROCm",),
        "llama-server",
        "external_server",
    ),
    BackendTemplate("llama.cpp-linux-vulkan", "llama.cpp", "llama.cpp-linux-vulkan", (), "llama-server", "external_server"),
    BackendTemplate("llama.cpp-linux-s390x", "llama.cpp", "llama.cpp-linux-s390x", (), "llama-server", "external_server"),
    BackendTemplate("omniinfer-native-linux", "llama.cpp", "omniinfer-native-linux", (), "llama-server", "external_server"),
    BackendTemplate("llama.cpp-linux-openvino", "llama.cpp", "llama.cpp-linux-openvino", (), "llama-server", "external_server"),
    BackendTemplate("ik_llama.cpp-linux", "llama.cpp", "ik_llama.cpp-linux", (), "llama-server", "external_server"),
    BackendTemplate("ik_llama.cpp-linux-cuda", "llama.cpp", "ik_llama.cpp-linux-cuda", (), "llama-server", "external_server"),
    BackendTemplate("mnn-linux", "mnn", "mnn-linux", (), None, "embedded", ("MNN", "MNN.llm", "MNN.cv")),
    BackendTemplate("vllm-linux-cuda", "vllm", "vllm-linux-cuda", (), "vllm", "external_server"),
)


def _candidate_runtime_dirs(runtime_root: Path, template: BackendTemplate) -> list[Path]:
    names = (template.runtime_dir_name, *template.fallback_runtime_dir_names)
    return [runtime_root / name for name in names]


def _is_file(path: Path) -> bool:
    return path.is_file() or path.is_symlink()


def _has_embedded_python_runtime(runtime_dir: Path) -> bool:
    return any(
        path.is_file() and path.stat().st_mode & stat.S_IXUSR
        for path in (
            runtime_dir / "bin" / "python3",
            runtime_dir / "venv" / "bin" / "python3",
        )
    )


def _copy_mode(template: BackendTemplate) -> str:
    if template.family == "llama.cpp" and template.launcher_name == "llama-server":
        return "binary-bin"
    return "full-runtime"


def discover_runtime_packages(runtime_root: Path) -> list[RuntimePackage]:
    packages: list[RuntimePackage] = []
    for template in LINUX_TEMPLATES:
        for runtime_dir in _candidate_runtime_dirs(runtime_root, template):
            if not runtime_dir.is_dir():
                continue
            launcher_ready = (
                template.launcher_name is not None
                and _is_file(runtime_dir / "bin" / template.launcher_name)
            )
            embedded_ready = (
                template.runtime_mode == "embedded"
                and template.python_modules
                and _has_embedded_python_runtime(runtime_dir)
            )
            if not launcher_ready and not embedded_ready:
                continue
            packages.append(
                RuntimePackage(
                    id=template.id,
                    runtime_dir_name=runtime_dir.name,
                    source_dir=str(runtime_dir),
                    copy_mode=_copy_mode(template),
                    launcher_name=template.launcher_name,
                    runtime_mode=template.runtime_mode,
                    priority=BACKEND_PRIORITY.get(template.id, 99),
                )
            )
            break
    return packages


def _copy_binary_bin_runtime(source_dir: Path, target_dir: Path) -> None:
    bin_src = source_dir / "bin"
    bin_dst = target_dir / "bin"
    bin_dst.mkdir(parents=True, exist_ok=True)
    (target_dir / "logs").mkdir(parents=True, exist_ok=True)
    if not bin_src.is_dir():
        return
    for src in bin_src.iterdir():
        if not src.is_file():
            continue
        is_shared_library = fnmatch.fnmatch(src.name, "*.so") or fnmatch.fnmatch(src.name, "*.so.*")
        if is_shared_library or (src.stat().st_mode & stat.S_IXUSR):
            shutil.copy2(src, bin_dst / src.name)


def _ignore_full_runtime(source_root: Path):
    def ignore(src: str, names: list[str]) -> set[str]:
        ignored = {name for name in names if name == "__pycache__" or name.endswith(".pyc")}
        if Path(src).resolve() == source_root.resolve():
            ignored.update({"build", "logs", "models", ".cache"})
        return ignored

    return ignore


def _replace_in_text_file(path: Path, old: str, new: str) -> None:
    try:
        data = path.read_bytes()
    except OSError:
        return
    if len(data) > 1024 * 1024 or old.encode() not in data:
        return
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return
    path.write_text(text.replace(old, new), encoding="utf-8")


def _rewrite_runtime_path_references(source_dir: Path, target_dir: Path) -> None:
    old = str(source_dir)
    new = str(target_dir)
    candidates: list[Path] = []
    for relative in ("bin", "venv/bin"):
        base = target_dir / relative
        if base.is_dir():
            candidates.extend(path for path in base.iterdir() if path.is_file())
    if (target_dir / "pyvenv.cfg").is_file():
        candidates.append(target_dir / "pyvenv.cfg")
    if (target_dir / "venv" / "pyvenv.cfg").is_file():
        candidates.append(target_dir / "venv" / "pyvenv.cfg")
    for path in candidates:
        _replace_in_text_file(path, old, new)


def copy_runtime_package(package: RuntimePackage, target_root: Path) -> None:
    source_dir = Path(package.source_dir)
    target_dir = target_root / package.id
    if target_dir.exists():
        shutil.rmtree(target_dir)
    if package.copy_mode == "binary-bin":
        _copy_binary_bin_runtime(source_dir, target_dir)
    elif package.copy_mode == "full-runtime":
        shutil.copytree(source_dir, target_dir, ignore=_ignore_full_runtime(source_dir))
        (target_dir / "logs").mkdir(parents=True, exist_ok=True)
        _rewrite_runtime_path_references(source_dir, target_dir)
    else:
        raise ValueError(f"unsupported copy mode: {package.copy_mode}")


def _default_backend(packages: list[RuntimePackage]) -> str:
    preferred = (
        "llama.cpp-linux",
        "llama.cpp-linux-vulkan",
        "llama.cpp-linux-openvino",
        "llama.cpp-linux-rocm",
        "llama.cpp-linux-s390x",
    )
    ids = {package.id for package in packages}
    for backend_id in preferred:
        if backend_id in ids:
            return backend_id
    return sorted(packages, key=lambda item: (item.priority, item.id))[0].id


def _load_packages(path: Path) -> list[RuntimePackage]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("runtime package manifest must be a list")
    return [RuntimePackage(**item) for item in data]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Discover and copy Linux release runtime backends.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser("discover")
    discover.add_argument("--runtime-root", required=True)
    discover.add_argument("--json", action="store_true")

    copy = subparsers.add_parser("copy")
    copy.add_argument("--manifest", required=True)
    copy.add_argument("--target-root", required=True)

    args = parser.parse_args(argv)

    if args.command == "discover":
        packages = discover_runtime_packages(Path(args.runtime_root).resolve())
        payload = [asdict(package) for package in packages]
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            for package in packages:
                print(package.id)
        return 0

    if args.command == "copy":
        packages = _load_packages(Path(args.manifest))
        target_root = Path(args.target_root)
        for package in packages:
            copy_runtime_package(package, target_root)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
