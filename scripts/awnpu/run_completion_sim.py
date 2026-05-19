#!/usr/bin/env python3
"""AWNPU CPU 模拟：用 llama-completion 跑通 Qwen2.5-0.5B（或其它 GGUF）。

示例:
  python3 scripts/awnpu/run_completion_sim.py
  python3 scripts/awnpu/run_completion_sim.py -p "1+1等于几？" -n 64
  python3 scripts/awnpu/run_completion_sim.py --list-devices
  MODEL=/mnt/c/Users/wangt/models/qwen2-5/qwen2.5-0.5b-instruct-fp16.gguf python3 scripts/awnpu/run_completion_sim.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_model(root: Path) -> Path:
    return Path("/mnt/c/Users/wangt/models/qwen2-5/qwen2.5-0.5b-instruct-fp16.gguf")
    return root / "models" / "qwen2-5" / "qwen2.5-0.5b-instruct-fp16.gguf"


def build_env(build_dir: Path, awnpu_sim: str) -> dict[str, str]:
    env = os.environ.copy()
    env["AWNPU_LLM_SIM"] = awnpu_sim
    lib_dir = str(build_dir / "bin")
    prev = env.get("LD_LIBRARY_PATH", "")
    if prev:
        if lib_dir not in prev.split(":"):
            env["LD_LIBRARY_PATH"] = f"{lib_dir}:{prev}"
    else:
        env["LD_LIBRARY_PATH"] = lib_dir
    return env


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = repo_root()
    p = argparse.ArgumentParser(description="AWNPU 模拟模式下运行 llama-completion")
    p.add_argument(
        "--root",
        type=Path,
        default=root,
        help=f"仓库根目录（默认: {root}）",
    )
    p.add_argument(
        "--build-dir",
        type=Path,
        default=None,
        help="构建目录（默认: <root>/build）",
    )
    p.add_argument(
        "--bin",
        type=Path,
        default=None,
        help="llama-completion 可执行文件路径",
    )
    p.add_argument(
        "-m",
        "--model",
        type=Path,
        default=None,
        help="GGUF 模型路径",
    )
    p.add_argument(
        "--device",
        default=os.environ.get("DEVICE", "AWNPU0"),
        help="后端设备名（默认: AWNPU0）",
    )
    p.add_argument(
        "--awnpu-sim",
        default=os.environ.get("AWNPU_LLM_SIM", "1"),
        help="AWNPU_LLM_SIM 环境变量（默认: 1）",
    )
    p.add_argument(
        "-p",
        "--prompt",
        default=os.environ.get(
            "PROMPT", "你好，请用一句话介绍你自己。"
        ),
        help="提示词",
    )
    p.add_argument(
        "-n",
        "--n-predict",
        type=int,
        default=int(os.environ.get("N_PREDICT", "128")),
        help="生成长度",
    )
    p.add_argument(
        "-c",
        "--ctx-size",
        type=int,
        default=int(os.environ.get("CTX_SIZE", "8192")),
        help="上下文长度",
    )
    p.add_argument(
        "-ngl",
        "--n-gpu-layers",
        type=int,
        default=int(os.environ.get("NGL", "99")),
        help="卸载到设备的层数",
    )
    p.add_argument(
        "--no-cnv",
        nargs="?",
        const=True,
        default=parse_bool(os.environ.get("NO_CNV", "1")),
        type=parse_bool,
        metavar="BOOL",
        help="纯补全模式，不加对话模板（默认: 开启；可用 --no-cnv=false 或 --no-cnv false 关闭）",
    )
    p.add_argument(
        "--system-prompt",
        default=os.environ.get("SYS_PROMPT", ""),
        help="系统提示（对话模式时使用）",
    )
    p.add_argument(
        "--list-devices",
        action="store_true",
        help="列出可用设备后退出",
    )
    p.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="透传给 llama-completion 的额外参数",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    root = args.root.resolve()
    build_dir = (args.build_dir or root / "build").resolve()
    bin_path = (args.bin or build_dir / "bin" / "llama-completion").resolve()
    model = (args.model or default_model(root)).resolve()

    if not bin_path.is_file():
        print(
            f"error: 未找到 {bin_path}\n"
            f"请先构建: cmake --build {build_dir} --target llama-completion",
            file=sys.stderr,
        )
        return 1

    env = build_env(build_dir, args.awnpu_sim)

    if args.list_devices:
        cmd = [str(bin_path), "--list-devices"]
        print(f"==> AWNPU_LLM_SIM={env['AWNPU_LLM_SIM']}", file=sys.stderr)
        print("==>", " ".join(cmd), file=sys.stderr)
        return subprocess.run(cmd, env=env, cwd=root).returncode

    if not model.is_file():
        print(f"error: 模型不存在: {model}", file=sys.stderr)
        return 1

    cmd: list[str] = [
        str(bin_path),
        "-m",
        str(model),
        "--device",
        args.device,
        "-ngl",
        str(args.n_gpu_layers),
        "-c",
        str(args.ctx_size),
        "-n",
        str(args.n_predict),
        "-p",
        args.prompt,
    ]
    if args.no_cnv:
        cmd.append("-no-cnv")
    if args.system_prompt:
        cmd.extend(["--system-prompt", args.system_prompt])

    # 去掉 argparse 可能留下的 '--'
    extra = [a for a in args.extra if a != "--"]
    cmd.extend(extra)

    print(f"==> AWNPU_LLM_SIM={env['AWNPU_LLM_SIM']}", file=sys.stderr)
    print("==>", " ".join(cmd), file=sys.stderr)
    print(file=sys.stderr)

    return subprocess.run(cmd, env=env, cwd=root).returncode


if __name__ == "__main__":
    raise SystemExit(main())
