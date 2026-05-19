#!/usr/bin/env bash
# AWNPU CPU 模拟模式下，用 llama-completion 跑通 Qwen2.5-0.5B（或其它 GGUF）。
#
# 用法:
#   ./scripts/awnpu/run-completion-sim.sh
#   ./scripts/awnpu/run-completion-sim.sh -p "1+1等于几？" -n 32
#   MODEL=/mnt/c/Users/wangt/models/qwen2-5/qwen2.5-0.5b-instruct-fp16.gguf DEVICE=AWNPU0 ./scripts/awnpu/run-completion-sim.sh
#
# 环境变量:
#   AWNPU_LLM_SIM   默认 1（启用 AWNPU 模拟）
#   ROOT            仓库根目录（默认：脚本上两级）
#   BUILD_DIR       构建目录（默认: $ROOT/build）
#   MODEL           GGUF 路径
#   DEVICE          后端设备名（默认: AWNPU0）
#   NGL             GPU/设备层数（默认: 99）
#   CTX_SIZE        上下文长度（默认: 8192）
#   N_PREDICT       生成长度（默认: 128）
#   PROMPT          提示词（可用 -p 覆盖）
#   NO_CNV          设为 1 时加 -no-cnv（纯补全，默认开启）
#   SYS_PROMPT      非空且未 -no-cnv 时作为 --system-prompt

set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
BIN="${BIN:-$BUILD_DIR/bin/llama-completion}"
MODEL="${MODEL:-/mnt/c/Users/wangt/models/qwen2-5/qwen2.5-0.5b-instruct-fp16.gguf}"
DEVICE="${DEVICE:-AWNPU0}"
NGL="${NGL:-99}"
CTX_SIZE="${CTX_SIZE:-8192}"
N_PREDICT="${N_PREDICT:-128}"
PROMPT="${PROMPT:-你好，请用一句话介绍你自己。}"
NO_CNV="${NO_CNV:-1}"

export AWNPU_LLM_SIM="${AWNPU_LLM_SIM:-1}"

if [[ ! -x "$BIN" ]]; then
    echo "error: $BIN 不存在，请先构建: cmake --build $BUILD_DIR --target llama-completion" >&2
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "error: 模型文件不存在: $MODEL" >&2
    exit 1
fi

args=(
    -m "$MODEL"
    --device "$DEVICE"
    -ngl "$NGL"
    -c "$CTX_SIZE"
    -n "$N_PREDICT"
)

if [[ "$NO_CNV" == "1" ]]; then
    args+=(-no-cnv)
fi

if [[ -n "${SYS_PROMPT:-}" ]]; then
    args+=(--system-prompt "$SYS_PROMPT")
fi

# 若用户未在 $@ 里传 -p/--prompt，则使用 PROMPT
has_prompt=0
for a in "$@"; do
    case "$a" in
        -p|--prompt|-f|--file) has_prompt=1 ;;
    esac
done
if [[ $has_prompt -eq 0 ]]; then
    args+=(-p "$PROMPT")
fi

echo "==> AWNPU_LLM_SIM=$AWNPU_LLM_SIM"
echo "==> $BIN ${args[*]} $*"
echo

exec env AWNPU_LLM_SIM="$AWNPU_LLM_SIM" "$BIN" "${args[@]}" "$@"
