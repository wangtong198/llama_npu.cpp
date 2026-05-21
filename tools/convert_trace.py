# 用法

# # 默认：只转 sched 1（一次 prefill / 图执行）
# python3 convert_trace.py ../llama_graph_exec.log -o llama_graph_exec.json

# # 指定某次 decode
# python3 convert_trace.py ../llama_graph_exec.log -o sched2.json --sched 2

# # 转全部 sched（prefill + 所有 decode token）
# python3 convert_trace.py ../llama_graph_exec.log -o all.json --all-scheds



#!/usr/bin/env python3
"""Convert llama_graph_exec.log to Chrome Trace JSON for Perfetto UI."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# ggml element sizes (bytes); quantized types use a conservative estimate.
TYPE_SIZE: dict[str, int] = {
    "f32": 4,
    "f16": 2,
    "bf16": 2,
    "f64": 8,
    "i8": 1,
    "i16": 2,
    "i32": 4,
    "i64": 8,
}

HEADER_RE = re.compile(
    r"^=+ sched (\d+) split (\d+)/(\d+) dispatch=(\S+) n_nodes=(\d+) =+$"
)
NODE_RE = re.compile(
    r"^seq=(\d+) op=(\S+) name=(.+?) type=(\S+) "
    r"shape=\[([^\]]+)\] contiguous: (\S+) fallback: (\S+) "
    r"time-elapsed: ([\d.]+)ms$"
)
SRC_RE = re.compile(
    r"^\s+src\[(\d+)\] op=(\S+) name=(.+?) type=(\S+) "
    r"shape=\[([^\]]+)\] contiguous: (\S+)$"
)

TID_DISPATCH_OPS = 1
TID_FALLBACK_OPS = 2
TID_DISPATCH_MEM = 3
TID_FALLBACK_MEM = 4


def parse_shape(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def tensor_bytes(type_name: str, shape: list[int]) -> int:
    elems = 1
    for dim in shape:
        elems *= dim
    type_size = TYPE_SIZE.get(type_name, 4)
    return elems * type_size


def format_bytes(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.2f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def elapsed_us(raw_ms: str) -> int:
    ms = float(raw_ms)
    if ms <= 0.0:
        return 1
    return max(int(round(ms * 1000.0)), 1)


def parse_nodes(log_path: Path, sched_filter: int | None) -> tuple[list[dict], str]:
    nodes: list[dict] = []
    dispatch = "UNKNOWN"
    active_sched: int | None = None
    current: dict | None = None

    with log_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            header = HEADER_RE.match(line)
            if header:
                sched = int(header.group(1))
                dispatch = header.group(4)
                if sched_filter is not None and sched != sched_filter:
                    active_sched = None
                    current = None
                    continue
                active_sched = sched
                current = None
                continue

            if active_sched is None:
                continue

            node_match = NODE_RE.match(line)
            if node_match:
                if current is not None:
                    nodes.append(current)
                seq, op, name, type_name, shape_raw, contiguous, fallback, elapsed = (
                    node_match.groups()
                )
                shape = parse_shape(shape_raw)
                current = {
                    "seq": int(seq),
                    "op": op,
                    "name": name,
                    "type": type_name,
                    "shape": shape,
                    "contiguous": contiguous == "true",
                    "fallback": fallback == "true",
                    "elapsed_ms": elapsed,
                    "dispatch": dispatch,
                    "sched": active_sched,
                    "srcs": [],
                }
                continue

            if current is not None:
                src_match = SRC_RE.match(line)
                if src_match:
                    _, src_op, src_name, src_type, src_shape_raw, src_contiguous = (
                        src_match.groups()
                    )
                    src_shape = parse_shape(src_shape_raw)
                    current["srcs"].append(
                        {
                            "op": src_op,
                            "name": src_name,
                            "type": src_type,
                            "shape": src_shape,
                            "contiguous": src_contiguous == "true",
                            "bytes": tensor_bytes(src_type, src_shape),
                        }
                    )

    if current is not None:
        nodes.append(current)

    if not nodes:
        sched_hint = f"sched {sched_filter}" if sched_filter is not None else "any sched"
        raise ValueError(f"no nodes found for {sched_hint} in {log_path}")

    return nodes, dispatch


def thread_metadata(name: str, tid: int) -> dict:
    return {
        "name": "thread_name",
        "ph": "M",
        "pid": 1,
        "tid": tid,
        "args": {"name": name},
    }


def make_op_event(
    node: dict,
    ts_us: int,
    dur_us: int,
    tid: int,
    cat: str,
) -> dict:
    return {
        "name": f"{node['op']} ({node['name']})",
        "cat": cat,
        "ph": "X",
        "ts": ts_us,
        "dur": dur_us,
        "pid": 1,
        "tid": tid,
        "args": {
            "seq": node["seq"],
            "sched": node["sched"],
            "dispatch": node["dispatch"],
            "fallback": node["fallback"],
            "type": node["type"],
            "shape": node["shape"],
            "contiguous": node["contiguous"],
            "elapsed_ms": float(node["elapsed_ms"]),
        },
    }


def make_mem_event(
    node: dict,
    ts_us: int,
    dur_us: int,
    tid: int,
    cat: str,
    output_bytes: int,
    input_bytes: int,
) -> dict:
    return {
        "name": f"out {format_bytes(output_bytes)} / in {format_bytes(input_bytes)}",
        "cat": cat,
        "ph": "X",
        "ts": ts_us,
        "dur": dur_us,
        "pid": 1,
        "tid": tid,
        "args": {
            "seq": node["seq"],
            "output_bytes": output_bytes,
            "input_bytes": input_bytes,
            "total_io_bytes": output_bytes + input_bytes,
            "output_human": format_bytes(output_bytes),
            "input_human": format_bytes(input_bytes),
        },
    }


def build_trace(nodes: list[dict], dispatch: str) -> dict:
    events: list[dict] = [
        thread_metadata(f"{dispatch} ops", TID_DISPATCH_OPS),
        thread_metadata("CPU fallback ops", TID_FALLBACK_OPS),
        thread_metadata(f"{dispatch} memory", TID_DISPATCH_MEM),
        thread_metadata("CPU fallback memory", TID_FALLBACK_MEM),
    ]

    ts_us = 0
    for node in nodes:
        dur_us = elapsed_us(node["elapsed_ms"])
        output_bytes = tensor_bytes(node["type"], node["shape"])
        input_bytes = sum(src["bytes"] for src in node["srcs"])

        if node["fallback"]:
            op_tid = TID_FALLBACK_OPS
            mem_tid = TID_FALLBACK_MEM
            cat = "CPU fallback"
        else:
            op_tid = TID_DISPATCH_OPS
            mem_tid = TID_DISPATCH_MEM
            cat = dispatch

        events.append(make_op_event(node, ts_us, dur_us, op_tid, cat))
        events.append(
            make_mem_event(
                node,
                ts_us,
                dur_us,
                mem_tid,
                cat,
                output_bytes,
                input_bytes,
            )
        )
        ts_us += dur_us

    return {
        "traceEvents": events,
        "displayTimeUnit": "ms",
        "metadata": {
            "dispatch": dispatch,
            "node_count": len(nodes),
            "total_duration_us": ts_us,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert llama_graph_exec.log to Chrome Trace JSON for Perfetto."
    )
    parser.add_argument(
        "log",
        nargs="?",
        default="../llama_graph_exec.log",
        help="path to llama_graph_exec.log (default: ../llama_graph_exec.log)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="llama_graph_exec.json",
        help="output JSON path (default: llama_graph_exec.json)",
    )
    parser.add_argument(
        "--sched",
        type=int,
        default=1,
        help="only convert the given sched round (default: 1)",
    )
    parser.add_argument(
        "--all-scheds",
        action="store_true",
        help="convert all sched rounds into one timeline",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.is_file():
        print(f"error: log file not found: {log_path}", file=sys.stderr)
        return 1

    sched_filter = None if args.all_scheds else args.sched
    nodes, dispatch = parse_nodes(log_path, sched_filter)
    trace = build_trace(nodes, dispatch)

    out_path = Path(args.output)
    with out_path.open("w", encoding="utf-8") as out:
        json.dump(trace, out, indent=2)
        out.write("\n")

    print(
        f"wrote {out_path} "
        f"(dispatch={dispatch}, nodes={len(nodes)}, "
        f"duration={trace['metadata']['total_duration_us'] / 1000:.3f} ms)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
