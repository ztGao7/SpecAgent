#!/usr/bin/env python3
"""Bind a command to one NUMA node's CPUs and future memory allocations."""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import NoReturn


class Bitmask(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_ulong),
        ("maskp", ctypes.POINTER(ctypes.c_ulong)),
    ]


def parse_cpu_list(value: str) -> set[int]:
    cpus: set[int] = set()
    for part in value.strip().split(","):
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            cpus.update(range(start, end + 1))
        elif part:
            cpus.add(int(part))
    return cpus


def fail(message: str) -> NoReturn:
    print(f"numa_exec: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    if len(sys.argv) < 3:
        fail("usage: numa_exec.py NODE COMMAND [ARG ...]")

    node = int(sys.argv[1])
    command = sys.argv[2:]
    cpulist_path = Path(f"/sys/devices/system/node/node{node}/cpulist")
    if not cpulist_path.is_file():
        fail(f"NUMA node {node} does not exist")
    cpus = parse_cpu_list(cpulist_path.read_text(encoding="ascii"))
    if not cpus:
        fail(f"NUMA node {node} has no CPUs")
    os.sched_setaffinity(0, cpus)

    try:
        numa = ctypes.CDLL("libnuma.so.1", use_errno=True)
    except OSError as exc:
        fail(f"cannot load libnuma.so.1: {exc}")
    numa.numa_available.restype = ctypes.c_int
    if numa.numa_available() < 0:
        fail("NUMA is not available")

    bitmask_ptr = ctypes.POINTER(Bitmask)
    numa.numa_allocate_nodemask.restype = bitmask_ptr
    numa.numa_bitmask_clearall.argtypes = [bitmask_ptr]
    numa.numa_bitmask_setbit.argtypes = [bitmask_ptr, ctypes.c_uint]
    numa.numa_set_membind.argtypes = [bitmask_ptr]
    numa.numa_get_membind.restype = bitmask_ptr
    numa.numa_bitmask_isbitset.argtypes = [bitmask_ptr, ctypes.c_uint]
    numa.numa_bitmask_isbitset.restype = ctypes.c_int
    numa.numa_bitmask_weight.argtypes = [bitmask_ptr]
    numa.numa_bitmask_weight.restype = ctypes.c_uint
    numa.numa_bitmask_free.argtypes = [bitmask_ptr]

    mask = numa.numa_allocate_nodemask()
    if not mask:
        fail("cannot allocate NUMA node mask")
    numa.numa_bitmask_clearall(mask)
    numa.numa_bitmask_setbit(mask, node)
    numa.numa_set_membind(mask)

    effective = numa.numa_get_membind()
    valid = bool(
        effective
        and numa.numa_bitmask_isbitset(effective, node)
        and numa.numa_bitmask_weight(effective) == 1
    )
    if effective:
        numa.numa_bitmask_free(effective)
    numa.numa_bitmask_free(mask)
    if not valid:
        fail(f"kernel did not apply memory binding for NUMA node {node}")

    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
