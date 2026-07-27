"""Read Eonego `dumpgraph` EONGF01 feature dumps."""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

MAGIC = b"EONGF01!"
VERSION = 1
SCHEMA_HASH = 0x47463101
MAX_NODES = 32
NODE_FEATURES = 12
EDGE_FEATURES = 8
GLOBAL_FEATURES = 32
RECORD_BYTES = 4 + MAX_NODES * NODE_FEATURES + MAX_NODES * MAX_NODES * EDGE_FEATURES + GLOBAL_FEATURES
HEADER = struct.Struct("<8sIIII")

NODE_BYTES = MAX_NODES * NODE_FEATURES
EDGE_BYTES = MAX_NODES * MAX_NODES * EDGE_FEATURES


@dataclass(frozen=True)
class GraphDump:
    node_count: np.ndarray
    stm: np.ndarray
    in_check: np.ndarray
    checker_count: np.ndarray
    nodes: np.ndarray
    edges: np.ndarray
    globals: np.ndarray


def read_dump(path: str) -> GraphDump:
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) < HEADER.size:
        raise ValueError(f"truncated graph dump: {path}")
    magic, version, schema, rec_bytes, count = HEADER.unpack_from(raw, 0)
    if magic != MAGIC:
        raise ValueError(f"bad graph dump magic {magic!r}")
    if version != VERSION:
        raise ValueError(f"unsupported graph dump version {version}")
    if schema != SCHEMA_HASH:
        raise ValueError(f"schema mismatch {schema:#x} != {SCHEMA_HASH:#x}")
    if rec_bytes != RECORD_BYTES:
        raise ValueError(f"record size mismatch {rec_bytes} != {RECORD_BYTES}")
    want = HEADER.size + count * RECORD_BYTES
    if len(raw) != want:
        raise ValueError(f"bad graph dump length {len(raw)} != {want}")

    body = memoryview(raw)[HEADER.size:]
    flat = np.frombuffer(body, dtype=np.uint8).reshape(count, RECORD_BYTES)
    nodes0 = 4
    edges0 = nodes0 + NODE_BYTES
    globals0 = edges0 + EDGE_BYTES
    return GraphDump(
        node_count=flat[:, 0].copy(),
        stm=flat[:, 1].copy(),
        in_check=flat[:, 2].copy(),
        checker_count=flat[:, 3].copy(),
        nodes=flat[:, nodes0:edges0].reshape(count, MAX_NODES, NODE_FEATURES).copy(),
        edges=flat[:, edges0:globals0].reshape(count, MAX_NODES, MAX_NODES, EDGE_FEATURES).copy(),
        globals=flat[:, globals0:].reshape(count, GLOBAL_FEATURES).copy(),
    )