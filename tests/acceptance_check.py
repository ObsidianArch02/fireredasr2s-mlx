#!/usr/bin/env python3
"""Standalone field-by-field acceptance check between two result.jsonl files.

Usage:
    python acceptance_check.py <torch_result.jsonl> <mlx_result.jsonl>

Both files are parsed as JSON lines, records are sorted by ``uttid`` on each
side, then EVERY field of every record is compared for exact equality —
including nested sentences/words lists and the JSON number type (``1`` and
``1.0`` count as different, matching byte-level identity of the emitted line).

Exit code: 0 = all fields identical; 1 = any difference (diffs printed);
2 = usage / IO / malformed-input error.

No third-party imports: runs with any Python 3.
"""

from __future__ import annotations

import json
import sys


def load_records(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from e
    if not records:
        raise ValueError(f"{path}: no JSON records found")
    return records


def _type_name(v):
    # bool is a subclass of int: keep the distinction (True != 1 in the jsonl)
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, (int, float)):
        return type(v).__name__  # "int" vs "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        return "list"
    if isinstance(v, dict):
        return "object"
    return type(v).__name__


def diff_value(path, a, b, out, max_out):
    if len(out) >= max_out:
        return
    ta, tb = _type_name(a), _type_name(b)
    if ta != tb:
        out.append(f"{path}: type {ta} != {tb} ({a!r} vs {b!r})")
    elif isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{path}.{k}: missing on the left")
            elif k not in b:
                out.append(f"{path}.{k}: missing on the right")
            else:
                diff_value(f"{path}.{k}", a[k], b[k], out, max_out)
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diff_value(f"{path}[{i}]", x, y, out, max_out)
    elif a != b:
        out.append(f"{path}: {a!r} != {b!r}")


def compare_files(torch_path, mlx_path, max_diffs=50):
    """Compare two result.jsonl files field-by-field (sorted by uttid).

    Returns ``(diffs, n_compared)``: a list of human-readable difference
    strings (capped at ``max_diffs``) and the number of uttids compared.
    """
    torch_recs = sorted(load_records(torch_path), key=lambda r: r["uttid"])
    mlx_recs = sorted(load_records(mlx_path), key=lambda r: r["uttid"])

    t_ids = [r["uttid"] for r in torch_recs]
    m_ids = [r["uttid"] for r in mlx_recs]
    out = []
    if t_ids != m_ids:
        only_t = [u for u in t_ids if u not in m_ids]
        only_m = [u for u in m_ids if u not in t_ids]
        out.append(f"uttid sets differ: only-torch={only_t} only-mlx={only_m}")

    by_t = {r["uttid"]: r for r in torch_recs}
    by_m = {r["uttid"]: r for r in mlx_recs}
    compared = 0
    for uttid in t_ids:
        if uttid not in by_m:
            continue
        compared += 1
        diff_value(uttid, by_t[uttid], by_m[uttid], out, max_diffs)
    return out, compared


def main(argv):
    if len(argv) != 2:
        print("usage: python acceptance_check.py "
              "<torch_result.jsonl> <mlx_result.jsonl>")
        return 2
    try:
        diffs, compared = compare_files(argv[0], argv[1])
    except (OSError, ValueError, KeyError) as e:
        print(f"ERROR: {e}")
        return 2
    if diffs:
        print(f"FAIL: {len(diffs)}+ field difference(s) between "
              f"{argv[0]} and {argv[1]}:")
        for d in diffs:
            print("  -", d)
        return 1
    print(f"PASS: {compared} record(s) identical in every field")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
