"""DReSTG project-local Reference/cache contract.

Scientific contract
-------------------
Base reference (horizon-independent, project-local ``reference/``):
- full unique-undirected signed raw Pearson rho on canonical c<d pairs;
- coarse patch statistics after the original non-affine LayerNorm;
- contiguous zero-based sample indices.

Derived screen cache (physical branch name ``st`` preserved from the baseline):
- positive-only raw-rho cumulative-mass SoftK;
- rho <= 0 is dropped before ranking and is never refilled;
- K_s = floor(k_m * C(C-1)/2);
- packed/ragged storage without padding.

Identity rules
--------------
Builder and verifier share the same exact ranking/selector and contract helper.
Reference/cache identity uses exact stored values and exact selectors.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

BASE_SCHEMA = "drestg_base_rawrho_v1"
SCREEN_SCHEMA = "drestg_st_rawpos_softk_v1"
CONTRACT_VERSION = 1
TIE_BREAK = "exact_r_desc_then_canonical_id_asc"
BASE_PEARSON = "signed_centered_normalized_gemm_upper_i_lt_j"
BASE_STATISTICS = "coarse_mean_std_plus_non_affine_LayerNorm"
SCREEN = "raw_positive_undirected_cumulative_mass_softk"
POSITIVE_RULE = "rho_gt_0"
SOFTK_RULE = "maximal_positive_prefix_float64_sum_strict_budget"
SCREEN_BRANCH = "st"


def split_dir(root, branch: str, dataset: str, split: str) -> Path:
    return Path(root) / branch / str(dataset) / str(split)


def canonical_pairs(channels: int) -> Tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(int(channels), k=1)


def softk_budget(channels: int, mass_ratio: float) -> int:
    c = int(channels)
    u = c * (c - 1) // 2
    km = float(mass_ratio)
    if not (0.0 <= km <= 1.0):
        raise ValueError("softk_mass_ratio must satisfy 0 <= k_m <= 1")
    return int(np.floor(km * u))


def canonical_pair_ids(pair: np.ndarray, channels: int) -> np.ndarray:
    p = np.asarray(pair, dtype=np.int64)
    if p.ndim < 2 or p.shape[-1] != 2:
        raise ValueError("pair array must end with dimension 2")
    c = p[..., 0]
    d = p[..., 1]
    C = int(channels)
    if np.any(c < 0) or np.any(d >= C) or np.any(c >= d):
        raise ValueError("pair contract requires 0 <= c < d < C")
    return c * (2 * C - c - 1) // 2 + (d - c - 1)


def canonical_rank(row: np.ndarray) -> np.ndarray:
    """Exact ranking: raw rho descending; exact ties -> canonical id ascending."""
    r = np.asarray(row)
    if r.ndim != 1:
        raise ValueError("canonical_rank expects a 1-D row")
    ids = np.arange(r.shape[0], dtype=np.int64)
    return np.lexsort((ids, -r))


def select_softk(row: np.ndarray, budget: float) -> np.ndarray:
    """Maximal prefix of strictly positive raw rho under cumulative mass budget."""
    r = np.asarray(row)
    if r.ndim != 1:
        raise ValueError("select_softk expects a 1-D row")
    b = float(budget)
    if b < 0:
        raise ValueError("SoftK budget must be non-negative")
    order = canonical_rank(r)
    vals = r[order]
    keep = vals > 0
    order = order[keep]
    vals = vals[keep]
    if vals.size == 0:
        return np.empty((0,), dtype=np.int64)
    csum = np.cumsum(vals.astype(np.float64, copy=False), dtype=np.float64)
    m = int(np.searchsorted(csum, b, side="right"))
    return order[:m].astype(np.int64, copy=False)



def summary_stats(values) -> dict:
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return {"min": 0.0, "mean": 0.0, "median": 0.0, "q05": 0.0, "q95": 0.0, "max": 0.0}
    return {
        "min": float(np.min(a)),
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "q05": float(np.quantile(a, 0.05)),
        "q95": float(np.quantile(a, 0.95)),
        "max": float(np.max(a)),
    }

def base_contract(*, dataset: str, split: str, seq_len: int, c: int, l: int,
                  n: Optional[int], patch_len: int, stride: int,
                  padding_patch: str, eps: float, scale: bool,
                  data_path: str, target: str) -> dict:
    u = int(c) * (int(c) - 1) // 2
    out = {
        "schema": BASE_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "dataset": str(dataset),
        "split": str(split),
        "seq_len": int(seq_len),
        "c": int(c),
        "l": int(l),
        "u": int(u),
        "patch_len": int(patch_len),
        "stride": int(stride),
        "padding_patch": str(padding_patch),
        "eps": float(eps),
        "scale": bool(scale),
        "affine": False,
        "subtract_last": False,
        "decomposition": False,
        "statistics_num": 2,
        "data_path": str(data_path),
        "target": str(target),
        "pearson": BASE_PEARSON,
        "statistics": BASE_STATISTICS,
        "horizon_independent": True,
    }
    if n is not None:
        out["n"] = int(n)
    return out


def screen_contract(*, dataset: str, split: str, n: Optional[int], c: int,
                    mass_ratio: float) -> dict:
    u = int(c) * (int(c) - 1) // 2
    out = {
        "schema": SCREEN_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "dataset": str(dataset),
        "split": str(split),
        "c": int(c),
        "u": int(u),
        "k_m": float(mass_ratio),
        "budget": softk_budget(int(c), float(mass_ratio)),
        "base_schema": BASE_SCHEMA,
        "screen": SCREEN,
        "positive_rule": POSITIVE_RULE,
        "no_refill": True,
        "softk_rule": SOFTK_RULE,
        "tie_break": TIE_BREAK,
        "packed": True,
        "padding": False,
        "storage_branch": SCREEN_BRANCH,
    }
    if n is not None:
        out["n"] = int(n)
    return out


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _check_expected(meta: dict, expected: Optional[Dict], label: str) -> None:
    if not expected:
        return
    for key, value in expected.items():
        if key not in meta:
            raise ValueError("%s missing mandatory contract field %s" % (label, key))
        if meta[key] != value:
            raise ValueError("%s contract mismatch %s: %r != %r" % (label, key, meta[key], value))


def _load_meta(directory: Path, schema: str, expected: Optional[Dict],
               label: str, require_ok: bool) -> dict:
    meta = _read_json(directory / "meta.json")
    if meta.get("schema") != schema:
        raise ValueError("reference schema mismatch at %s: %r != %r" %
                         (directory, meta.get("schema"), schema))
    _check_expected(meta, expected, label)
    if require_ok:
        ok = _read_json(directory / "ok.json")
        if ok.get("status") != "VERIFIED" or ok.get("schema") != schema:
            raise ValueError("reference is not VERIFIED: %s" % directory)
        if ok.get("contract_version") != CONTRACT_VERSION:
            raise ValueError("reference VERIFIED marker contract version mismatch: %s" % directory)
    return meta


def _load_base_dir(directory: Path, expected: Optional[Dict] = None,
                   mmap_mode: str = "r", require_ok: bool = True):
    d = Path(directory)
    meta = _load_meta(d, BASE_SCHEMA, expected, "base reference", require_ok)
    rho = np.load(d / "rho.npy", mmap_mode=mmap_mode)
    stat = np.load(d / "stat.npy", mmap_mode=mmap_mode)
    idx = np.load(d / "idx.npy", mmap_mode=mmap_mode)
    n, c, l = int(meta["n"]), int(meta["c"]), int(meta["l"])
    u = c * (c - 1) // 2
    if rho.shape != (n, u):
        raise ValueError("rho shape mismatch %r != %r" % (rho.shape, (n, u)))
    if stat.shape != (n, c, l, 2):
        raise ValueError("stat shape mismatch %r != %r" % (stat.shape, (n, c, l, 2)))
    if idx.shape != (n,):
        raise ValueError("idx shape mismatch %r != %r" % (idx.shape, (n,)))
    if not np.array_equal(np.asarray(idx), np.arange(n, dtype=np.int64)):
        raise ValueError("reference idx must be contiguous zero-based")
    return meta, {"rho": rho, "stat": stat, "idx": idx}


def _load_screen_dir(directory: Path, expected: Optional[Dict] = None,
                     mmap_mode: str = "r", require_ok: bool = True):
    d = Path(directory)
    meta = _load_meta(d, SCREEN_SCHEMA, expected, "DReSTG ST SoftK screen", require_ok)
    off = np.load(d / "off.npy", mmap_mode=mmap_mode)
    pair = np.load(d / "pair.npy", mmap_mode=mmap_mode)
    r = np.load(d / "r.npy", mmap_mode=mmap_mode)
    n, total = int(meta["n"]), int(meta["total"])
    if off.shape != (n + 1,):
        raise ValueError("screen off shape mismatch %r != %r" % (off.shape, (n + 1,)))
    if pair.shape != (total, 2):
        raise ValueError("screen pair shape mismatch %r != %r" % (pair.shape, (total, 2)))
    if r.shape != (total,):
        raise ValueError("screen r shape mismatch %r != %r" % (r.shape, (total,)))
    off_arr = np.asarray(off, dtype=np.int64)
    if int(off_arr[0]) != 0 or int(off_arr[-1]) != total or np.any(np.diff(off_arr) < 0):
        raise ValueError("screen offsets are invalid")
    counts = np.diff(off_arr)
    derived = {
        "total": int(total),
        "min_pairs": int(counts.min()) if counts.size else 0,
        "max_pairs": int(counts.max()) if counts.size else 0,
        "mean_pairs": float(counts.mean()) if counts.size else 0.0,
    }
    for key, value in derived.items():
        if key not in meta or meta[key] != value:
            raise ValueError("screen derived metadata mismatch %s: %r != %r" %
                             (key, meta.get(key), value))
    return meta, {"off": off, "pair": pair, "r": r}


def load_base(root, dataset: str, split: str, expected: Optional[Dict] = None,
              mmap_mode: str = "r"):
    return _load_base_dir(split_dir(root, "base", dataset, split), expected, mmap_mode, True)


def load_screen(root, dataset: str, split: str, expected: Optional[Dict] = None,
                mmap_mode: str = "r"):
    return _load_screen_dir(split_dir(root, SCREEN_BRANCH, dataset, split), expected, mmap_mode, True)


def verify_base_dir(directory: Path, expected: Optional[Dict] = None,
                    chunk_rows: int = 64, require_ok: bool = True) -> dict:
    meta, base = _load_base_dir(directory, expected=expected, mmap_mode="r", require_ok=require_ok)
    rho, stat = base["rho"], base["stat"]
    n = int(meta["n"])
    step = max(1, int(chunk_rows))
    for lo in range(0, n, step):
        hi = min(n, lo + step)
        rr = np.asarray(rho[lo:hi])
        ss = np.asarray(stat[lo:hi])
        if not np.isfinite(rr).all():
            raise ValueError("non-finite base rho in rows [%d,%d)" % (lo, hi))
        if not np.isfinite(ss).all():
            raise ValueError("non-finite base statistics in rows [%d,%d)" % (lo, hi))
        if rr.size and (np.any(rr < -1) or np.any(rr > 1)):
            raise ValueError("signed rho range violation in rows [%d,%d)" % (lo, hi))
    return meta


def verify_screen_dir(directory: Path, base_directory: Path,
                      expected: Optional[Dict] = None,
                      base_expected: Optional[Dict] = None,
                      require_ok: bool = True) -> dict:
    meta, screen = _load_screen_dir(directory, expected=expected, mmap_mode="r", require_ok=require_ok)
    bmeta, base = _load_base_dir(base_directory, expected=base_expected, mmap_mode="r", require_ok=True)
    if int(meta["n"]) != int(bmeta["n"]) or int(meta["c"]) != int(bmeta["c"]) or int(meta["u"]) != int(bmeta["u"]):
        raise ValueError("screen/base metadata mismatch")
    c, n, budget = int(meta["c"]), int(meta["n"]), float(meta["budget"])
    off = np.asarray(screen["off"], dtype=np.int64)
    pair, rv, rho = screen["pair"], screen["r"], base["rho"]
    selected_nonpositive = 0
    neg_counts = np.empty(n, dtype=np.int64)
    zero_counts = np.empty(n, dtype=np.int64)
    pos_counts = np.empty(n, dtype=np.int64)
    pos_mass = np.empty(n, dtype=np.float64)
    selected_counts = np.empty(n, dtype=np.int64)
    selected_mass = np.empty(n, dtype=np.float64)
    for i in range(n):
        a, b = int(off[i]), int(off[i + 1])
        pp = np.asarray(pair[a:b], dtype=np.int64)
        vv = np.asarray(rv[a:b])
        if not np.isfinite(vv).all():
            raise ValueError("screen contains non-finite r at sample %d" % i)
        if vv.size and (np.any(vv <= 0) or np.any(vv > 1)):
            raise ValueError("screen r must satisfy 0 < r <= 1 at sample %d" % i)
        stored_ids = canonical_pair_ids(pp, c) if b > a else np.empty((0,), dtype=np.int64)
        row = np.asarray(rho[i])
        expected_ids = select_softk(row, budget)
        if not np.array_equal(stored_ids, expected_ids):
            raise ValueError("packed pair IDs do not equal deterministic raw-positive SoftK at sample %d" % i)
        expected_r = row[expected_ids]
        if not np.array_equal(vv, expected_r):
            raise ValueError("screen r does not exactly match base signed rho at sample %d" % i)
        selected_nonpositive += int(np.count_nonzero(expected_r <= 0))
        neg_counts[i] = np.count_nonzero(row < 0)
        zero_counts[i] = np.count_nonzero(row == 0)
        pos_counts[i] = np.count_nonzero(row > 0)
        pos_mass[i] = np.asarray(row[row > 0], dtype=np.float64).sum(dtype=np.float64)
        selected_counts[i] = int(expected_ids.size)
        selected_mass[i] = np.asarray(expected_r, dtype=np.float64).sum(dtype=np.float64)
        if selected_mass[i] > budget:
            raise ValueError("selected mass exceeds strict budget at sample %d" % i)
    if selected_nonpositive != 0:
        raise ValueError("positive-only invariant violated")
    if int(meta.get("selected_nonpositive_count", -1)) != 0:
        raise ValueError("metadata selected_nonpositive_count invariant violated")
    if meta.get("no_refill") is not True:
        raise ValueError("metadata no_refill invariant violated")
    expected_diag = {
        "negative_count": summary_stats(neg_counts),
        "zero_count": summary_stats(zero_counts),
        "positive_count": summary_stats(pos_counts),
        "positive_mass": summary_stats(pos_mass),
        "selected_pair_count": summary_stats(selected_counts),
        "selected_mass": summary_stats(selected_mass),
    }
    if meta.get("truncation_diagnostics") != expected_diag:
        raise ValueError("truncation diagnostics metadata mismatch")
    return meta


def verify_base_full(root, dataset: str, split: str, expected: Optional[Dict] = None,
                     chunk_rows: int = 64) -> dict:
    return verify_base_dir(split_dir(root, "base", dataset, split), expected, chunk_rows, True)


def verify_screen_full(root, dataset: str, split: str, expected: Optional[Dict] = None,
                       base_expected: Optional[Dict] = None) -> dict:
    return verify_screen_dir(
        split_dir(root, SCREEN_BRANCH, dataset, split),
        split_dir(root, "base", dataset, split),
        expected=expected, base_expected=base_expected, require_ok=True,
    )


def screen_payload(screen: Dict[str, np.ndarray], index: int) -> dict:
    off = screen["off"]
    a, b = int(off[index]), int(off[index + 1])
    return {"pair": screen["pair"][a:b], "r": screen["r"][a:b]}
