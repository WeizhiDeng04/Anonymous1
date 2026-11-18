#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_provider.data_loader import Dataset_ETTh, Dataset_ETTm, Dataset_Custom
from utils.normal_utils import get_statistics_full
from utils.patch_utils import compute_patch_layout
from utils.ref_cache import (
    BASE_SCHEMA, SCREEN_SCHEMA, CONTRACT_VERSION, SCREEN_BRANCH,
    base_contract, screen_contract, canonical_pairs, select_softk, summary_stats,
    load_base, split_dir,
    verify_base_full, verify_screen_full,
    verify_base_dir, verify_screen_dir,
)

DATASET_CLASS = {
    "ETTh1": Dataset_ETTh, "ETTh2": Dataset_ETTh,
    "ETTm1": Dataset_ETTm, "ETTm2": Dataset_ETTm,
    "Weather": Dataset_Custom, "Electricity": Dataset_Custom, "Traffic": Dataset_Custom,
}


def pearson_gemm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"pearson_gemm expects [B,C,T], got {tuple(x.shape)}")
    xc = x - x.mean(dim=-1, keepdim=True)
    norm = torch.linalg.vector_norm(xc, ord=2, dim=-1, keepdim=True)
    z = torch.where(norm > 0, xc / norm.clamp_min(eps), torch.zeros_like(xc))
    return torch.matmul(z, z.transpose(-1, -2)).clamp(-1.0, 1.0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--action", choices=["build", "verify"], required=True)
    p.add_argument("--config_yaml", required=True)
    p.add_argument("--dataset", required=True, choices=tuple(DATASET_CLASS))
    p.add_argument("--split", required=True, choices=["train", "val", "test"])
    p.add_argument("--root", default="reference")
    p.add_argument("--seq_len", type=int, default=336)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch", type=int, default=64)
    return p.parse_args()


def make_dataset(cfg, dataset, split, seq_len):
    return DATASET_CLASS[dataset](
        root_path=cfg["root_path"], flag=split, size=[seq_len, 0, 0],
        data_path=cfg["data_path"], target=cfg.get("target", cfg.get("traget", "OT")),
        scale=bool(cfg.get("scale", True)),
    )


def _tmp_dir(out: Path) -> Path:
    t = out.with_name(out.name + f".tmp.{os.getpid()}")
    if t.exists():
        shutil.rmtree(t)
    t.mkdir(parents=True, exist_ok=False)
    return t


def _write_meta(tmp: Path, meta: dict) -> None:
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def _publish_verified(tmp: Path, out: Path, schema: str) -> None:
    (tmp / "ok.json").write_text(
        json.dumps(
            {"status": "VERIFIED", "schema": schema, "contract_version": CONTRACT_VERSION},
            indent=2, sort_keys=True,
        ),
        encoding="utf-8",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError("refusing to overwrite existing Reference: %s" % out)
    tmp.rename(out)


def _base_expected(a, cfg, n):
    return base_contract(
        dataset=a.dataset, split=a.split, seq_len=int(a.seq_len),
        c=int(cfg["enc_in"]), l=int(cfg["coarse_patch_count"]), n=int(n),
        patch_len=int(cfg["fine_patch_len"]), stride=int(cfg["fine_patch_stride"]),
        padding_patch=str(cfg.get("padding_patch", "start")),
        eps=float(cfg.get("eps", 1e-5)), scale=bool(cfg.get("scale", True)),
        data_path=str(cfg["data_path"]), target=str(cfg.get("target", cfg.get("traget", "OT"))),
    )


def _screen_expected(a, cfg, n):
    if "softk_mass_ratio" not in cfg:
        raise RuntimeError("DReSTG config missing softk_mass_ratio")
    return screen_contract(
        dataset=a.dataset, split=a.split, n=int(n), c=int(cfg["enc_in"]),
        mass_ratio=float(cfg["softk_mass_ratio"]),
    )


def build_base(a, cfg):
    c = int(cfg["enc_in"])
    l = int(cfg["coarse_patch_count"])
    ds = make_dataset(cfg, a.dataset, a.split, a.seq_len)
    n = len(ds)
    u = c * (c - 1) // 2
    expected = _base_expected(a, cfg, n)
    out = split_dir(a.root, "base", a.dataset, a.split)
    if out.exists():
        verify_base_full(a.root, a.dataset, a.split, expected=expected, chunk_rows=max(1, int(a.batch)))
        print(f"[REF BASE EXISTS+VERIFIED] {out}")
        return

    tmp = _tmp_dir(out)
    rho_mm = np.lib.format.open_memmap(tmp / "rho.npy", mode="w+", dtype=np.float32, shape=(n, u))
    stat_mm = np.lib.format.open_memmap(tmp / "stat.npy", mode="w+", dtype=np.float32, shape=(n, c, l, 2))
    idx_mm = np.lib.format.open_memmap(tmp / "idx.npy", mode="w+", dtype=np.int64, shape=(n,))
    _, _, padding_len = compute_patch_layout(a.seq_len, l, int(cfg["fine_patch_len"]), int(cfg["fine_patch_stride"]))
    norm = nn.LayerNorm([c, l], elementwise_affine=False)
    device = torch.device(a.device)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for Reference build but unavailable")
    norm = norm.to(device).eval()
    tri = torch.triu_indices(c, c, offset=1, device=device)
    bs = max(1, int(a.batch))
    t0 = time.time()
    with torch.no_grad():
        for lo in range(0, n, bs):
            hi = min(n, lo + bs)
            x_np = np.stack([ds[i][0] for i in range(lo, hi)], axis=0)
            x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
            x_ct = x.permute(0, 2, 1)
            
            vals = pearson_gemm(x_ct)[:, tri[0], tri[1]]
            rho_mm[lo:hi] = vals.detach().cpu().numpy().astype(np.float32, copy=False)
            st = get_statistics_full(
                x_in=x, padding_len=padding_len,
                padding_mode=cfg.get("padding_patch", "start"), block_num=l,
                subtract_last=False, eps=float(cfg.get("eps", 1e-5)),
            )
            st = norm(st.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            stat_mm[lo:hi] = st.detach().cpu().numpy().astype(np.float32, copy=False)
            idx_mm[lo:hi] = np.arange(lo, hi, dtype=np.int64)
            if hi % 1000 == 0 or hi == n:
                print(f"[REF BASE] {a.dataset} {a.split} {hi}/{n}")
    rho_mm.flush(); stat_mm.flush(); idx_mm.flush()
    del rho_mm, stat_mm, idx_mm
    meta = dict(expected)
    meta["elapsed_seconds"] = float(time.time() - t0)
    _write_meta(tmp, meta)
    verify_base_dir(tmp, expected=expected, chunk_rows=max(1, int(a.batch)), require_ok=False)
    _publish_verified(tmp, out, BASE_SCHEMA)
    print(f"[REF BASE BUILT+VERIFIED] {out}")


def build_screen(a, cfg):
    c = int(cfg["enc_in"])
    bmeta, base = load_base(a.root, a.dataset, a.split, expected=None, mmap_mode="r")
    n = int(bmeta["n"])
    expected = _screen_expected(a, cfg, n)
    budget = float(expected["budget"])
    out = split_dir(a.root, SCREEN_BRANCH, a.dataset, a.split)
    if out.exists():
        verify_screen_full(a.root, a.dataset, a.split, expected=expected)
        print(f"[REF SCREEN EXISTS+VERIFIED] {out}")
        return

    tmp = _tmp_dir(out)
    rho = base["rho"]
    off = np.zeros((n + 1,), dtype=np.int64)
    raw_ids = tmp / "ids.bin"
    raw_r = tmp / "r.bin"

    neg_counts = np.empty(n, dtype=np.int64)
    zero_counts = np.empty(n, dtype=np.int64)
    pos_counts = np.empty(n, dtype=np.int64)
    pos_mass = np.empty(n, dtype=np.float64)
    selected_counts = np.empty(n, dtype=np.int64)
    selected_mass = np.empty(n, dtype=np.float64)

    with raw_ids.open("wb") as f_id, raw_r.open("wb") as f_r:
        total = 0
        for i in range(n):
            row = np.asarray(rho[i])
            neg_counts[i] = np.count_nonzero(row < 0)
            zero_counts[i] = np.count_nonzero(row == 0)
            pos_counts[i] = np.count_nonzero(row > 0)
            pos_mass[i] = np.asarray(row[row > 0], dtype=np.float64).sum(dtype=np.float64)
            ids = select_softk(row, budget)
            vv = row[ids]
            if vv.size and np.any(vv <= 0):
                raise RuntimeError("positive-only selector returned nonpositive edge")
            ids.astype(np.int32, copy=False).tofile(f_id)
            vv.astype(np.float32, copy=False).tofile(f_r)
            selected_counts[i] = int(ids.size)
            selected_mass[i] = np.asarray(vv, dtype=np.float64).sum(dtype=np.float64)
            total += int(ids.size)
            off[i + 1] = total
            if (i + 1) % 1000 == 0 or i + 1 == n:
                print(f"[REF SCREEN] {a.dataset} {a.split} {i+1}/{n} total_pairs={total}")

    np.save(tmp / "off.npy", off)
    pair_dtype = np.int16 if c < np.iinfo(np.int16).max else np.int32
    total = int(off[-1])
    pair_mm = np.lib.format.open_memmap(tmp / "pair.npy", mode="w+", dtype=pair_dtype, shape=(total, 2))
    r_mm = np.lib.format.open_memmap(tmp / "r.npy", mode="w+", dtype=np.float32, shape=(total,))
    ids_mm = np.memmap(raw_ids, mode="r", dtype=np.int32, shape=(total,)) if total else np.empty((0,), dtype=np.int32)
    rv_mm = np.memmap(raw_r, mode="r", dtype=np.float32, shape=(total,)) if total else np.empty((0,), dtype=np.float32)
    src, dst = canonical_pairs(c)
    chunk = 1_000_000
    for lo in range(0, total, chunk):
        hi = min(total, lo + chunk)
        ii = np.asarray(ids_mm[lo:hi], dtype=np.int64)
        pair_mm[lo:hi, 0] = src[ii]
        pair_mm[lo:hi, 1] = dst[ii]
        r_mm[lo:hi] = np.asarray(rv_mm[lo:hi], dtype=np.float32)
    pair_mm.flush(); r_mm.flush()
    del pair_mm, r_mm, ids_mm, rv_mm
    raw_ids.unlink(missing_ok=True); raw_r.unlink(missing_ok=True)

    counts = np.diff(off)
    meta = dict(expected)
    meta.update({
        "total": total,
        "min_pairs": int(counts.min()) if counts.size else 0,
        "max_pairs": int(counts.max()) if counts.size else 0,
        "mean_pairs": float(counts.mean()) if counts.size else 0.0,
        "selected_nonpositive_count": 0,
        "truncation_diagnostics": {
            "negative_count": summary_stats(neg_counts),
            "zero_count": summary_stats(zero_counts),
            "positive_count": summary_stats(pos_counts),
            "positive_mass": summary_stats(pos_mass),
            "selected_pair_count": summary_stats(selected_counts),
            "selected_mass": summary_stats(selected_mass),
        },
    })
    _write_meta(tmp, meta)
    verify_screen_dir(
        tmp, split_dir(a.root, "base", a.dataset, a.split),
        expected=expected, require_ok=False,
    )
    _publish_verified(tmp, out, SCREEN_SCHEMA)
    print(f"[REF SCREEN BUILT+VERIFIED] {out}")


def verify(a, cfg):
    ds = make_dataset(cfg, a.dataset, a.split, a.seq_len)
    n = len(ds)
    base_expected = _base_expected(a, cfg, n)
    screen_expected = _screen_expected(a, cfg, n)
    verify_base_full(
        a.root, a.dataset, a.split,
        expected=base_expected, chunk_rows=max(1, int(a.batch)),
    )
    verify_screen_full(
        a.root, a.dataset, a.split,
        expected=screen_expected, base_expected=base_expected,
    )
    print(f"[REF VERIFY PASS] {a.dataset} {a.split}")


def main():
    a = parse_args()
    cfg = yaml.safe_load(Path(a.config_yaml).read_text(encoding="utf-8"))
    cfg.setdefault("padding_patch", "start")
    cfg.setdefault("eps", 1e-5)
    if bool(cfg.get("affine", False)) or bool(cfg.get("subtract_last", False)) or bool(cfg.get("decomposition", False)):
        raise RuntimeError("DReSTG Reference requires affine=False, subtract_last=False, decomposition=False")
    if "softk_mass_ratio" not in cfg:
        raise RuntimeError("DReSTG config missing frozen softk_mass_ratio")
    if a.action == "verify":
        verify(a, cfg)
        return
    build_base(a, cfg)
    build_screen(a, cfg)
    print(f"[REF BUILD PASS] {a.dataset} {a.split}")


if __name__ == "__main__":
    main()
