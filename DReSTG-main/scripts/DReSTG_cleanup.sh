#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
IDENTIFIER="${1:?setting/model_id required}"
COMPACT_DIR="${2:?compact cell directory required}"
MODE="${3:-complete}"
CHECKPOINT_ROOT="${4:-}"
safe_identifier() {
  [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] && [[ "$1" != *".."* ]]
}
safe_identifier "$IDENTIFIER" || { echo "[CLEANUP ERROR] unsafe identifier: $IDENTIFIER" >&2; exit 2; }
case "$MODE" in complete|failed) ;; *) echo '[CLEANUP ERROR] mode=complete|failed required' >&2; exit 2 ;; esac
COMPACT_ABS="$(realpath -m -- "$COMPACT_DIR")"
case "$COMPACT_ABS" in
  "$ROOT"/drestg_results/completed_cells/*|"$ROOT"/drestg_smoke_results/*) ;;
  *) echo "[CLEANUP ERROR] compact directory outside project roots: $COMPACT_ABS" >&2; exit 2 ;;
esac
[[ -d "$COMPACT_ABS" && ! -L "$COMPACT_ABS" ]] || { echo '[CLEANUP ERROR] missing or symlink compact cell' >&2; exit 2; }
before="$(df -Pk . | awk 'NR==2{print $4}')"
if [[ "$MODE" == failed ]]; then
  [[ -f "$COMPACT_ABS/formal.log" ]] || { echo '[CLEANUP ERROR] failed mode needs formal.log' >&2; exit 3; }
  echo "[CLEANUP FAILED] preserve logs, incomplete metrics, transient results and checkpoints: $COMPACT_ABS"
  exit 0
fi
for f in metrics.json runtime.json checkpoint_path.txt formal.log; do
  [[ -f "$COMPACT_ABS/$f" && ! -L "$COMPACT_ABS/$f" ]] || { echo "[CLEANUP ERROR] mandatory artifact missing/symlink: $f" >&2; exit 3; }
done
[[ -n "$CHECKPOINT_ROOT" ]] || { echo '[CLEANUP ERROR] checkpoint root required for complete' >&2; exit 2; }
CKPT_ROOT="$(realpath -m -- "$CHECKPOINT_ROOT")"
case "$CKPT_ROOT" in "$ROOT"/checkpoints/DReSTG/*) ;; *) echo '[CLEANUP ERROR] checkpoint root outside DReSTG' >&2; exit 2 ;; esac
CKPT="$(cat "$COMPACT_ABS/checkpoint_path.txt")"
CKPT_ABS="$(realpath -m -- "$CKPT")"
EXPECTED_CKPT="$CKPT_ROOT/$IDENTIFIER/checkpoint.pth"
[[ "$CKPT_ABS" == "$EXPECTED_CKPT" && "$CKPT_ABS" == "$ROOT"/checkpoints/DReSTG/* ]] || {
  echo "[CLEANUP ERROR] checkpoint path mismatch: $CKPT_ABS expected $EXPECTED_CKPT" >&2; exit 2;
}
RESULTS_ROOT="$(realpath -m -- "$ROOT/results")"
TEST_ROOT="$(realpath -m -- "$ROOT/test_results")"
RESULT_TARGET="$(realpath -m -- "$RESULTS_ROOT/$IDENTIFIER")"
TEST_TARGET="$(realpath -m -- "$TEST_ROOT/$IDENTIFIER")"
[[ "$RESULT_TARGET" == "$RESULTS_ROOT/$IDENTIFIER" && "$TEST_TARGET" == "$TEST_ROOT/$IDENTIFIER" ]] || {
  echo '[CLEANUP ERROR] unexpected result/test path' >&2; exit 2;
}
# A verified COMPLETE can safely be cleaned again: the first cleanup has already
# removed the checkpoint and the exact result/test directories.
if [[ ! -e "$CKPT_ABS" && ! -L "$CKPT_ABS" ]]; then
  [[ -f "$COMPACT_ABS/complete.json" && ! -e "$RESULT_TARGET" && ! -L "$RESULT_TARGET" && ! -e "$TEST_TARGET" && ! -L "$TEST_TARGET" ]] || {
    echo '[CLEANUP ERROR] checkpoint absent without verified completed cleanup' >&2; exit 3;
  }
  python - "$COMPACT_ABS/complete.json" "$IDENTIFIER" "$CKPT" <<'PY_CLEANUP_VERIFY'
import json,sys
c=json.load(open(sys.argv[1]))
if c.get('schema')!='DReSTG_complete_v1' or c.get('status')!='COMPLETE' or c.get('setting')!=sys.argv[2] or c.get('checkpoint')!=sys.argv[3] or c.get('checkpoint_removed') is not True:
    raise SystemExit('cannot treat nonverified COMPLETE as cleaned')
PY_CLEANUP_VERIFY
  echo "[CLEANUP ALREADY COMPLETE] $IDENTIFIER"
  exit 0
fi
[[ -f "$CKPT_ABS" && ! -L "$CKPT_ABS" ]] || { echo '[CLEANUP ERROR] checkpoint path is not a regular file' >&2; exit 3; }
# Restricted to exact setting, after the caller has validated compact identity and metrics.
for target in "$RESULT_TARGET" "$TEST_TARGET"; do
  if [[ -e "$target" || -L "$target" ]]; then
    [[ -d "$target" && ! -L "$target" ]] || { echo "[CLEANUP ERROR] unexpected link/file: $target" >&2; exit 2; }
    rm -rf -- "$target"
  fi
done
rm -f -- "$CKPT_ABS"
rmdir -- "$(dirname "$CKPT_ABS")" 2>/dev/null || true
after="$(df -Pk . | awk 'NR==2{print $4}')"
freed_kb=$((after-before))
echo "[CLEANUP COMPLETE] $IDENTIFIER transient results/test/checkpoint removed; freed≈${freed_kb}KiB; compact metrics/runtime/log retained"
