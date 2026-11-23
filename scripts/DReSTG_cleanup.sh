#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

IDENTIFIER="${1:?setting/model_id required}"
COMPACT_DIR="${2:?compact cell directory required}"
MODE="${3:-complete}"
CHECKPOINT_ROOT="${4:-}"

safe_identifier() {
  [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] && [[ "$1" != *"/"* ]] && [[ "$1" != *"\\"* ]] && [[ "$1" != *".."* ]]
}

safe_identifier "$IDENTIFIER" || { echo "[CLEANUP-ERROR] unsafe identifier: $IDENTIFIER" >&2; exit 2; }
case "$MODE" in
  complete|failed) ;;
  *) echo '[CLEANUP-ERROR] mode must be complete|failed' >&2; exit 2 ;;
esac

COMPACT_ABS="$(realpath -m -- "$COMPACT_DIR")"
case "$COMPACT_ABS" in
  "$ROOT"/drestg_results/completed_cells/*|"$ROOT"/drestg_smoke_results/*) ;;
  *) echo "[CLEANUP-ERROR] compact directory outside DReSTG roots: $COMPACT_ABS" >&2; exit 2 ;;
esac

if [[ "$MODE" == complete && -f "$COMPACT_ABS/complete.json" ]]; then
  echo "[CLEANUP ALREADY COMPLETE] $COMPACT_ABS"
  exit 0
fi

RESULTS_ROOT="$(realpath -m -- "$ROOT/results")"
TEST_ROOT="$(realpath -m -- "$ROOT/test_results")"

if [[ "$MODE" == complete ]]; then
  for f in metrics.json runtime.json checkpoint_path.txt formal.log; do
    [[ -f "$COMPACT_ABS/$f" ]] || { echo "[CLEANUP-ERROR] mandatory artifact missing: $f" >&2; exit 3; }
  done

  CKPT="$(cat "$COMPACT_ABS/checkpoint_path.txt")"
  CKPT_ABS="$(realpath -m -- "$CKPT")"
  [[ -f "$CKPT_ABS" ]] || { echo "[CLEANUP-ERROR] checkpoint missing: $CKPT_ABS" >&2; exit 3; }

  if [[ -n "$CHECKPOINT_ROOT" ]]; then
    CHECKPOINT_ROOT_ABS="$(realpath -m -- "$CHECKPOINT_ROOT")"
    case "$CKPT_ABS" in
      "$CHECKPOINT_ROOT_ABS"/*) ;;
      *) echo '[CLEANUP-ERROR] checkpoint outside declared checkpoint root' >&2; exit 2 ;;
    esac
  else
    case "$CKPT_ABS" in
      "$ROOT"/checkpoints/*) ;;
      *) echo '[CLEANUP-ERROR] checkpoint outside repository checkpoint root' >&2; exit 2 ;;
    esac
  fi

  RESULT_TARGET="$(realpath -m -- "$RESULTS_ROOT/$IDENTIFIER")"
  TEST_TARGET="$(realpath -m -- "$TEST_ROOT/$IDENTIFIER")"
  case "$RESULT_TARGET" in "$RESULTS_ROOT"/*) ;; *) echo '[CLEANUP-ERROR] unsafe results target' >&2; exit 2 ;; esac
  case "$TEST_TARGET" in "$TEST_ROOT"/*) ;; *) echo '[CLEANUP-ERROR] unsafe test target' >&2; exit 2 ;; esac

  [[ -d "$RESULT_TARGET" ]] && rm -rf -- "$RESULT_TARGET"
  [[ -d "$TEST_TARGET" ]] && rm -rf -- "$TEST_TARGET"
  rm -f -- "$CKPT_ABS"
  rmdir --ignore-fail-on-non-empty -- "$(dirname "$CKPT_ABS")" 2>/dev/null || true

  echo '[CLEANUP] complete: transient result/test directories and checkpoint removed; compact artifacts preserved'
  exit 0
fi

[[ -f "$COMPACT_ABS/formal.log" ]] || { echo '[CLEANUP-ERROR] failed mode requires formal.log evidence' >&2; exit 3; }
echo '[CLEANUP] failed: NO DELETION; formal.log, partial results and checkpoints preserved for recovery'
