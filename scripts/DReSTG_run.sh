#!/usr/bin/env bash
set -Eeuo pipefail

GPU="${1:-0}"
MODE="${2:-dry-run}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

positive_int_or_one() { local v="$1"; [[ "$v" =~ ^[1-9][0-9]*$ ]] && echo "$v" || echo 1; }
export OMP_NUM_THREADS="$(positive_int_or_one "${OMP_NUM_THREADS:-1}")"
export MKL_NUM_THREADS="$(positive_int_or_one "${MKL_NUM_THREADS:-1}")"
export OPENBLAS_NUM_THREADS="$(positive_int_or_one "${OPENBLAS_NUM_THREADS:-1}")"
export NUMEXPR_NUM_THREADS="$(positive_int_or_one "${NUMEXPR_NUM_THREADS:-1}")"

case "$MODE" in
  dry-run|prepare|smoke|chain2|full) ;;
  *) echo 'usage: bash scripts/DReSTG_run.sh <gpu> {dry-run|prepare|smoke|chain2|full}' >&2; exit 2 ;;
esac

REF_MODE="${DRESTG_REFERENCE_MODE:-reuse}"
REF_ROOT="${DRESTG_REFERENCE_ROOT:-reference}"
REF_DEVICE="${REF_DEVICE:-cpu}"
REF_BATCH="${REF_BATCH:-64}"
MIN_FREE_GB="${MIN_FREE_GB:-8}"
ETT_SECOND_GPU="${ETT_SECOND_GPU:-$GPU}"
PLAN="drestg_plan.tsv"
RESULT_ROOT="drestg_results"
SMOKE_ROOT="drestg_smoke_results"
SEEDS=(2021 2022 2023)

case "$REF_MODE" in
  build|reuse) ;;
  *) echo '[FAIL] reference mode must be build|reuse' >&2; exit 2 ;;
esac
[[ -n "$REF_ROOT" ]] || { echo '[FAIL] Reference root empty' >&2; exit 2; }

resolve_dataset() {
  local ds="$1"
  case "$ds" in
    ETTh1|ETTh2|ETTm1|ETTm2)
      DATA_ARG="$ds"; CFG="configs/${ds}.yaml"; ROOT_PATH="datasets/Ett_Series"; DATA_PATH="${ds}.csv" ;;
    Weather)
      DATA_ARG="custom"; CFG="configs/weather.yaml"; ROOT_PATH="datasets/Customs"; DATA_PATH="weather.csv" ;;
    Electricity)
      DATA_ARG="custom"; CFG="configs/electricity.yaml"; ROOT_PATH="datasets/Customs"; DATA_PATH="electricity.csv" ;;
    Traffic)
      DATA_ARG="custom"; CFG="configs/traffic.yaml"; ROOT_PATH="datasets/Customs"; DATA_PATH="traffic.csv" ;;
    *) echo "[FAIL] Unsupported dataset=$ds" >&2; exit 2 ;;
  esac
}

check_disk() {
  local free
  free=$(df -Pk . | awk 'NR==2{print int($4/1024/1024)}')
  (( free >= MIN_FREE_GB )) || { echo "[FAIL] disk free=${free}GiB < ${MIN_FREE_GB}GiB" >&2; exit 8; }
  echo "[DISK] free=${free}GiB minimum=${MIN_FREE_GB}GiB"
}

km_for_dataset() {
  case "$1" in
    ETTh1) echo .50 ;;
    ETTh2) echo .25 ;;
    ETTm1) echo .35 ;;
    ETTm2) echo .65 ;;
    Weather) echo .20 ;;
    Electricity) echo .01 ;;
    Traffic) echo .0037 ;;
    *) return 2 ;;
  esac
}

write_plan() {
  printf 'dataset\thorizon\tseed\treference_mode\tsoftk_mass_ratio\tmodel_id\n' > "$PLAN"
  local ds h seed km
  for seed in "${SEEDS[@]}"; do
    for ds in ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity Traffic; do
      km="$(km_for_dataset "$ds")"
      for h in 96 192 336 720; do
        printf '%s\t%s\t%s\t%s\t%s\tdrestg_formal_%s_L336_H%s_seed%s\n' \
          "$ds" "$h" "$seed" "$REF_MODE" "$km" "$ds" "$h" "$seed" >> "$PLAN"
      done
    done
  done
}

static_preflight() {
  write_plan
  python - "$PLAN" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
assert len(rows)==84, len(rows)
assert len(set(x['model_id'] for x in rows))==84
expected_seeds={'2021','2022','2023'}
for ds in ('ETTh1','ETTh2','ETTm1','ETTm2','Weather','Electricity','Traffic'):
    r=[x for x in rows if x['dataset']==ds]
    assert len(r)==12, (ds,len(r))
    for seed in expected_seeds:
        rs=[x for x in r if x['seed']==seed]
        assert sorted(int(x['horizon']) for x in rs)==[96,192,336,720], (ds,seed,rs)
print('[PREFLIGHT] unique 7×4×3-seed ledger PASS')
PY

  for f in scripts/ref.py utils/ref_cache.py utils/drestg_identity.py data_provider/ref_dataset.py scripts/DReSTG_cleanup.sh; do
    [[ -f "$f" ]] || { echo "[FAIL] missing $f" >&2; exit 1; }
  done
  [[ -r scripts/DReSTG_cleanup.sh ]] || { echo '[FAIL] cleanup not readable' >&2; exit 1; }
  grep -q 'extract_fine_patches' layers/DReSTG_backbone.py
  grep -q 'pool_coarse_patch_nodes' utils/token_utils.py
  grep -q 'OneHopMixHopPropagation' layers/GraphPropagation.py
  grep -q 'GlobalGraphModule' layers/DualResolutionGraph.py

  python - <<'PY'
import yaml
for name in ('ETTh1','ETTh2','ETTm1','ETTm2','weather','electricity','traffic'):
    y=yaml.safe_load(open('configs/'+name+'.yaml'))
    assert y['coarse_patch_count']==7 and y['fine_patch_len']==2*y['fine_patch_stride']
    assert y['gnn_r_degree']==1 and y['mixhop_depth']==1
    assert y['mixhop_alpha']==.05 and not y['affine']
t=yaml.safe_load(open('configs/traffic.yaml'))
expected={'enc_in':862,'e_layers':3,'n_heads':16,'d_model':128,'d_ff':256,
          'fine_patch_len':16,'fine_patch_stride':8,'batch_size':8,'learning_rate':0.0001,
          'dropout':0.2,'fc_dropout':0.2,'head_dropout':0.0,'train_epochs':100,
          'patience':10,'lradj':'TST','pct_start':0.2,'softk_mass_ratio':0.0037}
for k,v in expected.items():
    assert t.get(k)==v,(k,t.get(k),v)
assert (336//t['coarse_patch_count'])==48
assert (48-t['fine_patch_len'])%t['fine_patch_stride']==0
print('[PREFLIGHT] frozen dataset YAML settings PASS')
PY

  local tmp
  tmp="$(mktemp /tmp/drestg_recipe.XXXXXXXX.json)"
  python -u run.py --config_yaml configs/ETTh1.yaml \
    --is_training 1 --model_id drestg_dryrun_ETTh1_H96_seed2021 --model DReSTG --data ETTh1 \
    --seq_len 336 --pred_len 96 --random_seed 2021 \
    --drestg_dataset_id ETTh1 --drestg_reference_root "$REF_ROOT" \
    --drestg_reference_mode "$REF_MODE" --lambda_init 0.1 --drestg_recipe_out "$tmp" >/dev/null
  [[ -s "$tmp" ]] || { echo '[FAIL] recipe-only manifest' >&2; exit 1; }
  rm -f -- "$tmp"
  echo '[PREFLIGHT] DReSTG model contract PASS'
}

prepare_assets() {
  local ds="$1" prep_mode="${2:-$REF_MODE}" action split
  resolve_dataset "$ds"
  [[ -s "$ROOT_PATH/$DATA_PATH" ]] || { echo "[FAIL] dataset missing: $ROOT_PATH/$DATA_PATH" >&2; exit 3; }
  case "$prep_mode" in
    build) action=build ;;
    reuse) action=verify ;;
    *) exit 2 ;;
  esac
  for split in train val test; do
    [[ "$prep_mode" == build ]] && check_disk
    python -u scripts/ref.py --action "$action" --config_yaml "$CFG" --dataset "$ds" --split "$split" \
      --root "$REF_ROOT" --device "$REF_DEVICE" --batch "$REF_BATCH"
    [[ "$prep_mode" == build ]] && check_disk
  done
  echo "[REFERENCE READY] dataset=$ds mode=$prep_mode root=$REF_ROOT"
}

prepare_all() {
  local ds
  for ds in ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity Traffic; do
    prepare_assets "$ds" "$REF_MODE"
  done
  echo '[REFERENCE PASS] 7 datasets × 3 splits'
}

if [[ "$MODE" == dry-run ]]; then
  static_preflight
  echo '[DRY-RUN PASS] 84 unique cells'
  exit 0
fi
if [[ "$MODE" == prepare ]]; then
  prepare_all
  exit 0
fi

check_disk

build_cmd() {
  local ds="$1" h="$2" seed="$3" model_id="$4" checkpoint_root="$5" gpu_id="$6"
  resolve_dataset "$ds"
  CMD=(python -u run.py
    --config_yaml "$CFG" --is_training 1 --model_id "$model_id" --model DReSTG
    --data "$DATA_ARG" --root_path "$ROOT_PATH" --data_path "$DATA_PATH"
    --seq_len 336 --pred_len "$h" --random_seed "$seed" --gpu "$gpu_id"
    --checkpoints "$checkpoint_root" --drestg_dataset_id "$ds"
    --drestg_reference_root "$REF_ROOT" --drestg_reference_mode "$REF_MODE"
    --theta_s_init 0.5413248546 --boundary_eps 0.001 --lambda_init 0.1
    --mixhop_depth 1 --mixhop_alpha 0.05)
}

write_manifest() {
  local dst="$1"
  local -a call=("${CMD[@]}" --drestg_recipe_out "$dst")
  "${call[@]}" >/dev/null
  [[ -s "$dst" ]] || { echo '[FAIL] manifest absent' >&2; return 1; }
}

find_result() {
  local model_id="$1"
  mapfile -t MATCH < <(find results -maxdepth 1 -mindepth 1 -type d -name "${model_id}_*" -print | sort)
  [[ ${#MATCH[@]} -eq 1 ]] || {
    echo "[FAIL] expected one result dir: $model_id found ${#MATCH[@]}" >&2
    return 4
  }
  RESULT_DIR="${MATCH[0]}"
  SETTING="$(basename "$RESULT_DIR")"
}

validate_compact() {
  local cell="$1" manifest="$2" ckpt
  for f in metrics.json runtime.json checkpoint_path.txt formal.log; do
    [[ -f "$cell/$f" ]] || return 2
  done
  ckpt="$(cat "$cell/checkpoint_path.txt")"
  python - "$manifest" "$cell/metrics.json" "$cell/runtime.json" "$ckpt" <<'PY'
import json,math,os,sys
from utils.drestg_identity import relation_settlement
current,mfile,rfile,ckpt=sys.argv[1:]
cur=json.load(open(current))
m=json.load(open(mfile))
r=json.load(open(rfile))
if m.get('schema')!='DReSTG_metrics_v1' or m.get('model')!='DReSTG':
    raise SystemExit('metrics contract')
for o,label in ((m,'metrics'),(r,'runtime')):
    if o.get('cell_identity')!=cur.get('cell_identity') or o.get('recipe')!=cur.get('recipe'):
        raise SystemExit(label+' identity/recipe mismatch')
if m['cell_identity'].get('relation_settlement')!=relation_settlement():
    raise SystemExit('relation contract')
for k in ('mse','mae','rmse','rse'):
    if not math.isfinite(float(m.get(k,float('nan')))):
        raise SystemExit('nonfinite metric '+k)
if r.get('schema')!='DReSTG_runtime_v1' or not math.isfinite(float(r.get('elapsed_wall_seconds',float('nan')))):
    raise SystemExit('runtime contract')
if not os.path.isfile(ckpt):
    raise SystemExit('checkpoint absent')
PY
}

validate_complete() {
  local cell="$1" manifest="$2" ckpt
  for f in complete.json metrics.json runtime.json checkpoint_path.txt formal.log; do
    [[ -f "$cell/$f" ]] || return 1
  done
  ckpt="$(cat "$cell/checkpoint_path.txt")"
  python - "$manifest" "$cell/metrics.json" "$cell/runtime.json" "$cell/complete.json" "$ckpt" <<'PY'
import json,math,os,sys
cur=json.load(open(sys.argv[1]))
m=json.load(open(sys.argv[2]))
r=json.load(open(sys.argv[3]))
c=json.load(open(sys.argv[4]))
ckpt=sys.argv[5]
for o,label in ((m,'metrics'),(r,'runtime'),(c,'complete')):
    if o.get('cell_identity')!=cur.get('cell_identity') or o.get('recipe')!=cur.get('recipe'):
        raise SystemExit(label+' identity/recipe mismatch')
if c.get('schema')!='DReSTG_complete_v1' or c.get('status')!='COMPLETE':
    raise SystemExit('COMPLETE schema/status')
if c.get('checkpoint')!=ckpt or c.get('checkpoint_removed') is not True:
    raise SystemExit('COMPLETE checkpoint contract')
if os.path.isfile(ckpt):
    raise SystemExit('checkpoint should have been removed')
for k in ('mse','mae','rmse','rse'):
    if not math.isfinite(float(m.get(k,float('nan')))):
        raise SystemExit('nonfinite metric '+k)
if r.get('schema')!='DReSTG_runtime_v1' or not math.isfinite(float(r.get('elapsed_wall_seconds',float('nan')))):
    raise SystemExit('runtime contract')
PY
}

run_cell() {
  local ds="$1" h="$2" seed="$3" epochs="${4:-100}" lifecycle="${5:-formal}" gpu_id="${6:-$GPU}"
  local out cell checkpoint_root log start end rc metrics ckpt manifest model_id cleanup_rc

  model_id="drestg_${lifecycle}_${ds}_L336_H${h}_seed${seed}"
  checkpoint_root="checkpoints/DReSTG/${lifecycle}/${ds}/seed${seed}"
  build_cmd "$ds" "$h" "$seed" "$model_id" "$checkpoint_root" "$gpu_id"
  [[ "$epochs" == 100 ]] || CMD+=(--train_epochs "$epochs" --patience 1)

  if [[ "$lifecycle" == formal ]]; then
    out="$RESULT_ROOT"
    cell="$out/completed_cells/drestg_${ds}_L336_H${h}_seed${seed}"
  else
    out="$SMOKE_ROOT"
    cell="$out/drestg_${lifecycle}_${ds}_L336_H${h}_seed${seed}"
  fi

  mkdir -p "$cell"
  manifest="$cell/current_manifest.tmp.json"
  write_manifest "$manifest"

  if [[ -f "$cell/complete.json" ]]; then
    if validate_complete "$cell" "$manifest"; then
      rm -f -- "$manifest"
      echo "[SKIP COMPLETE] $cell"
      return 0
    fi
    echo "[FAIL] invalid old COMPLETE: $cell" >&2
    rm -f -- "$manifest"
    return 6
  fi

  if [[ -f "$cell/formal.log" ]]; then
    echo "[FAIL] existing unfinished cell; investigate manually: $cell" >&2
    rm -f -- "$manifest"
    return 9
  fi

  log="$cell/formal.log"
  check_disk
  start=$(date +%s)
  set +e
  "${CMD[@]}" >"$log" 2>&1
  rc=$?
  set -e
  end=$(date +%s)

  if (( rc != 0 )); then
    printf 'cell=%s\nexit=%s\nlog=%s\n' "$model_id" "$rc" "$log" > "$cell/fail.txt"
    set +e
    bash scripts/DReSTG_cleanup.sh "$model_id" "$cell" failed "$checkpoint_root"
    cleanup_rc=$?
    set -e
    rm -f -- "$manifest"
    if (( cleanup_rc != 0 )); then
      echo "[FAIL] failed-run cleanup handler rc=$cleanup_rc for $model_id" >&2
      return "$cleanup_rc"
    fi
    echo "[FAIL] $model_id rc=$rc" >&2
    return "$rc"
  fi

  find_result "$model_id" || return $?
  metrics="$RESULT_DIR/metrics.json"
  [[ -f "$metrics" ]] || { echo "[FAIL] metrics absent: $metrics" >&2; return 7; }
  ckpt="$checkpoint_root/$SETTING/checkpoint.pth"
  [[ -f "$ckpt" ]] || { echo "[FAIL] checkpoint absent: $ckpt" >&2; return 7; }

  cp -a -- "$metrics" "$cell/metrics.json"
  printf '%s\n' "$ckpt" > "$cell/checkpoint_path.txt"
  python - "$cell/metrics.json" "$cell/runtime.json" "$((end-start))" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
with open(sys.argv[2],'w') as f:
    json.dump({'schema':'DReSTG_runtime_v1',
               'elapsed_wall_seconds':int(sys.argv[3]),
               'cell_identity':m['cell_identity'],
               'recipe':m['recipe']},f,indent=2,sort_keys=True)
PY

  validate_compact "$cell" "$manifest" || {
    echo '[FAIL] compact verification; preserve transient' >&2
    return 7
  }

  set +e
  bash scripts/DReSTG_cleanup.sh "$SETTING" "$cell" complete "$checkpoint_root"
  cleanup_rc=$?
  set -e
  if (( cleanup_rc != 0 )); then
    echo "[FAIL] cleanup failed for $model_id rc=$cleanup_rc; COMPLETE not written" >&2
    return "$cleanup_rc"
  fi

  python - "$cell/metrics.json" "$cell/complete.json.tmp" "$SETTING" "$ckpt" <<'PY'
import datetime,json,sys
m=json.load(open(sys.argv[1]))
with open(sys.argv[2],'w') as f:
    json.dump({'schema':'DReSTG_complete_v1',
               'status':'COMPLETE',
               'cell_identity':m['cell_identity'],
               'recipe':m['recipe'],
               'setting':sys.argv[3],
               'checkpoint':sys.argv[4],
               'checkpoint_removed':True,
               'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat()},
              f,indent=2,sort_keys=True)
PY
  mv "$cell/complete.json.tmp" "$cell/complete.json"

  validate_complete "$cell" "$manifest" || {
    echo '[FAIL] COMPLETE validation' >&2
    return 7
  }
  rm -f -- "$manifest"
  check_disk
  echo "[COMPLETE] $ds H=$h seed=$seed lifecycle=$lifecycle"
}

write_summary() {
  mkdir -p "$RESULT_ROOT"
  python - "$RESULT_ROOT/completed_cells" "$RESULT_ROOT/summary.tsv" <<'PY'
import glob,json,os,sys
root,out=sys.argv[1:]
rows=[]
for path in sorted(glob.glob(os.path.join(root,'*','complete.json'))):
    cell=os.path.dirname(path)
    c=json.load(open(path))
    m=json.load(open(os.path.join(cell,'metrics.json')))
    if c.get('status')!='COMPLETE' or c.get('cell_identity')!=m.get('cell_identity') or c.get('recipe')!=m.get('recipe'):
        raise SystemExit('invalid COMPLETE '+path)
    i=m['cell_identity']
    rows.append((i['dataset'],int(i['horizon']),int(i['seed']),m['mse'],m['mae'],m['rmse'],m['rse'],i['model_id']))
expected={(ds,h,s) for ds in ('ETTh1','ETTh2','ETTm1','ETTm2','Weather','Electricity','Traffic')
                    for h in (96,192,336,720) for s in (2021,2022,2023)}
actual={(r[0],r[1],r[2]) for r in rows}
if len(rows)!=84 or actual!=expected:
    raise SystemExit('expected exact 84 COMPLETE; got rows=%d unique=%d missing=%d extra=%d' %
                     (len(rows),len(actual),len(expected-actual),len(actual-expected)))
rows.sort(key=lambda r: (r[0],r[1],r[2]))
with open(out+'.tmp','w') as f:
    f.write('dataset\thorizon\tseed\tmse\tmae\trmse\trse\tmodel_id\n')
    for row in rows:
        f.write('\t'.join(map(str,row))+'\n')
os.replace(out+'.tmp',out)
print('[SUMMARY] 84 cells',out)
PY
}

case "$MODE" in
  smoke)
    prepare_assets Traffic "$REF_MODE"
    run_cell Traffic 96 2021 1 smoke "$GPU"
    ;;

  chain2)
    prepare_assets ETTh1 "$REF_MODE"
    (run_cell ETTh1 96 2021 1 chain2 "$GPU") & pid1=$!
    (run_cell ETTh1 192 2021 1 chain2 "$ETT_SECOND_GPU") & pid2=$!
    rc1=0; rc2=0
    wait "$pid1" || rc1=$?
    wait "$pid2" || rc2=$?
    if (( rc1 != 0 || rc2 != 0 )); then
      echo "[CHAIN2 FAIL] H96=$rc1 H192=$rc2" >&2
      exit 1
    fi
    ;;

  full)
    prepare_all

    for seed in "${SEEDS[@]}"; do
      for ds in ETTh1 ETTh2 ETTm1 ETTm2; do
        for h in 96 336; do
          h2=$((h == 96 ? 192 : 720))
          (run_cell "$ds" "$h" "$seed" 100 formal "$GPU") & pid1=$!
          (run_cell "$ds" "$h2" "$seed" 100 formal "$ETT_SECOND_GPU") & pid2=$!
          rc1=0; rc2=0
          wait "$pid1" || rc1=$?
          wait "$pid2" || rc2=$?
          if (( rc1 != 0 || rc2 != 0 )); then
            echo "[PAIR FAIL] seed=$seed $ds H${h}=$rc1 H${h2}=$rc2" >&2
            exit 1
          fi
        done
      done
    done

    for seed in "${SEEDS[@]}"; do
      for ds in Weather Electricity Traffic; do
        for h in 96 192 336 720; do
          run_cell "$ds" "$h" "$seed" 100 formal "$GPU"
        done
      done
    done

    write_summary
    echo '[FULL PASS] 84/84 DReSTG cells COMPLETE'
    ;;
esac
