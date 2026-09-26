#!/bin/bash
# One arm of the scaling work: train, score every checkpoint on the SELECTION suites
# (in-distribution, 8-24 corners), keep the best on the scorer, and when training
# ends score that best checkpoint on the TRANSFER suites (25-50 corners) with the
# paired Gmsh line.
#   utilities/run_scale_arm.sh <arm-name> <warm-start.zip> [num_envs]
#   env: THREADS (2), SELECT (straight,polycube,curved,curved-holes), N (24),
#        TRANSFER (straight-transfer,curved-transfer,curved-holes-transfer), NT (48)
set -uo pipefail
cd "$(dirname "$0")/.."
ARM=$1; CK=$2; ENVS=${3:-6}; THREADS=${THREADS:-2}
SELECT=${SELECT:-straight,polycube,curved,curved-holes}; N=${N:-24}
TRANSFER=${TRANSFER:-straight-transfer,curved-transfer,curved-holes-transfer}; NT=${NT:-48}
R=experiments/self-play/quad/$ARM; OUT=out/$ARM; PY=venv/bin/python
mkdir -p $OUT/scores
die () { echo "FAILED: $*" | tee -a $OUT/progress.log >&2; exit 1; }
[ -f "$R/config.yml" ] || die "no config for $ARM"
[ -f "$CK" ] || die "no warm-start checkpoint $CK"

dens_for () { case "$1" in curved*) echo 45;; *) echo 0;; esac; }
score_one () {  # <checkpoint> <label> <suites> <n>
  local f=$1 label=$2 suites=$3 n=$4
  local out=$OUT/scores/$(printf "%08d" "$label")
  for suite in ${suites//,/ }; do
    [ -f "$out.$suite.json" ] && continue
    OMP_NUM_THREADS=1 nice -n 15 "$PY" utilities/score_quality_objective.py -config $R/config.yml \
        -checkpoint "$f" -n "$n" -n_samples 4 -usable_at 0.2 -steps "$label" -densify $(dens_for $suite) \
        -suites $suite -out "$out.$suite.json" 2>> $OUT/score.log \
        | grep -v LOADING >> $OUT/progress.$suite.log &
  done
  wait
}
score_new () {
  score_one "$CK" 0 "$SELECT" "$N"
  for f in $(ls -t $R/warm_ppo_*_steps.zip 2>/dev/null); do
    label=$(basename "$f" | sed -nE 's/^warm_ppo_([0-9]+)_steps\.zip$/\1/p')
    [ -n "$label" ] || continue
    score_one "$f" "$label" "$SELECT" "$N"
  done
  "$PY" utilities/select_best_checkpoint.py "$ARM" -suites "$SELECT" >> $OUT/progress.log 2>&1
}
watcher () { local pid=$1
  while true; do score_new; kill -0 "$pid" 2>/dev/null || break; sleep 180; done
  score_new; echo "watcher done $(date)" >> $OUT/progress.log
  # transfer suites on the selected checkpoint (and the warm start, paired), plus Gmsh
  best=$R/best_on_scorer.zip; [ -f $best ] || best=$CK
  label=$("$PY" -c "import json; print(json.load(open('$OUT/best.json'))['best']['steps'])" 2>/dev/null || echo 0)
  score_one "$best" "$label" "$TRANSFER" "$NT"; score_one "$CK" 0 "$TRANSFER" "$NT"
  for suite in ${TRANSFER//,/ }; do
    OMP_NUM_THREADS=1 nice -n 15 "$PY" utilities/gmsh_curved_suite.py -suite $suite \
        -agent $OUT/scores/$(printf "%08d" "$label").$suite.json -out $OUT/gmsh.$suite.json >> $OUT/gmsh.log 2>&1 &
  done; wait
  echo "transfer scored $(date)" >> $OUT/progress.log; }

echo "=== $ARM from $CK $(date) select=$SELECT transfer=$TRANSFER ===" >> $OUT/progress.log
# ANCHOR / PPO_FLAGS / STEPS let an arm keep the pinwheel recipe's certified anchor
# (ANCHOR=mixture PPO_FLAGS="-demo_generator ... -demo_probability 0.35") and a
# shorter budget; the defaults are the mixture arms' (geo2d only, 4M).
ANCHOR=${ANCHOR:-none}; PPO_FLAGS=${PPO_FLAGS:-}; STEPS=${STEPS:-4000000}; SEED=${SEED:-0}
# shellcheck disable=SC2086
nice -n 5 "$PY" workflows/train_warm_ppo.py -config $R/config.yml -checkpoint "$CK" \
    -num_envs "$ENVS" -torch_threads "$THREADS" -anchor "$ANCHOR" $PPO_FLAGS -critic_warmup 40 \
    -total_timesteps "$STEPS" -seed "$SEED" -out $R > $R/train.log 2>&1 &
TRAIN_PID=$!
watcher "$TRAIN_PID" & WATCH_PID=$!
wait "$TRAIN_PID"; STATUS=$?
echo "=== $ARM exited status $STATUS $(date) ===" >> $OUT/progress.log
wait "$WATCH_PID"
