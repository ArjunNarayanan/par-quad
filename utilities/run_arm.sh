#!/usr/bin/env bash
# One experiment arm, end to end: instances, cloning, PPO, scoring.
#
#   utilities/run_arm.sh <arm> [num_envs] [seed]
#
# Arms form a ladder. Each adds ONE thing to the one above it, so a difference
# can be attributed instead of guessed at. reference-v1 is the baseline and has
# already been run: 129.0 of 135 over four passes, straight-holes 18.75/24.
#
#   reference   the published recipe                        (baseline, done)
#   gate        + the evaluator's gate in the env AND the generator
#   anchor      + the certified anchor drawn fresh, not from a fixed pickle
#   pinwheel    + the pinwheel seed family
#
# Run them SIDE BY SIDE, not in sequence. Each uses about six cores well, so a
# 32-vCPU box holds four at once and the whole ladder finishes in one run's
# wall time. Sequentially it is four times the wall clock for the same answer.
set -uo pipefail
cd "$(dirname "$0")/.."

ARM=${1:?usage: run_arm.sh <reference|gate|anchor|pinwheel|pinwheel-wide|quality|transformer|depth1> [num_envs] [seed]}
ENVS=${2:-6}
SEED=${3:-0}

# DATASET is the instance file, and two arms that would generate the same data
# share one. `gate` and `anchor` differ only in where PPO's anchor comes from,
# so cloning them from the identical pickle is both half an hour cheaper and a
# cleaner comparison -- any difference between them is then the anchor alone.
case "$ARM" in
  reference) R=experiments/self-play/quad/reference-v1
             DATASET=plain
             GEN_FLAGS="-hole_probability 0.30"
             PPO_FLAGS="-demos data/instances_arm_plain.pkl" ;;
  gate)      R=experiments/self-play/quad/gate-v1
             DATASET=gated
             GEN_FLAGS="-hole_probability 0.30 -gate untangle"
             PPO_FLAGS="-demos data/instances_arm_gated.pkl" ;;
  anchor)    R=experiments/self-play/quad/anchor-v1
             DATASET=gated
             GEN_FLAGS="-hole_probability 0.30 -gate untangle"
             PPO_FLAGS="-demo_generator -demo_hole_probability 0.30 -demo_gate untangle" ;;
  transformer) R=experiments/self-play/quad/ablate-transformer
             DATASET=pinwheel
             GEN_FLAGS="-hole_probability 0.30 -pinwheel_probability 0.40 -gate untangle"
             PPO_FLAGS="-demo_generator -demo_hole_probability 0.30 \
                        -demo_pinwheel_probability 0.40 -demo_gate untangle" ;;
  depth1)    R=experiments/self-play/quad/ablate-depth1
             DATASET=pinwheel
             GEN_FLAGS="-hole_probability 0.30 -pinwheel_probability 0.40 -gate untangle"
             PPO_FLAGS="-demo_generator -demo_hole_probability 0.30 \
                        -demo_pinwheel_probability 0.40 -demo_gate untangle" ;;
  quality)   R=experiments/self-play/quad/quality-v1
             DATASET=pinwheel
             GEN_FLAGS="-hole_probability 0.30 -pinwheel_probability 0.40 -gate untangle"
             PPO_FLAGS="-demo_generator -demo_hole_probability 0.30 \
                        -demo_pinwheel_probability 0.40 -demo_gate untangle" ;;
  pinwheel)  R=experiments/self-play/quad/pinwheel-v1
             DATASET=pinwheel
             GEN_FLAGS="-hole_probability 0.30 -pinwheel_probability 0.40 -gate untangle"
             PPO_FLAGS="-demo_generator -demo_hole_probability 0.30 \
                        -demo_pinwheel_probability 0.40 -demo_gate untangle" ;;
  pinwheel-wide)  # the pinwheel recipe with a 6-layer, width-192 extractor (capacity test)
             R=experiments/self-play/quad/pinwheel-wide-v1
             DATASET=pinwheel
             GEN_FLAGS="-hole_probability 0.30 -pinwheel_probability 0.40 -gate untangle"
             PPO_FLAGS="-demo_generator -demo_hole_probability 0.30 \
                        -demo_pinwheel_probability 0.40 -demo_gate untangle" ;;
  *) echo "unknown arm: $ARM" >&2; exit 1 ;;
esac

TAG=$ARM$([ "$SEED" = 0 ] || echo "-seed$SEED")
OUT=out/$TAG
INSTANCES=data/instances_arm_$DATASET.pkl
# The config is tracked and per-ARM; the checkpoints are per-SEED. Writing both
# to $R would have two seeds of one arm overwriting each other's
# best_holdout_model.zip and warm_ppo_*.zip, and the watcher scoring a mixture.
RUN=$R$([ "$SEED" = 0 ] || echo "-seed$SEED")
mkdir -p "$OUT/scores" "$RUN"
# Each run gets its OWN copy of the arm's config, carrying `output_dir` so every
# checkpoint lands beside it. train_bc derives its output directory from the
# config's location, so two seeds sharing one config would overwrite each
# other's best_holdout_model.zip -- and the copy also leaves each run
# self-documenting once the directories are archived.
CFG=$RUN/config.yml
if [ ! -f "$CFG" ] || [ "$R/config.yml" -nt "$CFG" ]; then
  { cat "$R/config.yml"; echo; echo "output_dir: $RUN"; } > "$CFG"
fi

die () { echo "FAILED: $* -- see $RUN and $OUT" | tee -a "$OUT/progress.log" >&2; exit 1; }
PY=venv/bin/python
[ -x "$PY" ] || PY=python3
# A box whose `python3` is 3.8 will fail somewhere deep in numpy rather than
# here, so check before anything expensive starts. Build the venv WITH the
# right interpreter -- `python3.12 -m venv venv` -- rather than changing what
# `python` means system-wide, which apt and the image's own tooling depend on.
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null \
  || { echo "need python 3.11+, got: $("$PY" --version 2>&1). Create the venv with" \
            "python3.12 -m venv venv -- see SETUP_REMOTE.md" >&2; exit 1; }
"$PY" -c "import geo2d" 2>/dev/null || die "geo2d is not importable (pip install -e ../geogen)"
[ -f "$R/config.yml" ] || die "no config at $R/config.yml"

# The instance set is per-ARM, not per-seed: the seed varies the TRAINING, and
# sharing the demonstrations is what makes two seeds a measure of run variance
# rather than of two different datasets.
# Arms start together and two of them share a dataset, so the second must wait
# for the first rather than race it into a half-written file.
LOCK=$INSTANCES.lock
if [ ! -f "$INSTANCES" ] && ! mkdir "$LOCK" 2>/dev/null; then
  echo "=== waiting for another arm to finish $INSTANCES $(date) ===" >> "$OUT/progress.log"
  while [ -d "$LOCK" ] && [ ! -s "$INSTANCES" ]; do sleep 20; done
fi
if [ ! -f "$INSTANCES" ]; then
  echo "=== generating certified instances $(date) ===" >> "$OUT/progress.log"
  # shellcheck disable=SC2086
  "$PY" utilities/generate_instances.py -n 9000 $GEN_FLAGS -max_cells 14 \
      -seed 11 -out "$INSTANCES" >> "$RUN/instances.log" 2>&1 || die "instance generation"
  [ -s "$INSTANCES" ] || die "instance generation produced no file"
  "$PY" utilities/instance_stats.py -instances "$INSTANCES" \
      >> "$OUT/progress.log" 2>&1 || true
fi
rmdir "$LOCK" 2>/dev/null || true

echo "=== cloning $(date) ===" >> "$OUT/progress.log"
# cloning is CPU-only: its collection pass feeds CPU tensors straight into the extractor,
# which crashes with "Expected all tensors to be on the same device" on a CUDA build
# train_bc sets no thread limit, so torch takes every core it can see. One
# process alone is fine; six side by side spawn ~59 threads EACH and the box
# spends itself on OpenMP barriers -- measured load 91 on 40 cores, epochs
# running 2-3x slower than they do capped. BC_THREADS keeps each arm inside
# its share. The PPO stage sets its own threads and is unaffected.
# Cloning is also CPU-only: its collection pass feeds CPU tensors straight into
# the extractor, which crashes with "Expected all tensors to be on the same
# device" on a CUDA build.
BC_THREADS=${BC_THREADS:-6}
env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="$BC_THREADS" MKL_NUM_THREADS="$BC_THREADS" \
    OPENBLAS_NUM_THREADS="$BC_THREADS" NUMEXPR_NUM_THREADS="$BC_THREADS" \
    "$PY" workflows/train_bc.py -config "$CFG" -instances "$INSTANCES" \
    -epochs 6 -holdout_every 2 -replays 2 -seed $((SEED + 3)) \
    -out "$RUN/bc_model" > "$RUN/bc.log" 2>&1 || die "cloning"
BC=$RUN/best_holdout_model.zip
[ -f "$BC" ] || BC=$RUN/bc_model.zip
[ -f "$BC" ] || die "cloning produced no checkpoint"

score_new () {
  for f in "$BC" "$RUN"/warm_ppo_*_steps.zip; do
    [ -e "$f" ] || continue
    case "$f" in
      *bc_model.zip|*best_holdout_model.zip) label=0 ;;
      *) label=$(basename "$f" | sed -E 's/warm_ppo_([0-9]+)_steps.zip/\1/') ;;
    esac
    out=$OUT/scores/$(printf "%08d" "$label").json
    [ -f "$out" ] && continue
    nice -n 15 "$PY" utilities/score_suites.py -config "$CFG" \
        -checkpoint "$f" -seed 7 -n 24 -n_samples 4 -holdout -steps "$label" \
        -out "$out" >> "$OUT/score.log" 2>&1 || continue
    "$PY" - "$out" "$TAG" <<'PYEOF' >> "$OUT/progress.log"
import json, sys, datetime
r = json.load(open(sys.argv[1]))
s = {k: v["best_of_n"] for k, v in r["suites"].items()}
mq = r.get("holdout", {}).get("best_of_n", 0)
den = sum(v["total"] for v in r["suites"].values()) + r.get("holdout", {}).get("total", 0)
print(f"{datetime.datetime.now():%H:%M}  {sys.argv[2]:<9} {r['steps']/1e6:6.2f}M  "
      f"total {sum(s.values())+mq:3d}/{den}   "
      + "  ".join(f"{k} {v}" for k, v in s.items()) + f"  mesh-quest {mq}", flush=True)
PYEOF
  done
}

watcher () { local pid=$1
  while true; do score_new; kill -0 "$pid" 2>/dev/null || break; sleep 120; done
  score_new; echo "watcher done $(date)" >> "$OUT/progress.log"; }

echo "=== PPO $TAG from $BC into $RUN $(date) ===" >> "$OUT/progress.log"
# shellcheck disable=SC2086
"$PY" workflows/train_warm_ppo.py -config "$R/config.yml" -checkpoint "$BC" \
    -num_envs "$ENVS" -torch_threads 4 -anchor mixture $PPO_FLAGS \
    -demo_probability 0.35 -critic_warmup 30 -total_timesteps 4000000 \
    -seed "$SEED" -out "$RUN" > "$RUN/train.log" 2>&1 &
TRAIN_PID=$!
watcher "$TRAIN_PID" &
wait "$TRAIN_PID"; TRAIN_STATUS=$?
echo "=== $TAG training exited status $TRAIN_STATUS $(date) ===" >> "$OUT/progress.log"

echo "=== $TAG final score, four passes $(date) ===" >> "$OUT/progress.log"
# Not `die`: training succeeded and its checkpoints are on disk, so a failure
# here is recoverable by rerunning score_suites. But it must SAY so -- the first
# time this crashed, on a singular Newton step in the optimised boundary slide,
# the `&&` swallowed it and the log simply had no FINAL block.
if ! "$PY" utilities/score_suites.py -config "$CFG" \
    -checkpoint "$RUN/warm_ppo_model.zip" -repeats 4 -n_samples 5 -holdout \
    -steps 4000000 -out "$OUT/final_score.json" >> "$OUT/score.log" 2>&1; then
  echo "FINAL SCORING FAILED for $TAG -- see $OUT/score.log. The checkpoints are" \
       "in $RUN; rerun utilities/score_suites.py against warm_ppo_model.zip." \
       >> "$OUT/progress.log"
  exit 0
fi
"$PY" - "$OUT/final_score.json" "$TAG" <<'PYEOF' >> "$OUT/progress.log"
import json, sys
r = json.load(open(sys.argv[1]))
total = sum(v["best_of_n"] for v in r["suites"].values())
mq = r.get("holdout", {}).get("best_of_n", 0)
print(f"\nFINAL {sys.argv[2]}: {total + mq:.2f}/111   (96 geo2d + 15 Mesh Quest; pinwheel-v1 was 111.00)")
for k, v in r["suites"].items():
    print(f"  {k:<18}{v['best_of_n']:>6.2f}/{v['total']:<3} spread {v['best_of_n_spread']}"
          f"  quality {v.get('mean_quality', 0):.3f}  passes {v['passes']}")
PYEOF
