#!/bin/bash
# The released agent with the repair search on every curved suite, then Gmsh on the same domains
# (four configurations at the agent's element count, scored on the arc tangents and on chords).
#   utilities/run_surgery_suites.sh <outdir>
# Without geogen, point GEOGEN_PATH at tools/geo2d_lite (its domains differ from geo2d's).
set -u
OUT=${1:-out/surgery/final}
PY=${PY:-venv/bin/python}
REL_CFG=models/paper-2026-09-24/pinwheel-mitq-e2e-v1.config.yml
REL=models/paper-2026-09-24/pinwheel-mitq-e2e-v1-4M.zip
mkdir -p "$OUT"
SUITES="curved:48 curved-holes:48 curved-transfer:48 curved-holes-transfer:16"
for entry in $SUITES; do
  s=${entry%%:*}; n=${entry##*:}
  $PY utilities/curved_surgery_eval.py -config $REL_CFG -checkpoint $REL -suite $s -n $n -workers 4 \
      -rounds 8 -polish 2 -rescue 8 -time_limit 600 -save_meshes "$OUT/meshes/$s" \
      -out "$OUT/released.$s.json" > "$OUT/released.$s.log" 2>&1
done
for entry in $SUITES; do
  s=${entry%%:*}
  $PY utilities/gmsh_curved_regularity.py -suite $s -agent "$OUT/released.$s.json" -key surgery \
      -variants blossom,frontal-quad,quasi-structured,blossom-full -scales 1 \
      -out "$OUT/gmsh.$s.json" > "$OUT/gmsh.$s.log" 2>&1
done
echo ALLDONE > "$OUT/done"
