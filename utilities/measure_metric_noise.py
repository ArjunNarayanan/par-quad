"""Score ONE checkpoint several times with different rollout randomness.

Every comparison made tonight rests on a single best-of-5 evaluation of a single
checkpoint. A hole-free domain flipped between two of them, which the slit rule
cannot explain, so the spread of the metric itself needs measuring before any of
those differences are believed.
"""
import sys, os, glob; sys.path.append(os.getcwd())
import numpy as np, torch
from src.utils import load_yaml_config
from src.geo2d_bridge import make_geo2d_env, env_summary, load_model
from utilities.evaluate_geo2d import rollout
from utilities.score_suites import SUITES, geometries
from src.holdout import LEVEL_NAMES
from utilities.evaluate_holdout import evaluate_holdout

CFG = "experiments/self-play/quad/geo2d-long-v1/config.yml"
ec = load_yaml_config(CFG)["environment"]
ckpt = sorted(glob.glob("experiments/self-play/quad/geo2d-long-v1/warm_ppo_*_steps.zip"))[-1]
print("checkpoint:", os.path.basename(ckpt), flush=True)
model = load_model(ckpt, CFG, template_size=160)

geoms = {name: list(geometries(name, 7, 24, 24)) for name in SUITES}
totals, per_suite = [], {name: [] for name in SUITES}
for trial in range(6):
    torch.manual_seed(1000 + trial); np.random.seed(1000 + trial)
    total = 0
    for name, gs in geoms.items():
        hit = 0
        for g in gs:
            env = make_geo2d_env(ec, g, template_size=160); best = None
            for k in range(5):
                rollout(model, env, deterministic=(k == 0)); s = env_summary(env)
                key = (not s["solved"], s["face_score"], abs(s["vertex_excess"]))
                if best is None or key < best: best = key
            hit += int(best[0] is False)
        per_suite[name].append(hit); total += hit
    _, summary = evaluate_holdout(model, ec, levels=LEVEL_NAMES, n_samples=4, verbose=False)
    total += summary["bestN"]
    totals.append(total)
    print(f"  trial {trial + 1}: total {total}/135   "
          + "  ".join(f"{n} {per_suite[n][-1]}" for n in SUITES)
          + f"  mesh-quest {summary['bestN']}", flush=True)

a = np.array(totals)
print(f"\nsame checkpoint, same domains, six evaluations:")
print(f"  totals {sorted(a)}   mean {a.mean():.1f}   sd {a.std(ddof=1):.1f}   spread {a.max()-a.min()}")
for n in SUITES:
    v = np.array(per_suite[n])
    print(f"  {n:16s} mean {v.mean():5.1f}  sd {v.std(ddof=1):.1f}  spread {v.max()-v.min()}")
