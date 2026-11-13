"""Selection-head model scale, offline: agreement with the GPT-5.5 H = 4 teacher's pick on the 1,727 held-out validation panels
(199 states, 90 tasks) at the final (20th) epoch, for every head size and training seed. Joint transformer, soft CE, Recipe 3 labels,
lr 3e-4, same data and split; only width/depth and seed change. Also: best-epoch agreement (picked on the same validation panels,
so optimistic), offline success of the head's pick on the 348 held-out test states (continuation draws) and pairwise pick agreement
on those states. Interval: Wilson 95% over panels (panels of one state are correlated, so it is a lower bound on the noise).
Writes the adjacent JSON."""
import itertools, json, math
from pathlib import Path
R = "/gpfs/scrubbed/zixianma/openwebrl-runtime/recipe3-ablations-20261104/heads-scale"
DEF = "/gpfs/scrubbed/zixianma/openwebrl-runtime/rl-actor-recollect-20261016/p16-k3-20261031/train/heads-all2/k3-best/seed0"
HEADS = {("d128-L2", 0): f"{R}/d128-L2", ("d128-L2", 1): f"{R}/d128-L2-seed1", ("d128-L2", 2): f"{R}/d128-L2-seed2",
         ("d256-L2", 0): f"{R}/d256-L2", ("d512-L2", 0): DEF, ("d512-L2", 1): f"{R}/d512-L2-seed1", ("d512-L2", 2): f"{R}/d512-L2-seed2",
         ("d2048-L4", 0): f"{R}/d2048-L4", ("d4096-L8", 0): f"{R}/d4096-L8"}


def wilson(p, n, z=1.96):
    c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [c - h, c + h]


res, picks = {}, {}
for (arch, seed), d in HEADS.items():
    s = json.loads(Path(d, "scores.json").read_text()); v = s["extra_validation"]; a = v["agreement_by_epoch"]
    picks[f"{arch}-s{seed}"] = json.loads(Path(d, "picks.json").read_text())["picks"]
    res[f"{arch}-s{seed}"] = dict(arch=arch, seed=seed, params=s["params"], dir=d, val_panels=v["panels"], epochs=len(a),
                                  final_agreement=a[-1], final_agreement_ci95=wilson(a[-1], v["panels"]), best_agreement=v["best_agreement"],
                                  best_epoch=v["best_epoch"], agreement_by_epoch=a, test_states=s["test_states"],
                                  offline_success_head=s["success"]["head"], offline_success_uniform=s["success"]["uniform"],
                                  offline_success_oracle=s["success"]["oracle"], offline_head_minus_uniform_pp=s["head_minus_uniform"])
pair = {f"{a} | {b}": sum(picks[a][t] == picks[b][t] for t in picks[a]) / len(picks[a]) for a, b in itertools.combinations(picks, 2)}
Path(__file__).with_suffix(".json").write_text(json.dumps(dict(heads=res, pairwise_pick_agreement_test=pair), indent=1) + "\n")
for k, r in res.items():
    print(k.ljust(14), f"{r['params']/1e6:7.1f}M final {100*r['final_agreement']:.1f} [{100*r['final_agreement_ci95'][0]:.1f}, {100*r['final_agreement_ci95'][1]:.1f}] best {100*r['best_agreement']:.1f} offline {r['offline_success_head']}")
