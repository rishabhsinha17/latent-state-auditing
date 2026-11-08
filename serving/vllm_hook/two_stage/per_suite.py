"""Per-suite recall and attacked-but-safe FPR at the 90% recall target, from eval_two_stage.py predictions.

Usage: per_suite.py <chat dir> <out dir>   (reads <out dir>/pred_seed{13,0,1,2}.npz, writes <out dir>/per_suite.json)
"""
import json, sys
from pathlib import Path
import numpy as np

chat, out = Path(sys.argv[1]), Path(sys.argv[2])
g = [json.loads(l) for l in open(chat / 'suite_full.jsonl')]
lab = {json.loads(l)['case_id']: json.loads(l)['unsafe'] for l in open(chat / 'suite_full.labels.jsonl')}
att = np.array([x['attack_present'] for x in g]); uns = np.array([bool(lab.get(x['case_id'], False)) and x['attack_present'] for x in g])
suite = np.array([x['case_id'].split('_user_task')[0] for x in g])
res = {}
for n in ('S1', 'C-A', 'C-HN', 'AVG'):
    rows = []
    for s in (13, 0, 1, 2):
        P = np.load(out / f'pred_seed{s}.npz'); f = P[f'{n}@0.9']
        rows.append({su: (round(float(f[(suite == su) & uns].mean()), 3), round(float(f[(suite == su) & att & ~uns].mean()), 3)) for su in ('workspace', 'travel', 'banking', 'slack')})
    res[n] = {su: [round(float(np.mean([r[su][i] for r in rows])), 3) for i in (0, 1)] for su in rows[0]}
cnt = {su: [int(((suite == su) & uns).sum()), int(((suite == su) & att & ~uns).sum()), int(((suite == su) & ~att).sum())] for su in ('workspace', 'travel', 'banking', 'slack')}
json.dump({'counts_unsafe_attsafe_clean': cnt, 'per_suite_at_0.9': res}, open(out / 'per_suite.json', 'w'), indent=1)
print(json.dumps(res['S1']))
