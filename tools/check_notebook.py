import json, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
nb_path, drive, work, figdir = map(Path, sys.argv[1:5])
figdir.mkdir(exist_ok=True)
shown_figs = iter(range(100))
plt.show = lambda: plt.savefig(figdir / f'fig{next(shown_figs)}.png', dpi=70) or plt.close('all')
rng = np.random.default_rng(1)

def fake_detections(rows):
    out = []
    for r in rows.itertuples():
        for cx, cy, w, h in g['read_stones'](r.labels[0]):
            conf = rng.uniform(0.3, 0.95) if rng.random() > 0.15 else rng.uniform(0.001, 0.25)
            out.append({'uid': r.uid, 'conf': conf, 'cx': cx + rng.uniform(-.4, .4) * w, 'cy': cy, 'w': w, 'h': h})
            if rng.random() < 0.3:
                out.append({'uid': r.uid, 'conf': conf * 0.5, 'cx': cx, 'cy': cy + 0.3 * h, 'w': w, 'h': h})
        for _ in range(rng.poisson(1.0)):
            out.append({'uid': r.uid, 'conf': rng.uniform(0.001, 0.6), 'cx': rng.random(), 'cy': rng.random(), 'w': .03, 'h': .03})
    return pd.DataFrame(out, columns=['uid', 'conf', 'cx', 'cy', 'w', 'h'])

g = {'display': lambda x: print(x.to_string() if hasattr(x, 'to_string') else x),
     'model': None, 'detections': fake_detections, 'map50': lambda: 0.5}

cells = json.load(open(nb_path, encoding='utf-8'))['cells']
for i, c in enumerate(cells):
    if c['cell_type'] != 'code' or 'colab' in c['metadata'].get('tags', []):
        continue
    src = ''.join(c['source'])
    src = src.replace("Path('/content/drive/MyDrive/kidney_capstone')", f'Path({str(drive)!r})').replace("Path('/content/work')", f'Path({str(work)!r})')
    if 'def match' in src:
        defs, rest = src.split('val_stones, val_dets = match', 1)
        exec(compile(defs, f'cell{i}-defs', 'exec'), g)
        s = pd.DataFrame([{'uid': 'u', 'session': 0, 'cx': .5, 'cy': .5, 'w': .1, 'h': .1},
                          {'uid': 'u', 'session': 0, 'cx': .2, 'cy': .2, 'w': .02, 'h': .02},
                          {'uid': 'u', 'session': 0, 'cx': .8, 'cy': .8, 'w': .1, 'h': .1}])
        d = pd.DataFrame([{'uid': 'u', 'conf': .9, 'cx': .52, 'cy': .5, 'w': .1, 'h': .1},   # centre inside stone 0
                          {'uid': 'u', 'conf': .6, 'cx': .48, 'cy': .52, 'w': .1, 'h': .1},  # second box on stone 0
                          {'uid': 'u', 'conf': .7, 'cx': .2125, 'cy': .2, 'w': .02, 'h': .02},  # centre outside stone 1, IoU 0.6
                          {'uid': 'u', 'conf': .8, 'cx': .2, 'cy': .5, 'w': .05, 'h': .05},  # on nothing
                          {'uid': 'v', 'conf': .5, 'cx': .8, 'cy': .8, 'w': .1, 'h': .1}])   # other slice
        ms, md = g['match'](s, d)
        assert ms.best_conf.tolist() == [.9, .7, 0.0], ms.best_conf.tolist()
        kinds = dict(zip(md.conf, md.kind))
        assert kinds == {.9: 'finds a stone', .8: 'false alarm', .7: 'finds a stone', .6: 'duplicate', .5: 'false alarm'}, kinds
        m = g['at'](ms, md, 2, .65)
        assert (round(m['stone_sensitivity'], 3), m['boxes_per_slice'], m['false_alarms_per_slice'], m['precision']) == (0.667, 1.5, 0.5, 2 / 3), m
        assert g['red_threshold'](md, 0.6) == .7, g['red_threshold'](md, 0.6)
        assert g['show_threshold'](pd.DataFrame({'best_conf': [.9, .8, .7, .6, .5, .4, .3, .2, .1, 0]}), .9) == .1
        print('match/at/threshold unit checks passed')
    print(f'\n===== cell {i} =====')
    exec(compile(src, f'cell{i}', 'exec'), g)

assert g['RED'] >= g['SHOW']
print('\nharness OK  SHOW', round(g['SHOW'], 3), 'RED', round(g['RED'], 3))
