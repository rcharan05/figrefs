"""Print unmatched GT / predicted (figure, paragraph) pairs for one paper."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from run_eval import run_ours, run_grobid, toks, f1
from latex_gt import ground_truth
d = sys.argv[1]; system = sys.argv[2] if len(sys.argv) > 2 else 'ours'
gt = ground_truth(os.path.join(d, 'src'))
pred = run_ours(os.path.join(d, 'paper.pdf')) if system == 'ours' else run_grobid(os.path.join(d, 'paper.pdf'))
P = [(f, t, toks(t)) for f, t in pred['mentions']]
print('### GT misses')
for m in gt['mentions']:
    g = toks(m['paragraph'])
    best = max([(f1(g, pt), t) for f, t, pt in P if f == m['figure']] or [(0, '')])
    any_fig = max([(f1(g, pt), f) for f, t, pt in P] or [(0, None)])
    if best[0] < 0.5:
        print(f"fig {m['figure']} best={best[0]:.2f} (best any-fig {any_fig[0]:.2f} fig {any_fig[1]})\n  GT: {m['paragraph'][:300]}\n  PR: {best[1][:300]}")
print('### false positives')
for f, t, pt in P:
    best = max([f1(toks(m['paragraph']), pt) for m in gt['mentions'] if m['figure'] == f] or [0])
    if best < 0.5:
        print(f"fig {f} best={best:.2f}\n  PR: {t[:300]}")
