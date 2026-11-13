# Sealed test — pre-registered plan (written 11 Nov 2026, 21:05 IST, before opening any test score)

Test papers: 17483019, 19673075, 23041342 (fixed 10 Nov; never used for training, tuning or design).
Scores used: final models trained on all 15 training papers (seed 1): run 2b `C0L-final`, run 3 `SPAN-final`.
All thresholds come from the 15-paper out-of-fold scores (3-seed means); nothing is chosen on the test papers.

**Primary:** ensemble SPAN + C0L (mean of raw logits) + E (detected citing sentences), selection score =
logit minus the mean logit of the figure's own citing sentences, one threshold (grid -10..4, step 0.1) maximising
mean per-figure F1 on the 15 training papers.  Out-of-fold: F1 0.766 vs B0f 0.752.
**Also reported:** C0L+E raw score; SPAN+E raw score; B0f.
**Win criterion:** primary F1 > B0f F1 on the test papers; paired bootstrap over figures reported (18 figures, so
differences below ~0.05 are expected to be inconclusive).
**After this:** the test set is spent; results are reported as they come out and nothing is tuned on them.
Caveat known in advance: final models are single-seed and trained on 15 papers, so their raw-logit scale can differ
from the 3-seed out-of-fold scores the thresholds were chosen on.
