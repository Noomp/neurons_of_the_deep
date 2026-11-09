# Validation results

Every training run's validation results in one place, for the report. **Validation only:** these numbers chose
the configurations (selection); the test sets are not used until the final evaluation (`evaluation/`), whose
results are in [test_results.md](test_results.md).

**Source.** The `selected` block of each run's `outputs/runs/<ID>/<tier>/seed<k>/run.json` (the epoch kept by
lowest val cross-entropy, then val success measured once). `outputs/` is not in git, so this file is the record.

**Protocol (all runs below).** `config.BC`: AdamW, lr 3e-3, plateau decay x0.5 after 3 epochs without a lower val
cross-entropy, early stopping after 10, upper bound 200 epochs. 2000 demos per task unless the ID ends in `-n<k>`
(k demos per task); 200 val demos per task for val cross-entropy and accuracy; **val success** is the success rate
over 100 val episodes per task, averaged over the tier's tasks. 3 training seeds (0, 1, 2) per ID: independent
runs with different initialisation and batch order.

**Notation.** `mean ± SD`: mean and sample standard deviation (n − 1) over the 3 seeds. `run_plan --summary` prints
the population SD (n), which is about 18% smaller (e.g. B3 easy ± 0.025 there, ± 0.030 here); the means agree.
IDs: `P-L<layer>-<pooling>-h<hidden layers>`; pooling `last` (last token), `mean` (mean over the prompt), `attn`
(learned attention); h1/h2 = 1 or 2 hidden layers of width 256. Val CE and val accuracy are next-action
cross-entropy and accuracy on the val demos (teacher-forced), at the kept epoch. Wall time is per run, training
plus the val-success rollouts (for P, mostly the live LLM in the rollouts); it excludes the one-off hidden-state
caching (about 14 min for all four tasks' demos at every layer, plus the sequence caches for attention).

## 1. Easy tier (`goto`, `pickup`): every configuration, ranked

| # | ID | val success | per seed | goto | pickup | val CE | val acc | epoch kept (of run) | params | wall time |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **B3** | 0.890 ± 0.030 | 0.910 / 0.905 / 0.855 | 0.973 | 0.807 | 0.266 | 0.913 | 4 of 15 / 9 of 20 / 8 of 19 | 583,614 | 17 s |
| 2 | **P-L12-attn-h1** | 0.873 ± 0.013 | 0.860 / 0.885 / 0.875 | 0.873 | 0.873 | 0.238 | 0.915 | 23 of 34 / 18 of 29 / 23 of 34 | 331,654 | 5 min |
| 3 | P-L20-attn-h2 | 0.868 ± 0.008 | 0.860 / 0.875 / 0.870 | 0.900 | 0.837 | 0.232 | 0.920 | 21 of 32 / 18 of 29 / 19 of 30 | 397,446 | 5 min |
| 4 | P-L20-attn-h1 | 0.867 ± 0.019 | 0.845 / 0.880 / 0.875 | 0.877 | 0.857 | 0.240 | 0.921 | 20 of 31 / 18 of 29 / 19 of 30 | 331,654 | 6 min |
| 5 | P-L16-attn-h2 | 0.850 ± 0.009 | 0.855 / 0.855 / 0.840 | 0.863 | 0.837 | 0.239 | 0.916 | 16 of 27 / 18 of 29 / 13 of 24 | 397,446 | 5 min |
| 6 | P-L12-attn-h2 | 0.848 ± 0.010 | 0.860 / 0.840 / 0.845 | 0.867 | 0.830 | 0.250 | 0.914 | 23 of 34 / 18 of 29 / 20 of 31 | 397,446 | 5 min |
| 7 | P-L16-attn-h1 | 0.847 ± 0.032 | 0.865 / 0.865 / 0.810 | 0.877 | 0.817 | 0.236 | 0.920 | 16 of 27 / 18 of 29 / 13 of 24 | 331,654 | 7 min |
| 8 | P-L28-attn-h1 | 0.847 ± 0.012 | 0.840 / 0.840 / 0.860 | 0.880 | 0.813 | 0.260 | 0.912 | 24 of 35 / 39 of 50 / 32 of 43 | 331,654 | 8 min |
| 9 | P-L28-attn-h2 | 0.825 ± 0.023 | 0.845 / 0.830 / 0.800 | 0.860 | 0.790 | 0.260 | 0.909 | 34 of 45 / 23 of 34 / 32 of 43 | 397,446 | 7 min |
| 10 | P-L16-mean-h2 | 0.695 ± 0.080 | 0.605 / 0.720 / 0.760 | 0.727 | 0.663 | 0.514 | 0.797 | 22 of 33 / 27 of 38 / 28 of 39 | 331,782 | 88 s |
| 11 | P-L12-mean-h1 | 0.692 ± 0.054 | 0.630 / 0.730 / 0.715 | 0.780 | 0.603 | 0.510 | 0.801 | 97 of 108 / 66 of 77 / 69 of 80 | 265,990 | 80 s |
| 12 | P-L20-mean-h1 | 0.692 ± 0.019 | 0.705 / 0.670 / 0.700 | 0.760 | 0.623 | 0.510 | 0.811 | 37 of 48 / 44 of 55 / 45 of 56 | 265,990 | 2 min |
| 13 | P-L20-mean-h2 | 0.687 ± 0.053 | 0.640 / 0.745 / 0.675 | 0.743 | 0.630 | 0.522 | 0.803 | 24 of 35 / 28 of 39 / 27 of 38 | 331,782 | 2 min |
| 14 | P-L12-mean-h2 | 0.685 ± 0.035 | 0.670 / 0.725 / 0.660 | 0.753 | 0.617 | 0.529 | 0.797 | 34 of 45 / 40 of 51 / 36 of 47 | 331,782 | 76 s |
| 15 | P-L16-mean-h1 | 0.658 ± 0.033 | 0.660 / 0.625 / 0.690 | 0.717 | 0.600 | 0.503 | 0.807 | 42 of 53 / 49 of 60 / 53 of 64 | 265,990 | 92 s |
| 16 | P-L24-mean-h2 | 0.642 ± 0.057 | 0.625 / 0.595 / 0.705 | 0.693 | 0.590 | 0.527 | 0.797 | 28 of 39 / 33 of 44 / 34 of 45 | 331,782 | 2 min |
| 17 | P-L24-mean-h1 | 0.638 ± 0.058 | 0.580 / 0.695 / 0.640 | 0.723 | 0.553 | 0.527 | 0.794 | 37 of 48 / 40 of 51 / 38 of 49 | 265,990 | 2 min |
| 18 | P-L28-mean-h1 | 0.615 ± 0.035 | 0.650 / 0.580 / 0.615 | 0.663 | 0.567 | 0.563 | 0.777 | 44 of 55 / 44 of 55 / 46 of 57 | 265,990 | 3 min |
| 19 | P-L28-mean-h2 | 0.577 ± 0.013 | 0.590 / 0.565 / 0.575 | 0.637 | 0.517 | 0.554 | 0.779 | 44 of 55 / 33 of 44 / 38 of 49 | 331,782 | 3 min |
| 20 | P-L16-last-h1 | 0.525 ± 0.054 | 0.510 / 0.585 / 0.480 | 0.657 | 0.393 | 0.624 | 0.753 | 47 of 58 / 41 of 52 / 34 of 45 | 265,990 | 114 s |
| 21 | P-L8-mean-h1 | 0.525 ± 0.038 | 0.485 / 0.530 / 0.560 | 0.727 | 0.323 | 0.594 | 0.760 | 62 of 73 / 73 of 84 / 71 of 82 | 265,990 | 76 s |
| 22 | P-L28-last-h1 | 0.522 ± 0.015 | 0.535 / 0.505 / 0.525 | 0.613 | 0.430 | 0.608 | 0.748 | 83 of 94 / 49 of 60 / 77 of 88 | 265,990 | 3 min |
| 23 | P-L20-last-h1 | 0.518 ± 0.048 | 0.510 / 0.475 / 0.570 | 0.613 | 0.423 | 0.606 | 0.751 | 42 of 53 / 35 of 46 / 52 of 63 | 265,990 | 3 min |
| 24 | P-L28-last-h2 | 0.508 ± 0.019 | 0.495 / 0.500 / 0.530 | 0.643 | 0.373 | 0.627 | 0.734 | 59 of 70 / 48 of 59 / 51 of 62 | 331,782 | 3 min |
| 25 | P-L16-last-h2 | 0.490 ± 0.017 | 0.480 / 0.510 / 0.480 | 0.640 | 0.340 | 0.638 | 0.745 | 31 of 42 / 29 of 40 / 34 of 45 | 331,782 | 117 s |
| 26 | P-L8-last-h2 | 0.468 ± 0.010 | 0.480 / 0.465 / 0.460 | 0.527 | 0.410 | 0.683 | 0.714 | 46 of 57 / 50 of 61 / 41 of 52 | 331,782 | 80 s |
| 27 | P-L8-mean-h2 | 0.468 ± 0.020 | 0.490 / 0.465 / 0.450 | 0.590 | 0.347 | 0.596 | 0.768 | 42 of 53 / 42 of 53 / 38 of 49 | 331,782 | 77 s |
| 28 | P-L8-last-h1 | 0.453 ± 0.046 | 0.475 / 0.400 / 0.485 | 0.537 | 0.370 | 0.685 | 0.711 | 53 of 64 / 53 of 64 / 43 of 54 | 265,990 | 78 s |
| 29 | P-L12-last-h1 | 0.445 ± 0.030 | 0.425 / 0.480 / 0.430 | 0.543 | 0.347 | 0.662 | 0.733 | 42 of 53 / 35 of 46 / 26 of 37 | 265,990 | 105 s |
| 30 | P-L20-last-h2 | 0.440 ± 0.023 | 0.420 / 0.435 / 0.465 | 0.553 | 0.327 | 0.616 | 0.747 | 41 of 52 / 29 of 40 / 32 of 43 | 331,782 | 3 min |
| 31 | P-L24-last-h2 | 0.433 ± 0.038 | 0.450 / 0.390 / 0.460 | 0.520 | 0.347 | 0.617 | 0.745 | 41 of 52 / 40 of 51 / 41 of 52 | 331,782 | 3 min |
| 32 | P-L24-last-h1 | 0.425 ± 0.023 | 0.405 / 0.450 / 0.420 | 0.527 | 0.323 | 0.610 | 0.752 | 48 of 59 / 49 of 60 / 63 of 74 | 265,990 | 3 min |
| 33 | P-L4-mean-h1 | 0.397 ± 0.003 | 0.395 / 0.395 / 0.400 | 0.590 | 0.203 | 0.696 | 0.711 | 101 of 112 / 70 of 81 / 69 of 80 | 265,990 | 66 s |
| 34 | P-L12-last-h2 | 0.390 ± 0.018 | 0.370 / 0.405 / 0.395 | 0.497 | 0.283 | 0.675 | 0.721 | 34 of 45 / 29 of 40 / 26 of 37 | 331,782 | 106 s |
| 35 | P-L4-mean-h2 | 0.383 ± 0.008 | 0.385 / 0.390 / 0.375 | 0.583 | 0.183 | 0.688 | 0.713 | 83 of 94 / 70 of 81 / 69 of 80 | 331,782 | 63 s |
| 36 | P-L4-last-h1 | 0.370 ± 0.013 | 0.385 / 0.360 / 0.365 | 0.433 | 0.307 | 0.723 | 0.693 | 79 of 90 / 68 of 79 / 87 of 98 | 265,990 | 61 s |
| 37 | P-L4-last-h2 | 0.357 ± 0.058 | 0.365 / 0.410 / 0.295 | 0.417 | 0.297 | 0.731 | 0.689 | 56 of 67 / 53 of 64 / 69 of 80 | 331,782 | 60 s |

Random actions: 0.217 (goto 0.308, pickup 0.126; 500 val episodes per task, `outputs/results/reference_baselines.md`).
Attention pooling was trained only at layers 12, 16 and 20 (tied best with fixed pooling, §2) and 28 (control for
H2); layers 4, 8 and 24 have last-token and mean pooling only.

## 2. Easy tier: layer x pooling

Val success, mean ± SD over 3 seeds; `-` = not trained.

| layer | last h1 | last h2 | mean h1 | mean h2 | attn h1 | attn h2 |
|---|---|---|---|---|---|---|
| 4 | 0.370 ± 0.013 | 0.357 ± 0.058 | 0.397 ± 0.003 | 0.383 ± 0.008 | - | - |
| 8 | 0.453 ± 0.046 | 0.468 ± 0.010 | 0.525 ± 0.038 | 0.468 ± 0.020 | - | - |
| 12 | 0.445 ± 0.030 | 0.390 ± 0.018 | 0.692 ± 0.054 | 0.685 ± 0.035 | 0.873 ± 0.013 | 0.848 ± 0.010 |
| 16 | 0.525 ± 0.054 | 0.490 ± 0.017 | 0.658 ± 0.033 | 0.695 ± 0.080 | 0.847 ± 0.032 | 0.850 ± 0.009 |
| 20 | 0.518 ± 0.048 | 0.440 ± 0.023 | 0.692 ± 0.019 | 0.687 ± 0.053 | 0.867 ± 0.019 | 0.868 ± 0.008 |
| 24 | 0.425 ± 0.023 | 0.433 ± 0.038 | 0.638 ± 0.058 | 0.642 ± 0.057 | - | - |
| 28 | 0.522 ± 0.015 | 0.508 ± 0.019 | 0.615 ± 0.035 | 0.577 ± 0.013 | 0.847 ± 0.012 | 0.825 ± 0.023 |
| B3 (no LLM) | 0.890 ± 0.030 | | | | | |

Read-off, best of h1/h2 per cell:

- **Pooling.** Mean beats last-token at every layer (by 0.03 to 0.25); attention beats mean by 0.16 to 0.23 at the four layers where both exist.
- **Layer (H2).** With fixed pooling the middle layers are best: mean pooling peaks at layers 12-20 (0.692, 0.695, 0.692) and falls to 0.615 at the final layer and 0.397 at layer 4. With attention pooling the layers are within 0.03 of each other (0.873, 0.850, 0.868, 0.847 at 12, 16, 20, 28).
- **Depth.** One hidden layer is as good as two (the h1/h2 pairs are within noise, h1 ahead in most).
- **Against B3.** The best P (0.873) is within noise of B3 (0.890) with all 2000 demos per task; B3 is better at `goto`, P at `pickup` (§1).

## 3. Choices made from these results

| choice | value | from |
|---|---|---|
| layers for attention pooling | 12, 16, 20, plus 28 as an H2 control | tied best with fixed pooling (§2: best per layer 0.692, 0.695, 0.692) |
| best P, both tiers | **P-L12-attn-h1** | highest easy val success (0.873); within noise of P-L20-attn-h1/h2, and the cheapest of them (12 layers, 1 hidden layer) |
| hard tier | not swept: B3 and P-L12-attn-h1 unchanged, plus P-L12-mean-h1 (pooling check) and P-L28-attn-h1 (H2 control) | §5 |
| learning rate | 3e-3 for P and B3 | §8 |

## 4. Easy tier: demonstration-count curves

Val success, mean ± SD over 3 seeds. The 2000-demo points are the IDs' own runs (B3, P-L12-attn-h1).

| demos per task | B3 | P-L12-attn-h1 | P − B3 | B3 epoch kept | P epoch kept |
|---|---|---|---|---|---|
| 100 | 0.280 ± 0.074 | 0.673 ± 0.037 | +0.393 | 4 of 15 / 3 of 14 / 6 of 17 | 33 of 44 / 30 of 41 / 28 of 39 |
| 250 | 0.338 ± 0.035 | 0.730 ± 0.009 | +0.392 | 3 of 14 / 5 of 16 / 7 of 18 | 33 of 44 / 33 of 44 / 26 of 37 |
| 500 | 0.417 ± 0.059 | 0.740 ± 0.020 | +0.323 | 5 of 16 / 4 of 15 / 2 of 13 | 20 of 31 / 21 of 32 / 27 of 38 |
| 1000 | 0.608 ± 0.069 | 0.843 ± 0.015 | +0.235 | 5 of 16 / 8 of 19 / 5 of 16 | 33 of 44 / 32 of 43 / 30 of 41 |
| 2000 | 0.890 ± 0.030 | 0.873 ± 0.013 | -0.017 | 4 of 15 / 9 of 20 / 8 of 19 | 23 of 34 / 18 of 29 / 23 of 34 |

P needs far fewer demos: with 100 per task it is at 0.673, which B3 reaches only between 1000 and 2000. B3 catches
up only at 2000.

## 5. Hard tier (`putnext`, `pickup_then_goto`)

Not swept: the configuration chosen on the easy tier, unchanged, against B3, plus two checks. Random actions: 0.047
(putnext 0.024, pickup_then_goto 0.070; 500 val episodes per task).

| ID | val success | per seed | putnext | pickup_then_goto | val CE | val acc | train acc | epoch kept (of run) | wall time |
|---|---|---|---|---|---|---|---|---|---|
| P-L12-attn-h1 | 0.633 ± 0.020 | 0.610 / 0.645 / 0.645 | 0.627 | 0.640 | 0.366 | 0.862 | 0.909 | 38 of 49 / 28 of 39 / 26 of 37 | 43 min |
| P-L28-attn-h1 | 0.637 ± 0.033 | 0.600 / 0.645 / 0.665 | 0.667 | 0.607 | 0.390 | 0.851 | 0.901 | 37 of 48 / 35 of 46 / 44 of 55 | 50 min |
| B3 | 0.228 ± 0.058 | 0.295 / 0.190 / 0.200 | 0.320 | 0.137 | 0.540 | 0.766 | 0.787 | 5 of 16 / 4 of 15 / 2 of 13 | 43 s |
| P-L12-mean-h1 | 0.138 ± 0.020 | 0.135 / 0.160 / 0.120 | 0.063 | 0.213 | 0.639 | 0.738 | 0.826 | 67 of 78 / 48 of 59 / 57 of 68 | 4 min |

- P leads B3 by 0.40 (easy: -0.02).
- Attention's gain over mean pooling at layer 12 grows from 0.18 (easy) to 0.49 (hard): on the longer tasks mean pooling (0.138) is not far above random (0.047).
- Layer 28 with attention (0.637) equals layer 12 (0.633), as on the easy tier.
- B3's kept epochs are early (2-5): see §6.

## 6. Kept checkpoint vs last checkpoint

The kept checkpoint (`best.pt`, lowest val CE) against the last (`last.pt`, 10 epochs later, when early stopping
ended the run), re-measured on the same 100 val episodes per task. Mean over the tier's tasks.

| model | tier | kept | last | per seed, kept → last |
|---|---|---|---|---|
| P-L12-attn-h1 | easy | 0.873 | 0.893 | 0.860 → 0.905, 0.885 → 0.890, 0.875 → 0.885 |
| P-L12-attn-h1 | hard | 0.633 | 0.628 | 0.610 → 0.605, 0.645 → 0.630, 0.645 → 0.650 |
| B3 | easy | 0.890 | 0.922 | |
| B3 | hard | 0.228 | 0.393 | |

On the hard tier B3 overfits (training accuracy reaches 100% within about 15 epochs while val accuracy stays at
about 0.78), and its val CE is lowest at epoch 2-5 and rises after, while the actions it picks still improve, so the rule keeps
an early B3. The rule is kept as agreed, for every model; P leads B3 on the hard tier under either choice
(`training/README.md`, Limitations). The selection results (§1, §5) are the kept checkpoints.

## 7. Each layer against the final layer (H2)

Val success of each layer minus layer 28's, at the same pooling and depth, with a Welch 95% interval over the 3
seeds (t with Welch's degrees of freedom); **bold** when the interval excludes zero. Positive = the earlier layer is
better.

| pooling | L4 | L8 | L12 | L16 | L20 | L24 |
|---|---|---|---|---|---|---|
| last h1 | **-0.152 [-0.19, -0.11]** | -0.068 [-0.19, +0.05] | -0.077 [-0.16, +0.01] | +0.003 [-0.14, +0.14] | -0.003 [-0.13, +0.12] | **-0.097 [-0.15, -0.05]** |
| last h2 | **-0.152 [-0.30, -0.00]** | **-0.040 [-0.08, -0.00]** | **-0.118 [-0.17, -0.07]** | -0.018 [-0.07, +0.03] | **-0.068 [-0.12, -0.01]** | -0.075 [-0.18, +0.03] |
| mean h1 | **-0.218 [-0.31, -0.13]** | -0.090 [-0.18, +0.00] | +0.077 [-0.04, +0.19] | +0.043 [-0.04, +0.13] | **+0.077 [+0.00, +0.15]** | +0.023 [-0.10, +0.15] |
| mean h2 | **-0.193 [-0.22, -0.17]** | **-0.108 [-0.15, -0.06]** | **+0.108 [+0.02, +0.20]** | +0.118 [-0.08, +0.32] | +0.110 [-0.03, +0.25] | +0.065 [-0.08, +0.21] |
| attn h1 | - | - | +0.027 [-0.00, +0.06] | +0.000 [-0.08, +0.08] | +0.020 [-0.02, +0.06] | - |
| attn h2 | - | - | +0.023 [-0.04, +0.09] | +0.025 [-0.04, +0.09] | +0.043 [-0.02, +0.10] | - |
| attn h1, hard tier | - | - | -0.003 [-0.07, +0.07] | - | - | - |

What these support:

- **Early layers are worse than the final one.** Layer 4 is below layer 28 under every pooling, layer 8 in most
  comparisons. The most robust layer result.
- **Whether a middle layer beats the final one depends on the pooling.** Mean pooling: layers 12-20 are above
  layer 28 in all six comparisons (+0.04 to +0.12, about +0.09 on average), two individually significant. Last-token
  pooling: the opposite; layer 28 is as good as or better than every other layer. Attention pooling: +0.00 to +0.04
  on the easy tier and -0.003 on the hard tier, none significant. So H2's "intermediate layers outperform the final
  layer" holds for mean pooling only. A reading consistent with §2: fixed pooling depends on how the LLM has mixed
  information into each position, which changes with depth; attention picks the relevant tokens at any depth.
- **The final layer costs twice as much for no measurable gain.** With attention, layer 28 takes 21 ms per decision
  against 10 ms at layer 12 (all 28 layers instead of 12; RTX 5080, batch size 1, 5-episode pilot on the hard tier).

What they do not support:

- **A small layer effect under attention.** With 3 seeds the intervals are ±0.03 to ±0.08 wide, so a middle-layer
  advantage of 0.02-0.05 can be neither shown nor ruled out.
- **Precise absolute levels.** Every run is scored on the same 100 val episodes per task, so the seed spread leaves
  out episode sampling noise (about ±0.03 at these success rates). Comparisons are less affected, since every
  configuration faces the same episodes.
- **Unbiased scores for the chosen layers.** Layers 12-20 were picked as best on the same val episodes, which favours
  them slightly.
- **Single comparisons.** 31 comparisons at 95% give one or two significant results by chance; the patterns across
  many comparisons above are the reliable part.
- **Attention across all depths.** Attention was not trained at layers 4, 8 and 24.

The test-set evaluation of P-L28-attn-h1 next to P-L12-attn-h1 (hard tier) adds estimates independent of selection
and the comparison on each shifted set, but with 3 seeds it will not resolve a difference of a few points either.

## 8. Learning-rate selection

Chosen by val CE on the easy tier, all 2000 demos per task, 3 seeds, for B3 and the best head configuration of the
earlier lr 1e-2 sweep (appendix). Mean over seeds of the kept epoch's val CE, val success in brackets. These runs
used `lr_patience` 2 and `patience` 6 (B3 at 1e-3: `patience` 20); the final values are 3 and 10.

| model | 1e-3 | **3e-3** | 1e-2 |
|---|---|---|---|
| P, layer 20, mean, 1 hidden layer | 0.585 (0.50; 1 seed) | **0.507** (0.71) | 0.561 (0.67) |
| B3 | 0.376 (0.78) | **0.258** (0.87) | 0.517 (0.59) |

At 1e-2 two of B3's three seeds stopped before learning `pickup`; P converged, to a worse minimum.

## Appendix: the superseded lr 1e-2 sweep (easy tier)

Not used for any result; kept because it chose the head configuration for the learning-rate check (§8). lr 1e-2,
`lr_patience` 2, `patience` 6; otherwise as above (runs in `outputs/runs_lr1e-2`). Val success, mean ± SD over
3 seeds.

| layer | last h1 | last h2 | mean h1 | mean h2 |
|---|---|---|---|---|
| 4 | 0.345 ± 0.018 | 0.322 ± 0.072 | 0.388 ± 0.050 | 0.372 ± 0.035 |
| 8 | 0.433 ± 0.033 | 0.400 ± 0.015 | 0.480 ± 0.044 | 0.415 ± 0.074 |
| 12 | 0.383 ± 0.031 | 0.373 ± 0.058 | 0.618 ± 0.010 | 0.558 ± 0.015 |
| 16 | 0.455 ± 0.072 | 0.450 ± 0.074 | 0.658 ± 0.024 | 0.562 ± 0.067 |
| 20 | 0.493 ± 0.046 | 0.397 ± 0.058 | 0.667 ± 0.081 | 0.642 ± 0.045 |
| 24 | 0.388 ± 0.025 | 0.368 ± 0.023 | 0.617 ± 0.045 | 0.562 ± 0.020 |
| 28 | 0.437 ± 0.029 | 0.422 ± 0.104 | 0.538 ± 0.003 | 0.443 ± 0.050 |
| B3 (no LLM) | 0.590 ± 0.258 | | | |
