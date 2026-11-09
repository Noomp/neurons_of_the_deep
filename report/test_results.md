# Test results

The final evaluation's results in one place, for the report: every condition on the four test sets, then the failure
analysis. **Test sets were used for nothing else**: every choice was made on val ([val_results.md](val_results.md)).

**Source.** `outputs/eval/` (one cell per model, tier, run, task and test set), summarised by
`python -m evaluation.summarise` (`outputs/results/evaluation.md`) and `python -m analysis.failures`
(`outputs/results/failures.md`); figures by `python -m analysis.figures` (`report/figures/results_*`). `outputs/` is
not in git, so this file is the record.

**Protocol.** 500 episodes per task and test set, the same seeds (rooms) for every condition; `test_nonce` uses the
same rooms as `test` with invented names. Trained conditions: the checkpoint kept by val cross-entropy, for each of 3
training seeds, greedy actions. B1 and B2: one deterministic run. Latency: the policy call alone at batch size 1 on an
RTX 5080, nothing else on the GPU. B1 runs on the hard tier on `test` only (evaluation README, design choices).

**Notation.** Success of a tier = mean over its two tasks. `[a, b]`: 95% interval; trained conditions over the 3
seeds (t), B1, B2 and random over episodes (Wilson). Differences between two trained conditions: Welch interval over
seeds, degrees of freedom rounded down. **Bold**: the interval excludes zero. P = P-L12-attn-h1 (layer 12, learned
attention pooling, 1 hidden layer) unless named; P (L28) = P-L28-attn-h1.

## 1. Success rate

| tier | condition | test | 16 distractors | invented words | held-out goals |
|---|---|---|---|---|---|
| easy | random | 0.202 [0.178, 0.228] | 0.233 [0.208, 0.260] | 0.202 [0.178, 0.228] | 0.218 [0.194, 0.245] |
| easy | B1 generate | 0.062 [0.049, 0.079] | 0.065 [0.051, 0.082] | 0.062 [0.049, 0.079] | 0.063 [0.050, 0.080] |
| easy | B2 score | 0.062 [0.049, 0.079] | 0.065 [0.051, 0.082] | 0.062 [0.049, 0.079] | 0.063 [0.050, 0.080] |
| easy | B3 symbolic | 0.867 [0.796, 0.938] | 0.725 [0.651, 0.799] | 0.399 [0.363, 0.436] | 0.508 [0.441, 0.575] |
| easy | **P** | **0.876 [0.871, 0.881]** | 0.691 [0.685, 0.696] | **0.808 [0.785, 0.831]** | **0.879 [0.851, 0.907]** |
| hard | random | 0.046 [0.035, 0.061] | 0.032 [0.023, 0.045] | 0.046 [0.035, 0.061] | 0.043 [0.032, 0.057] |
| hard | B1 generate | 0.000 [0.000, 0.004] | - | - | - |
| hard | B2 score | 0.000 [0.000, 0.004] | 0.000 [0.000, 0.004] | 0.000 [0.000, 0.004] | 0.000 [0.000, 0.004] |
| hard | B3 symbolic | 0.225 [0.097, 0.354] | 0.157 [0.080, 0.234] | 0.165 [0.082, 0.248] | 0.189 [0.099, 0.280] |
| hard | **P** | 0.588 [0.538, 0.638] | 0.319 [0.296, 0.343] | **0.255 [0.202, 0.308]** | 0.588 [0.551, 0.626] |
| hard | P (L28) | 0.628 [0.603, 0.653] | 0.322 [0.287, 0.358] | 0.003 [-0.002, 0.009] | 0.622 [0.609, 0.636] |

Per task:

| condition | task | test | 16 distractors | invented words | held-out goals |
|---|---|---|---|---|---|
| B3 | goto | 0.950 | 0.863 | 0.632 | 0.680 |
| B3 | pickup | 0.783 | 0.587 | 0.167 | 0.336 |
| P | goto | 0.901 | 0.740 | 0.852 | 0.894 |
| P | pickup | 0.851 | 0.641 | 0.764 | 0.864 |
| B3 | putnext | 0.343 | 0.247 | 0.245 | 0.285 |
| B3 | pickup_then_goto | 0.107 | 0.067 | 0.085 | 0.094 |
| P | putnext | 0.650 | 0.338 | 0.229 | 0.632 |
| P | pickup_then_goto | 0.526 | 0.301 | 0.281 | 0.545 |
| P (L28) | putnext | 0.669 | 0.347 | 0.002 | 0.650 |
| P (L28) | pickup_then_goto | 0.587 | 0.298 | 0.005 | 0.595 |

B1 and B2 succeed only on `goto` (0.124-0.130, below random's 0.286), where facing the goal object is enough; both
score 0 on every other task.
Figure: `results_success`.

## 2. RQ1: the head against the zero-shot baselines and B3

| comparison | test | 16 distractors | invented words | held-out goals |
|---|---|---|---|---|
| easy: P − B3 | +0.009 [-0.062, +0.081] | -0.035 [-0.109, +0.039] | **+0.409 [+0.377, +0.441]** | **+0.371 [+0.299, +0.443]** |
| hard: P − B3 | **+0.363 [+0.225, +0.500]** | **+0.162 [+0.082, +0.243]** | **+0.090 [+0.017, +0.163]** | **+0.399 [+0.301, +0.497]** |

- **P against B1 and B2.** P is far above both on every set (0.876 against 0.062 on easy `test`); B1 and B2 are
  below random on both tiers. They act through the meaning of the action names and never pick anything up (B2: 0
  pickups; B1: 10 in 241,396 easy and 0 in 128,000 hard decisions), so every task but `goto` is out of reach.
- **P against B3, in distribution.** Tied on the easy tier, as H1 predicts; on the hard tier P is 2.6 times B3.
  B3's hard-tier success also varies more across seeds (0.285 / 0.196 / 0.195).
- **Cost.**

| condition | trainable parameters | training per seed (easy / hard) | latency median, test (easy / hard) | LLM forward passes per decision |
|---|---|---|---|---|
| B1 | 0 | - | 155 / 155 ms | 1 prompt + up to 8 generated tokens, then an embedding model |
| B2 | 0 | - | 42 / 41 ms | 1 prompt pass, then the six action names scored as a batch on its cache |
| B3 | 583,614 | 17 s / 43 s | 1.1 / 1.1 ms | none |
| P | 331,654 | 5.5 min / 43 min | 9.2 / 12.7 ms | 1, truncated after layer 12 |
| P (L28) | 331,654 | - / 50 min | - / 19.8 ms | 1, all 28 layers |

P is about 4.5 times faster than B2 and 17 times faster than B1, and 8-12 times slower than B3. P's training time
excludes the one-off hidden-state cache (about 14 min for all demos). Latency grows with the prompt: P takes 10.0 ms
over all four hard-tier sets but 12.7 ms on hard `test`. Figure: `results_cost`.

## 3. Demonstration-count curve (easy tier, `test`)

| demonstrations per task | 100 | 250 | 500 | 1000 | 2000 |
|---|---|---|---|---|---|
| B3 | 0.255 [0.105, 0.406] | 0.334 [0.220, 0.448] | 0.398 [0.252, 0.543] | 0.593 [0.407, 0.778] | 0.867 [0.796, 0.938] |
| P | 0.655 [0.593, 0.717] | 0.702 [0.677, 0.726] | 0.721 [0.687, 0.755] | 0.836 [0.816, 0.856] | 0.876 [0.871, 0.881] |
| P − B3 | **+0.400 [+0.238, +0.562]** | **+0.368 [+0.252, +0.484]** | **+0.323 [+0.174, +0.473]** | **+0.243 [+0.056, +0.430]** | +0.009 [-0.062, +0.081] |

With 100 demonstrations per task P reaches 0.655, which B3 needs more than 1000 to match. The gap closes only at
2000. P's seeds also agree more closely at every count. Figure: `results_demo_curve`.

## 4. RQ2: layer and pooling on the test sets

**Pooling under clutter (H2), easy tier, layer 12:**

| pooling | test | 16 distractors | gap | relative drop |
|---|---|---|---|---|
| last token | 0.394 [0.329, 0.458] | 0.250 [0.215, 0.285] | -0.144 | -36% |
| mean | 0.667 [0.576, 0.759] | 0.375 [0.316, 0.434] | -0.292 | -44% |
| learned attention | 0.876 [0.871, 0.881] | 0.691 [0.685, 0.696] | -0.185 | -21% |

| attention's gain | test | 16 distractors | change with clutter |
|---|---|---|---|
| over last token | **+0.482 [+0.418, +0.547]** | **+0.441 [+0.405, +0.476]** | -0.042 [-0.115, +0.032] |
| over mean | **+0.209 [+0.117, +0.300]** | **+0.315 [+0.256, +0.374]** | +0.107 [-0.035, +0.249] |

Attention is far better than either fixed pooling, and it loses the least in relative terms under clutter. However,
its absolute gain does not measurably grow with clutter: it grows over mean pooling, but the interval includes zero,
and it shrinks slightly over last-token pooling.

**Layer 12 against layer 28 (H2 control), hard tier, attention pooling:**

| | test | 16 distractors | invented words | held-out goals |
|---|---|---|---|---|
| P (L28) − P (L12) | +0.040 [-0.016, +0.096] | +0.003 [-0.028, +0.034] | **-0.252 [-0.305, -0.199]** | +0.034 [-0.006, +0.074] |

- **On real names, the final layer is no better.** L28 is +0.003 to +0.04 above L12 on three sets, with no interval
  excluding zero (val agreed: -0.003). It costs twice the latency (19.8 against 12.7 ms on `test`).
- **With invented names, the final layer collapses.** L28 succeeds in 0.3% of hard `test_nonce` episodes; L12 in
  25.5%. In 2,627 of L28's 2,990 failures it turns on the spot for the last 16 steps (`left` is 75% of its actions).
  This is the clearest layer effect in the project: the same head on layer 12 still acts, while on layer 28 the
  representation of unfamiliar words puts it where it never was in training. It is measured on the hard tier only
  (L28 was not evaluated on the easy tier).

Val results by layer and pooling (selection data, easy tier): `results_layers`; [val_results.md](val_results.md) §2,
§7.

## 5. RQ3: generalisation gaps

Shifted set minus `test` (relative drop in brackets):

| tier | condition | 16 distractors | invented words | held-out goals |
|---|---|---|---|---|
| easy | B2 | +0.003 [-0.018, +0.024] (+5%) | 0.000 [-0.021, +0.021] (0%) | +0.001 [-0.020, +0.022] (+2%) |
| easy | B3 | **-0.141 [-0.192, -0.091] (-16%)** | **-0.467 [-0.565, -0.370] (-54%)** | **-0.359 [-0.496, -0.221] (-41%)** |
| easy | P | **-0.185 [-0.188, -0.182] (-21%)** | **-0.068 [-0.090, -0.046] (-8%)** | +0.003 [-0.022, +0.028] (0%) |
| hard | B3 | **-0.068 [-0.124, -0.012] (-30%)** | -0.060 [-0.207, +0.087] (-27%) | -0.036 [-0.076, +0.004] (-16%) |
| hard | P | **-0.269 [-0.321, -0.216] (-46%)** | **-0.333 [-0.364, -0.302] (-57%)** | 0.000 [-0.017, +0.018] (0%) |
| hard | P (L28) | **-0.306 [-0.329, -0.283] (-49%)** | **-0.625 [-0.650, -0.599] (-99%)** | -0.006 [-0.037, +0.026] (-1%) |

- **Held-out goals: P has no gap on either tier.** P was never told to pick up a yellow box, red key or green ball,
  yet it does so as well as anything else. B3 loses 41% on the easy tier. Its failures there face the goal less often
  than on `test` (30% against 54%). That fits an avoidance of objects it saw only as distractors in training, though
  this is not tested directly.
- **Invented words: P loses 8% on the easy tier, B3 54%.** Under `test_nonce` the descriptions are renamed
  consistently with the goal ("dueno tism" in both), so P can still match the goal's words to an object's
  description. B3 reads the goal through its own word embeddings, for which the invented words are unknown. On the
  hard tier P loses 57% (L12), so matching two invented names in one instruction is much harder.
- **Clutter: P loses at least as much as B3.** Easy: -0.185 against -0.141 (difference -0.044 [-0.095, +0.007]).
  Hard: -46% against -30%. The prompt with 16 distractors is longer and lists more objects, while B3's 7x7 grid input
  keeps its size.
- **B1 and B2 have no gap, because they have nothing to lose.** Their success is the same on `test` and
  `test_nonce`, exactly so, because their behaviour does not depend on the names.

Figure: `results_gaps`.

## 6. Failure analysis

Every failure is a time-out: no episode ends early without success. Categories (share of **all** episodes; each row
adds up to the failure rate) for B2, B3, P and P (L28); full tables in `outputs/results/failures.md`. Figure:
`results_failures`.

| tier | test set | condition | failed | wrong object | first step only | reached, no pickup | stuck | wandered |
|---|---|---|---|---|---|---|---|---|
| easy | test | B2 | 0.938 | 0.000 | - | 0.057 | 0.881 | 0.000 |
| easy | test | B3 | 0.133 | 0.121 | - | 0.000 | 0.008 | 0.005 |
| easy | test | P | 0.124 | 0.049 | - | 0.000 | 0.061 | 0.014 |
| easy | 16 distractors | B3 | 0.275 | 0.252 | - | 0.000 | 0.018 | 0.005 |
| easy | 16 distractors | P | 0.309 | 0.134 | - | 0.002 | 0.136 | 0.037 |
| easy | invented words | B3 | 0.601 | 0.485 | - | 0.014 | 0.005 | 0.097 |
| easy | invented words | P | 0.192 | 0.081 | - | 0.002 | 0.098 | 0.012 |
| easy | held-out goals | B3 | 0.492 | 0.367 | - | 0.019 | 0.012 | 0.094 |
| easy | held-out goals | P | 0.121 | 0.050 | - | 0.000 | 0.057 | 0.014 |
| hard | test | B2 | 1.000 | 0.000 | 0.000 | 0.117 | 0.883 | 0.000 |
| hard | test | B3 | 0.775 | 0.742 | 0.032 | 0.000 | 0.000 | 0.000 |
| hard | test | P | 0.412 | 0.233 | 0.115 | 0.000 | 0.055 | 0.009 |
| hard | test | P (L28) | 0.372 | 0.199 | 0.103 | 0.002 | 0.053 | 0.016 |
| hard | invented words | P | 0.745 | 0.125 | 0.218 | 0.152 | 0.212 | 0.037 |
| hard | invented words | P (L28) | 0.997 | 0.013 | 0.017 | 0.016 | 0.946 | 0.004 |

`never_saw_goal` occurs once in 38,299 failures, so it is left out.

**Two failure modes separate B3 from P.**

- **B3 fails by picking up the wrong object.** That is 91% of its easy-tier failures and 96% of its hard-tier
  failures on `test`. It almost never gets stuck.
- **P fails in two ways about equally on the easy tier:**
  - a wrong pickup;
  - getting stuck, almost always by walking into an object or wall (157 of 183 stuck failures on easy `test`).

  Clutter mostly adds stuck failures (0.061 to 0.136 of episodes).
- **On the hard tier, over a quarter of P's failures complete the first step.** P picks up the right object and then
  fails to place it or to reach the second object; B3 almost never gets that far.

**What the wrong object is, against chance** (share of wrong-object failures; in brackets, the share a uniformly
random wrong pickup would have in the same rooms):

| tier | test set | condition | the other named object | look-alike (shares colour or type) | unrelated |
|---|---|---|---|---|---|
| easy | test | B3 | - | 0.80 (0.44) | 0.20 (0.56) |
| easy | test | P | - | 0.84 (0.45) | 0.16 (0.55) |
| easy | invented words | B3 | - | 0.38 (0.41) | 0.62 (0.59) |
| easy | invented words | P | - | 0.83 (0.44) | 0.17 (0.56) |
| hard | test | B3 | 0.20 (0.16) | 0.57 (0.56) | 0.23 (0.28) |
| hard | test | P | 0.39 (0.16) | 0.53 (0.58) | 0.07 (0.26) |

- **On the easy tier, both agents' wrong pickups are near misses:** look-alikes at almost twice the chance rate.
  They get one attribute right and the other wrong.
- **With invented words, B3's wrong pickups fall to chance; P's stay near misses.** B3 no longer reads the goal at
  all, while P still uses part of it, which matches the gaps in §5.
- **On the hard tier, P's characteristic error is role confusion.** It picks up the *other* named object (in putnext,
  the target instead of the object to move) at 2.4 times chance.
- **B3's hard-tier wrong pickups are at chance in every column.** It does not parse two-object instructions, which
  explains its 0.225.

**Action confusion** (the expert's action in the policy's own states against the policy's action, `test`;
`results_confusion`):
- **Forward is rarely confused** (0.83-0.87 for both agents).
- **Turns are the weak spot.** When the expert would turn, both agents often go forward.
- **B3 spams `pick up` on the easy tier.** Where the expert turns, B3 picks up in 52% of decisions, almost all in its
  64-step failed episodes (36 pick-ups or drops that change nothing per failed episode).
- **On the hard tier B3 rarely picks up when it should** (0.09 of the expert's pick-up decisions; P: 0.43).
- **`drop`, the putnext-only action, is rarely chosen when needed** (B3 0.02, P 0.17 on the hard tier).

**Drift** (`results_drift`):
- **Failures go wrong almost at once.** Failed episodes start close to the expert (step 0: 0.79 for P, 0.69 for B3 on
  the easy tier), fall to about 0.3 by step 4 and settle at 0.1-0.4.
- **Successes stay close for longer.** Successful easy-tier episodes stay above 0.8 for the first 5 steps.
- **The first step off the expert comes early.** 73% of P's failures and 83% of B3's leave the expert's choice within
  their first 4 steps.

These are compounding errors from an early mistake, which DAgger-style relabelling targets (not used:
`TRAIN.dagger_rounds` = 0).

Examples: `results_examples` shows, one row each:
- B3 picking up a grey ball for "the yellow ball";
- P picking up the putnext target;
- P walking into an object in a cluttered room;
- P (L28) turning on the spot under invented words.

## 7. The hypotheses

| hypothesis | verdict | evidence |
|---|---|---|
| **H1** P matches or exceeds B1 and B2 with one forward pass per decision, and is within B3's 95% interval in distribution | **Supported**, and exceeded on the hard tier | P 0.876 vs B1/B2 0.062 (easy), 0.588 vs 0 (hard); one truncated pass, 9-13 ms vs 41 / 155 ms. Easy P − B3 +0.009 [-0.062, +0.081]; hard +0.363 [+0.225, +0.500]. |
| **H2a** intermediate layers outperform the final layer | **Not supported on real names; strongly supported under invented words** | Hard L28 − L12: +0.04 / +0.00 / +0.03 (n.s.) on test / clutter / held-out; **-0.252** on invented words. Val: middle layers better with mean pooling only ([val_results.md](val_results.md) §7). |
| **H2b** attention pooling outperforms last-token pooling, most with many distractors | **First part supported; second not** | +0.48 on test, +0.44 with 16 distractors; the gain does not grow with clutter (-0.04 [-0.12, +0.03]). |
| **H3** P drops more than B2 under invented words, less under more distractors | **Untestable against B2; against B3 the opposite** | B2 is below random with no gap. Against B3, P drops far less under invented words (-8% vs -54% easy) and slightly more under clutter (-21% vs -16%; n.s.). |

## 8. What these results do not support

- **A ranking of P and B3 in distribution on the easy tier.** Their difference is +0.009 with an interval of ±0.07.
  B3's seeds vary much more than P's (0.842-0.898 against 0.874-0.878).
- **Claims about B1 or B2 in general.** One small LLM (Qwen3-0.6B), one prompt format (GLAM's), no thinking mode, no
  few-shot examples. Larger models or other prompts may act far better zero-shot.
- **The L28 collapse on the easy tier, or for other poolings.** It was measured for attention pooling on the hard tier
  only.
- **P's best achievable hard-tier score.** The configuration (layer 12, attention, 1 hidden layer) was chosen on the
  easy tier's val and carried to the hard tier unchanged, so the hard-tier head is untuned; a hard-tier sweep might do
  better. (Choosing on val keeps the test scores free of selection bias.)
- **The reason for B3's held-out drop.** Its failures face the goal less, which is consistent with learned avoidance
  of objects seen only as distractors, but no experiment isolates it.
- **Failure categories as exact causes.** They are rule-based on the replayed state (analysis README), ordered, with
  "stuck" a threshold (at most 2 cells in the last 16 steps).
