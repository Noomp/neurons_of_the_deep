# Project Proposal

## A Small Language Model with Learned Spatio-Temporal Memory as a Policy for a Text-Instructed Gridworld

**Course:** SC4001 Neural Networks and Deep Learning — Group Project
**Group:** [Group number]
**Members:** [Name 1], [Name 2], [Name 3]
**Date:** 4 November 2026 (revised; submitted version 23 October 2026, see §14)
**Submission deadline:** 16 November 2026, 11:59 PM

---

## 1. Context

Large language models (LLMs) are increasingly used as agents: they observe an environment, choose actions, and pursue goals over many steps. Most current LLM agents act by *generating text*, such as a tool call or a command, and frontier systems are post-trained end to end with reinforcement learning (RL) so that the text they write leads to good outcomes. This approach is flexible but expensive. Every decision requires token-by-token decoding, the generated text does not always correspond to a valid action, and adapting the policy means updating billions of parameters.

Prior work has identified three ways to turn an LLM into a policy over a fixed set of actions (Carta et al., 2023):

1. **Generate and map:** let the LLM write free text, then map it to the closest valid action (Huang et al., 2022).
2. **Score each action:** compute the LLM's log-probability of each candidate action string and normalise over the action set (Ahn et al., 2022; Carta et al., 2023; Tan et al., 2024).
3. **Action head:** attach a new network that maps the LLM's internal representation directly to a distribution over actions in a single forward pass.

GLAM (Carta et al., 2023) chose option 2 and fine-tuned the entire model with PPO in BabyAI-Text, using substantial compute. In an ablation, a pretrained model with a new action head attached to its final layer learned slowly, which the authors attributed to the final layer being specialised for next-word prediction. Related work initialised policies from pretrained language models and fine-tuned them by imitation (Li et al., 2022), or distilled an LLM's decisions into a small specialist agent (Zhou et al., 2024).

GLAM's prompt also carries the agent's history as text: the last three observations and actions, re-read in full at every step. A cheaper alternative remains under-explored: **give a small language model (SLM) a compact, learned summary of what the agent has seen, as a few soft tokens next to the instruction, keep most of the SLM frozen, and read the action from one hidden state in a single forward pass.** The summary comes from a fixed-size memory of past views that decides when to write, in the spirit of memory-augmented networks (Graves et al., 2014, 2016) and gated RNNs that skip state updates (Campos et al., 2018). Only the SLM's input-side layers are adapted, with LoRA (Hu et al., 2022). The choice of which layer to read from is motivated by probing studies showing that game state can be decoded from a sequence model's intermediate representations (Li et al., 2023). In robotics, replacing token-by-token action generation with direct action prediction has produced large speed-ups (Kim et al., 2025), suggesting practical value beyond this project.

## 2. Problem Statement

Prompted LLM agents are slow and can produce invalid actions, while fully fine-tuning LLMs as policies is computationally expensive. A design that keeps the SLM out of the recurrent path and tunes only a small part of it is affordable, but there is little controlled evidence on three questions:

- whether a pretrained SLM's language knowledge adds anything to a trained gridworld policy, compared with the same architecture without the SLM and with a small non-language model trained on the same demonstrations;
- where in the SLM the decision-relevant information is most accessible, and whether a learned memory of past views helps under partial observability;
- how such trained policies hold up under distribution shift compared with language-grounded zero-shot methods that exploit the meaning of object and action names.

## 3. Research Questions and Hypotheses

**Main question:** Can a small language model act as a gridworld policy when it reads a learned spatio-temporal memory of the agent's views as soft tokens, and what do its pretrained knowledge and its memory each add over a small non-language model?

| ID | Research question | Hypothesis |
|---|---|---|
| RQ1 | Does a pretrained SLM help a trained policy, compared with zero-shot prompting (B1, B2) and with the same architecture without an SLM (M1-long; M2-randlm if run)? | **H1:** After behaviour cloning, M2 and M3 far exceed B1 and B2, which stay near the random floor. In distribution, they are within the 95% CI of M1-long, because the 15-word mission vocabulary is learnable from demonstrations. LoRA (M3) gives at most a small gain over M2. |
| RQ2 | Which SLM layer gives the most decision-relevant read-out, and does the learned memory beat a fixed window of past views? | **H2:** Reading at layer 10 or 18 matches or beats the final layer 28 (M2), as GLAM's explanation predicts. With the full 7×7 view, memory gives no in-distribution gain (M1 ≈ M1-nomem ≈ B3) but helps on `test_dist16`. |
| RQ3 | How robust are trained policies under distribution shift compared with language-grounded methods? | **H3:** All trained models drop sharply on `test_nonce`, because invented words cannot be tied to the unchanged grid codes, and their relative drop is larger than B2's. On `test_heldout`, M2 and M3 drop less than M1 and B3. On `test_dist16`, M1 drops less than M1-nomem and B3. |

RL fine-tuning (M4, stage 4) is a stretch goal with no research question of its own (§6.3, §14).

## 4. Objectives

- **O1.** Build a reproducible pipeline: environment with fixed splits and shifted test sets, and expert demonstrations.
- **O2.** Implement baselines: random and expert references, prompted generation with action mapping (B1) and log-probability action scoring (B2) on the same SLM, and a non-language agent trained on symbolic observations (B3).
- **O3.** Implement the SLM policy (view encoder, write-sparse slot memory, space-time attention, SLM reading soft tokens) and train it by behaviour cloning in stages 1–3, with the planned controls; recurrent PPO (stage 4) if time allows.
- **O4.** Measure generalisation under controlled distribution shifts and report inference and training cost.
- **O5.** Analyse failure cases qualitatively and quantitatively, including the memory's write rate and attention maps.
- **O6.** Deliver a report and commented, reproducible code that meet the course requirements.

**Out of scope:** pretraining or full fine-tuning of the SLM (only LoRA on its low-level layers is trained), the SLM's thinking mode, multi-room or long-horizon tasks, and environments beyond BabyAI-Text.

## 5. Related Work

| Work | Approach | Relation to this project |
|---|---|---|
| Carta et al. (2023), GLAM | Flan-T5 as policy in BabyAI-Text; log-probability action scoring; full PPO fine-tuning | Closest work. Provides the environment, prompt template and held-out goals. Its action-head ablation used the final layer with full fine-tuning; we compare read-out layers with most of the SLM frozen. |
| Huang et al. (2022) | Frozen LLM generates plans; phrases mapped to valid actions by sentence-embedding similarity | Basis for our generate-and-map baseline (B1). |
| Li et al. (2022), LID | Policy initialised from a pretrained LM, trained by imitation in BabyAI and VirtualHome | Found input format mattered little while LM pretraining mattered; relevant to RQ1 and RQ3. |
| Tan et al. (2024), TWOSOME | Log-probability scoring of valid actions with normalisation, trained with RL | Source of length normalisation for our scoring baseline (B2). |
| Zhou et al. (2024), LLM4Teach | Small student agent distils an LLM teacher, then shifts to RL loss | Related to the dropped P-T condition (§14). |
| Li et al. (2023), Othello-GPT | Probes recover board state from a sequence model's hidden states | Motivates the read-out-layer comparison in RQ2. |
| Kim et al. (2025), OpenVLA-OFT | Parallel decoding and direct action regression in vision-language-action models | Evidence that bypassing token generation improves speed in practice. |
| Paglieri et al. (2025), BALROG | Benchmark of LLM and VLM agents on games including BabyAI | Reference for zero-shot LLM agent performance on related environments. |
| Graves et al. (2014, 2016), NTM / DNC | External memory with learned write heads and addressing | Our slot memory is a stripped-down version: one scalar write gate and a slot address. |
| Campos et al. (2018), Skip RNN | Gated RNN that learns to skip state updates, with straight-through binary gates (Bengio et al., 2013) | Basis for our hard-write mode and for making writes rare. |
| Parisotto & Salakhutdinov (2018), Neural Map; Parisotto et al. (2020), GTrXL | Allocentric map memory; transformer memory for RL | Alternatives to our memory, considered and not used (feasibility note §3.4). |
| Perez et al. (2018), FiLM | Feature-wise conditioning of convolutions on language | Used in B3 and in the space-time attention scorer, as in the BabyAI baselines. |
| Hu et al. (2022), LoRA | Low-rank adapters on frozen weights | Adapts the SLM's low-level layers (stage 3). |
| Schulman et al. (2016, 2017), PPO / GAE; Kapturowski et al. (2019), R2D2 | Clipped policy gradient, advantage estimation; stored recurrent state for sequence replay | Algorithm of the stretch stage 4. |

## 6. Methodology

### 6.1 Environment and Tasks

We use **BabyAI-Text** (Carta et al., 2023), a text-only version of BabyAI (Chevalier-Boisvert et al., 2019). The agent receives a goal and a text description of its partial view, with object positions given in steps relative to the agent, and chooses one of six actions: *turn left, turn right, go forward, pick up, drop, toggle*. Each observation also carries MiniGrid's 7×7 symbolic view and the agent's heading, which describe the same state as the text.

- **Tasks:** four single-room tasks with 8 distractors, in two tiers. Easy (64-step limit): *Go to <object>* and *Pick up <object>*. Hard (128-step limit, two named objects): *Put <object> next to <object>* and *Pick up <object>, then go to <object>*. Both hard tasks come from GLAM's training mix; the easy tier is the core, and the hard tier is used where time allows (§10).
- **Inputs:** B1 and B2 read the GLAM prompt (goal, last three observations, last two actions). B3 reads the same window symbolically (three stacked views, two actions) and the mission word by word. The SLM policy reads the symbolic view and step index each step, and the mission as text; it gets neither the heading (which no other condition gets) nor its previous actions.
- **Data splits:** environment seeds are partitioned into training (demonstrations and RL rollouts), validation (model selection), and test (final evaluation, used once).
- **Shifted test sets:** (a) `test_dist16`: 16 distractors instead of 8; (b) `test_nonce`: object and colour names replaced by invented words in the mission and the text (the symbolic view keeps its codes, so for the B3 and M models only the mission changes); (c) `test_heldout`: one named object is a held-out goal (yellow box, red key, green ball), never named in training. Shifted sets reuse the test seeds, so a drop comes from the shift, not from different rooms.
- **Implementation:** the BabyAI-Text package could not be installed on our compute, so, as planned in the submitted proposal, we wrote a text wrapper over Farama's MiniGrid. Tests check its descriptions and prompts against code transcribed from GLAM's repository.

### 6.2 Backbone

**Qwen3-0.6B** (Qwen Team, 2025; 28 layers, width 1024) in bfloat16, shared by B1, B2 and M2–M4. In the SLM policy its layers are split into **low-level** (layers 0–9, which first process the soft tokens and receive LoRA adapters of rank 8 on the attention q/v projections in stage 3) and **high-level** (layers 10–27 and the LM head, always frozen). The action is read from the last token's hidden state at a chosen layer; layers above it are dropped, so reading lower is also cheaper. The SLM's thinking mode is not used.

### 6.3 Experimental Conditions

Baselines and the main model:

| ID | What it is | Language model | Trained by | Question it answers |
|---|---|---|---|---|
| Random | uniform random actions | none | - | chance floor (report next to every result) |
| Expert | MiniGrid's BabyAI bot, reads the full state | none | - | ceiling; teacher for the demonstrations |
| B1 | Qwen3 reads the GLAM text prompt and generates text, mapped to the nearest valid action | frozen | - | can an untrained LM act from text? |
| B2 | Qwen3 log-probability scoring of the six action strings | frozen | - | same, without invalid outputs |
| B3 | frame-stacked CNN + FiLM on the last 3 symbolic views, mission read word by word | none | BC | standard BabyAI-style reference |
| M1 | SLM policy with an MLP in place of the SLM (encoder, memory, attention) | none | BC (stage 1) | base of the M chain |
| M2 | M1 + frozen Qwen3 reading the mission and 4 soft tokens | frozen | BC (stage 2) | does a pretrained LM help? |
| M3 | M2 + LoRA on Qwen3 layers 0-9 | partly tuned | BC (stage 3) | does tuning the LM's input layers help? |
| M4 *(stretch)* | M3 + recurrent PPO | partly tuned | PPO (stage 4) | does RL fix behaviour-cloning errors? |

Controls (each changes one thing):

| ID | Change | Question it answers | Status |
|---|---|---|---|
| M1-nomem | memory off (K = 0) | does the memory help? | planned |
| M1-long | M1 trained for as many epochs as stages 1 and 2 combined | makes M1 vs M2 fair (M2 gets extra training) | planned |
| M2-L{10,18} | read Qwen3 at layer 10 or 18 (layer 28, all of Qwen3, is M2 itself) | where the decision-relevant information sits | planned (easy tier) |
| M2-randlm | Qwen3 architecture with random weights | the LM's knowledge, or just its size? | stretch |
| M1-view{3,5} | agent view reduced to 3×3 or 5×5, with and without memory | makes memory necessary, so the K = 0 control can show a difference | stretch |

The **SLM policy** runs once per step. A view encoder (cell embeddings and residual 3×3 convolutions, plus a time encoding) turns the 7×7 view into 49 cell features. A **write-sparse slot memory** of K = 8 past views is updated by a scalar write gate ("save this view?") and a softmax slot address ("which slot does it replace?"); when nothing is written the memory is copied unchanged. This replaces a plain GRU, which would rewrite its whole state every step and compress the past into one vector (Cho et al., 2014; feasibility note §3). Writes are kept rare by a negative gate bias (Tallec & Ollivier, 2018), a write penalty, and optional hard 0/1 writes with a straight-through gradient (Bengio et al., 2013; Campos et al., 2018). A **space-time attention** scorer, conditioned on the mission through FiLM (Perez et al., 2018), weights all (K + 1) × 49 (time, cell) positions and produces H = 4 read-outs. In M1 a small MLP maps them to actions; in M2–M4 an adapter turns them into 4 soft tokens that Qwen3 reads after the mission text. A separate word GRU encodes the mission for the memory and the attention, so the SLM never sits inside the recurrence.

B3 isolates the value of language pretraining: it is trained on the same demonstrations without an LM. No condition gets the heading, because the GLAM prompt never gives it. Two input differences between B3 and the M models are still open (§14, row 20): B3 gets the last two actions and the M models do not, and the M models get the step index and B3 does not.

### 6.4 Training

- **Demonstrations:** collected with BabyAI's built-in procedural expert, which solves every task on every eval set: 2000 training and 200 validation episodes per task.
- **Behaviour cloning, in stages.** Each stage starts from the previous stage's best checkpoint, selected on validation success:

  | Stage | Model | Trains | Frozen |
  |---|---|---|---|
  | 1 | M1 | encoder, mission GRU, memory, attention, MLP, action head | (no SLM) |
  | 2 | M2 | adapter, new action head; stage-1 parts at 0.1× learning rate | all of Qwen3 |
  | 3 | M3 | stage 2 + LoRA on Qwen3 layers 0–9 | Qwen3 layers 10–27, LM head |
  | 4 *(stretch)* | M4 | stage 3 + value head, by recurrent PPO | Qwen3 layers 10–27, LM head |

  Stages 1–3 minimise cross-entropy on expert actions over whole episodes in time order, with full back-propagation through time, plus a penalty on the mean write gate. B3 is trained by behaviour cloning on the same demonstrations, seeds and batches of episodes, with the same optimiser settings and selection rule. Every condition and control is trained by one command with per-run settings (`--set`), and each run records the settings it actually used.
- **No cached hidden states.** The SLM's input depends on the learned memory, and its low-level layers are tuned, so hidden states change during training and cannot be computed once. Cost is kept down instead by one SLM pass per decision on a ~22-token input, a frozen top with no weight gradients, batching all steps of a sequence through the SLM after the cheap recurrent loop, and gradient checkpointing (feasibility note §4).
- **Stage 4 (stretch):** recurrent PPO (Schulman et al., 2017) with GAE (Schulman et al., 2016) on training seeds, 200k environment steps on the easy tier. Each rollout is replayed as a sequence from its stored initial memory, as in R2D2 (Kapturowski et al., 2019). The reward is the environment's sparse reward plus potential-based shaping towards the current goal object, which does not change the optimal policy (Ng et al., 1999).
- **Sample efficiency:** M1 and B3 are trained on increasing numbers of demonstration episodes (100 to 2000) to produce learning curves; the SLM stages use all 2000.
- **Compounding errors:** if behaviour cloning drifts off the expert's state distribution, we will add a DAgger-style step (Ross et al., 2011) in which the expert relabels states visited by the trained policy, with the same settings for every trained condition.

### 6.5 Evaluation

**Metrics**

- Success rate on held-out test episodes (primary metric), reported next to the random floor
- Episode length and BabyAI reward
- Invalid-output rate (B1)
- Inference latency per decision (median ms, batch size 1, same GPU for all conditions)
- Training cost (GPU-minutes) and number of trainable parameters
- Generalisation gap: success on each shifted set minus in-distribution success
- Success as a function of the number of demonstrations (M1, B3)
- Memory behaviour: write rate per step and space-time attention maps (M models)

**Protocol**

- At least 3 training seeds per trained configuration, reported as mean with 95% confidence intervals.
- Read-out layer and hyperparameters are selected on the validation split only; the test and shifted sets are evaluated once.
- All conditions are evaluated on the same `(task, eval set, seed)` episodes and trained on the same demonstrations; all SLM-based conditions share the same backbone.
- All conditions are reported, including negative results.

### 6.6 Failure Analysis

Failed test episodes will be categorised (for example looping or oscillating between actions, getting stuck behind distractors, approaching the wrong object, picking up or dropping at the wrong time, and drifting after an early mistake). We will report per-task breakdowns, action confusion matrices, and representative trajectories rendered with the environment's visualisation tools. For the M models we will also check whether the memory gate collapses to never writing, and show the space-time attention maps of representative steps.

## 7. Expected Contributions

1. A controlled comparison of zero-shot SLM methods, a non-language agent, and SLM-based trained policies, under matched demonstrations and episodes.
2. A read-out-layer analysis of where decision-relevant information sits in a partly frozen SLM, directly testing GLAM's explanation for the slow learning of final-layer action heads, and an ablation of a learned memory against a fixed window.
3. A write-sparse slot memory that decides when to store a view, with its write behaviour reported, and a characterisation of robustness to distribution shift.
4. A practical, single-GPU recipe for SLM-based game agents: one SLM pass per decision, with the SLM kept out of the recurrent path.

Negative results are also informative. For example, finding that M2 and M3 do no better than M1-long, or that the memory adds nothing with a full view, would directly answer the research questions.

## 8. Alignment with Assessment Criteria

| Criterion (weight) | How the project addresses it |
|---|---|
| Experiments and results (40%) | Zero-shot and non-language baselines; one-change controls for the SLM, the memory and the read-out layer; multiple seeds with confidence intervals; strict validation/test separation; shifted test sets; failure analysis. |
| Report presentation (30%) | Clear question-driven structure; learning curves, layer plots, attention maps and cost-versus-success charts; all required sections, including the AI Use Statement. |
| Novelty (20%) | An SLM reading a learned spatio-temporal memory as soft tokens; a write-sparse memory instead of a GRU; a read-out-layer analysis with most of the SLM frozen. |
| Peer review (10%) | Clearly divided responsibilities, with every member able to explain the full pipeline. |

## 9. Risks and Mitigation

| Risk | Likelihood | Mitigation |
|---|---|---|
| The SLM adds nothing measurable: the 15-word mission vocabulary is learnable without it | High | A valid answer to RQ1 if reported as such; M1-long and M2-randlm separate the LM's knowledge from extra training and size. |
| The memory adds nothing measurable: the 7×7 view covers most of the room and expert episodes take 6–12 steps | High | Report it; evaluate on `test_dist16`; smaller-view variant (M1-view{3,5}) if time allows. |
| Write gate collapses to never writing (already seen in early stage-1 runs) | Medium | Log the write rate; train a variant without the write penalty. |
| Behaviour cloning trains slowly or overfits (early stage-1 runs: 0.50 val success on `goto` vs 0.31 random) | Medium | Use all 2000 demonstrations per task, more updates than the current defaults, checkpoint selection on validation, DAgger. |
| GPU memory or time limits on an 8 GB laptop GPU | Medium | Gradient checkpointing, small PPO minibatches, stage 1 on CPU, stages 2–3 on the easy tier first. |
| Egocentric memory: an old slot's cells no longer line up with the current view | Medium | Slot age and the step index are given to the reader (the heading is not, to keep the inputs equal to B3's); note as a limitation (an allocentric map would remove it). |
| Zero-shot baselines perform near random | Medium | Still a valid reference point; report against the random floor and B3. |
| PPO stretch goal is unstable or forgets the behaviour-cloning policy | High | Kept optional; core results do not depend on it. Detach the value head if policy and value interfere. |
| Time pressure near the deadline | High | Fixed milestones below; runs started early; the full matrix of conditions is cut to the easy tier first. |

## 10. Timeline

| Period | Milestones |
|---|---|
| Done by 4 Nov | Environment with four tasks and six eval sets, verified against GLAM; expert demonstrations collected; B3 and the SLM policy implemented and tested; trainers for all four stages; Qwen3-0.6B integration checked; first stage-1 runs; design and feasibility note. |
| 4–7 Nov | Stage 1 on both tiers, 3 seeds, all demonstrations: M1, M1-nomem, M1-long; demonstration-count curves. B3 runs (trainer done 4 Nov). B1 and B2 implemented and evaluated. Batched validation. |
| 7–10 Nov | Stages 2–3 on the easy tier, 3 seeds: M2, M3, read-out layers 10 / 18 (layer 28 is M2). **Milestone: core results by 10 Nov.** |
| 10–13 Nov | Shifted test sets for every condition; latency benchmarks; failure analysis, write rates and attention maps. Stretch, if on schedule: M4 on the easy tier, M2-randlm, M1-view{3,5}, stages 2–3 on the hard tier. |
| Final days: 13–16 Nov | Report writing and figures; code clean-up and reproduction README; AI Use Statement; submission before 16 Nov, 11:59 PM. |

## 11. Team Roles

| Member | Primary responsibility |
|---|---|
| [Name 1] | Environment, expert demonstrations, baselines B1–B3 |
| [Name 2] | SLM policy, training stages 1–4 and the controls |
| [Name 3] | Evaluation protocol, generalisation tests, failure analysis, figures |

All members contribute to the report and review each other's code so that everyone can explain every component.

## 12. Resources and Reproducibility

- **Compute:** a single GPU. The estimates in the feasibility note (§4) assume an RTX 4060 laptop GPU (8 GB) as the cautious case and a desktop RTX 5080 as the fast case: about 18–53 minutes per behaviour-cloning run of stages 2–3, and 0.5–1.6 hours per 200k-step PPO run. Stage 1 runs on a CPU. Checkpoints are saved regularly.
- **Software:** PyTorch, Hugging Face Transformers, PEFT (LoRA), Farama MiniGrid.
- **Reproducibility:** fixed random seeds; every setting in one configuration file, saved next to each dataset and run; a README covering dependencies, demonstration collection, and commands to reproduce each table and figure.
- **Acknowledgement:** all reused code (GLAM's text description format and prompt template, MiniGrid's levels and expert bot) and pretrained models will be credited, and our own contributions identified.

## 13. AI Use Statement (draft)

*[Revise to reflect actual use before submission.]* We used Claude (Anthropic) to brainstorm project directions, search for and summarise related literature, and draft this proposal. We verified every reference against the original paper, and the team chose the research question, designed the experiments, and will run and interpret all experiments.

## 14. Changes since the submitted proposal (23 Oct 2026)

The submitted version is in git history (commit `aa012f3`). The design and its feasibility assessment are in the design note, [report/architecture/architecture_feasibility.pdf](report/architecture/architecture_feasibility.pdf) (cited below as "note §n").

| # | Change | Reason |
|---|---|---|
| 1 | Title and main question: a **frozen LLM with a small action head on cached hidden states** became an **SLM policy**: view encoder → write-sparse slot memory → space-time attention → Qwen3-0.6B reading the mission and 4 soft tokens (§3, §6.3). | The team adopted a design in which the language model reads a learned memory of past views instead of a text history (original design brief in the README). Architecture: note §2. |
| 2 | The memory is a **write-sparse slot memory**, not a plain GRU. | A GRU rewrites its whole state every step and keeps the past in one vector; the slot memory writes rarely and keeps whole views (note §3). |
| 3 | Backbone: **Qwen2.5** (0.5–1.5B) became **Qwen3-0.6B**; the planned second model size is dropped (§6.2). | Implemented and integration-tested with Qwen3-0.6B (note §2, §8). The full matrix of conditions already needs 40–60 GPU-hours, more than the remaining schedule allows (note §4.3). |
| 4 | The LM is **no longer fully frozen**: LoRA on layers 0–9 from stage 3 (§6.2). | The soft tokens are a new kind of input, so the layers that first process them are adapted; the top layers stay frozen to keep the cost of the backward pass low (note §2, §4.1). |
| 5 | **No cached hidden states** (§6.4). | The SLM's input depends on the learned memory and its low-level layers are tuned, so hidden states change during training. Cost is kept down by other means (note §4.1). |
| 6 | **Pooling dropped**; the layer half of old H2 survives as the read-out-layer control **M2-L{10,18,28}** (§3 RQ2, §6.3). | The policy reads one last-token hidden state of a ~22-token input; the layer comparison is the recommended sweep (note §9, rec. 2). |
| 7 | Training is **four stages**: BC without the SLM, BC with the SLM frozen, BC with LoRA, then recurrent PPO (§6.4). | Each stage adds one component, which gives the M1–M4 comparison; the BC warm start makes PPO affordable (note §6, §4.2). |
| 8 | **PPO** changed from a stretch goal with the LLM frozen (old condition S) to stage 4 with LoRA trainable (M4). It remains a **stretch goal with no research question**. | The note estimates 1–3 GPU-hours per easy-tier run and recommends one configuration (note §1, §9, rec. 3). The team decided on 4 Nov to keep the core results to behaviour cloning. |
| 9 | Research questions and hypotheses **rewritten** for the new design: RQ1 is now about the SLM's value, RQ2 about the read-out layer and the memory; **RQ3 (distribution shift) kept**, with H3 revised. | H3 changed because `test_nonce` renames only the text, so the B3 and M models see an unchanged room and only the mission changes (note §5, risk 1). |
| 10 | **RQ4 and condition P-T dropped** (training on B2's decisions). | B2 is expected to be near the random floor, so its labels would teach little; P-T needs B2 implemented first plus another training chain, and the schedule has no room for it (note §4.3). |
| 11 | **Thinking mode dropped.** | There are no reasoning traces to imitate, and hundreds of generated tokens per step are too slow inside PPO (note §1, §5, risk 4). |
| 12 | **Two task tiers**: `putnext` and `pickup_then_goto` added to `goto` and `pickup` (§6.1). | Tests carrying an object, two goals and `drop`, over episodes about twice as long; both tasks come from GLAM's training mix. |
| 13 | Conditions table replaced: **Random and Expert** references, **M1–M4**, and one-change **controls** (§6.3). | Needed to tell whether the SLM and the memory help, both of which the note flags as uncertain (note §1, §5, risks 1–2, §9). |
| 14 | Only B1 and B2 read the GLAM prompt; the M models read the symbolic view and the mission (§6.1). | The memory replaces the prompt's text history (note §2). |
| 15 | The environment is our own text wrapper over MiniGrid (§6.1). | The fallback from the submitted proposal was taken: the BabyAI-Text package is hard to install on current Python and Windows. |
| 16 | Metrics: **write rate and attention maps** added (§6.5–6.6). | They make the memory's behaviour visible and catch gate collapse (note §3.3, §9, rec. 6). |
| 17 | Risks, compute, timeline and team roles updated (§9–12). | New risks and compute estimates from note §4–5 and the early results in note §8; the timeline now covers 4–16 Nov. |
| 18 | References added for the memory, RL and LoRA work the design relies on. | Cited in the note's bibliography. |
| 19 | **No condition gets the heading**: it was removed from the SLM policy, and the `nodir` control was dropped. The read-out-layer control is **M2-L{10,18}**: layer 28 is the whole of Qwen3-0.6B, i.e. M2 itself (§6.1, §6.3). | Decided 4 Nov. B3 and the GLAM prompt never get the heading, so the M models must not either for differences between conditions to come from the models alone. A separate L28 run would duplicate M2. |
| 20 | *Open:* B3 gets the last two actions and the M models do not; the M models get the step index and B3 does not (§6.3). | Found 4 Nov while matching inputs across conditions; to be decided. |

## References

- Ahn, M., et al. (2022). Do As I Can, Not As I Say: Grounding Language in Robotic Affordances. *arXiv:2204.01691*.
- Bengio, Y., Léonard, N., & Courville, A. (2013). Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation. *arXiv:1308.3432*.
- Campos, V., Jou, B., Giró-i-Nieto, X., Torres, J., & Chang, S.-F. (2018). Skip RNN: Learning to Skip State Updates in Recurrent Neural Networks. *ICLR 2018*.
- Carta, T., Romac, C., Wolf, T., Lamprier, S., Sigaud, O., & Oudeyer, P.-Y. (2023). Grounding Large Language Models in Interactive Environments with Online Reinforcement Learning. *ICML 2023*.
- Chevalier-Boisvert, M., et al. (2019). BabyAI: A Platform to Study the Sample Efficiency of Grounded Language Learning. *ICLR 2019*.
- Cho, K., et al. (2014). Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation. *EMNLP 2014*.
- Graves, A., Wayne, G., & Danihelka, I. (2014). Neural Turing Machines. *arXiv:1410.5401*.
- Graves, A., et al. (2016). Hybrid Computing Using a Neural Network with Dynamic External Memory. *Nature, 538*.
- Hu, E. J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*.
- Huang, W., Abbeel, P., Pathak, D., & Mordatch, I. (2022). Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents. *ICML 2022*.
- Kapturowski, S., Ostrovski, G., Quan, J., Munos, R., & Dabney, W. (2019). Recurrent Experience Replay in Distributed Reinforcement Learning. *ICLR 2019*.
- Kim, M. J., Finn, C., & Liang, P. (2025). Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success. *RSS 2025*.
- Li, K., Hopkins, A. K., Bau, D., Viégas, F., Pfister, H., & Wattenberg, M. (2023). Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task. *ICLR 2023*.
- Li, S., et al. (2022). Pre-Trained Language Models for Interactive Decision-Making. *NeurIPS 2022*.
- Ng, A. Y., Harada, D., & Russell, S. (1999). Policy Invariance under Reward Transformations: Theory and Application to Reward Shaping. *ICML 1999*.
- Paglieri, D., et al. (2025). BALROG: Benchmarking Agentic LLM and VLM Reasoning on Games. *ICLR 2025*.
- Parisotto, E., & Salakhutdinov, R. (2018). Neural Map: Structured Memory for Deep Reinforcement Learning. *ICLR 2018*.
- Parisotto, E., et al. (2020). Stabilizing Transformers for Reinforcement Learning. *ICML 2020*.
- Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Courville, A. (2018). FiLM: Visual Reasoning with a General Conditioning Layer. *AAAI 2018*.
- Qwen Team (2025). Qwen3 Technical Report. *arXiv:2505.09388*.
- Ross, S., Gordon, G., & Bagnell, D. (2011). A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. *AISTATS 2011*.
- Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation. *ICLR 2016*.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.
- Tallec, C., & Ollivier, Y. (2018). Can Recurrent Neural Networks Warp Time? *ICLR 2018*.
- Tan, W., Zhang, W., Liu, S., Zheng, L., Wang, X., & An, B. (2024). True Knowledge Comes from Practice: Aligning LLMs with Embodied Environments via Reinforcement Learning. *ICLR 2024*.
- Zhou, Z., Hu, B., Zhao, C., Zhang, P., & Liu, B. (2024). Large Language Model as a Policy Teacher for Training Reinforcement Learning Agents. *IJCAI 2024*.

*Verify all references against the original sources before submission.*
