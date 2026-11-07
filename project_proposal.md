# Project Proposal

## Reading Actions from Frozen Language Models: Lightweight Action Heads as Policies for Text-Based Gridworld Agents

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

Options 1 and 2 are *language-grounded*: the action is chosen through the meaning of the action names ("go forward", "pick up") and of the objects in the prompt, using the LLM's pretraining. Option 3 is not: its outputs mean nothing until demonstrations teach them, and it cannot fall back on word meaning when the situation changes.

GLAM (Carta et al., 2023) chose option 2 and fine-tuned the entire model with PPO in BabyAI-Text, using substantial compute. In an ablation, a pretrained model with a new action head attached to its final layer learned slowly, which the authors attributed to the final layer being specialised for next-word prediction. Related work initialised policies from pretrained language models and fine-tuned them by imitation (Li et al., 2022), or distilled an LLM's decisions into a small specialist agent (Zhou et al., 2024).

A cheaper alternative remains under-explored: **keep the LLM frozen and train only a small head on its hidden states, choosing which layer to read from and how to summarise the tokens.** This mirrors frozen-backbone transfer learning in computer vision (a new classifier on a frozen ResNet) and is motivated by probing studies showing that game state can be decoded from a sequence model's intermediate representations (Li et al., 2023). In robotics, replacing token-by-token action generation with direct action prediction has produced large speed-ups (Kim et al., 2025), suggesting practical value beyond this project.

## 2. Problem Statement

Prompted LLM agents are slow and can produce invalid actions, while fully fine-tuning LLMs as policies is computationally expensive. There is little controlled evidence on three questions:

- whether a frozen LLM's hidden states already contain enough decision-relevant information for a small trained head to act well, compared with prompting the same LLM and with a small non-language agent trained on the same demonstrations;
- where in the network that information is most accessible, and how token representations should be pooled;
- what a fixed action head gives up, compared with language-grounded methods that can exploit the meaning of object and action names, when the environment changes.

## 3. Research Questions and Hypotheses

**Main question:** Can a lightweight action head trained on the hidden states of a frozen LLM serve as an effective and efficient policy for a text-based gridworld, and what does it trade away compared with language-grounded action selection?

| ID | Research question | Hypothesis |
|---|---|---|
| RQ1 | How does a frozen-LLM action head compare with prompted generation (B1), log-probability scoring (B2), and a non-language agent trained on the same demonstrations (B3), in success rate and inference cost? | **H1:** After behaviour cloning, the head matches or exceeds both zero-shot methods in success rate while requiring a single forward pass per decision. In distribution it is within the 95% CI of B3, which needs no LLM, so its cost must be justified by robustness (RQ3). |
| RQ2 | Which layer and which pooling method give the most decision-relevant representation? | **H2:** Intermediate layers outperform the final layer, and learned attention pooling outperforms last-token pooling, with the largest gain when observations contain many distractors. |
| RQ3 | How robust is the head under distribution shift compared with language-grounded methods? | **H3:** The head shows a larger relative drop than log-probability scoring when object names are replaced with invented words, but a smaller drop when the number of distractors increases. |

## 4. Objectives

- **O1.** Build a reproducible pipeline: environment, expert demonstrations, and cached hidden-state extraction from a frozen LLM.
- **O2.** Implement baselines that share the same backbone and prompt: prompted generation with action mapping, and log-probability action scoring, plus a non-LLM agent trained on symbolic observations.
- **O3.** Train and evaluate action heads across layers and pooling methods under a controlled protocol.
- **O4.** Measure generalisation under controlled distribution shifts and report inference and training cost.
- **O5.** Analyse failure cases qualitatively and quantitatively.
- **O6.** Deliver a report and commented, reproducible code that meet the course requirements.

**Out of scope:** pretraining or fine-tuning of the LLM, reinforcement learning, the LLM's thinking mode, multi-room or long-horizon tasks, and environments beyond BabyAI-Text.

## 5. Related Work

| Work | Approach | Relation to this project |
|---|---|---|
| Carta et al. (2023), GLAM | Flan-T5 as policy in BabyAI-Text; log-probability action scoring; full PPO fine-tuning | Closest work. Provides the environment, prompt template and held-out goals. Its action-head ablation used the final layer with full fine-tuning; we study frozen layers instead. |
| Huang et al. (2022) | Frozen LLM generates plans; phrases mapped to valid actions by sentence-embedding similarity | Basis for our generate-and-map baseline (B1). |
| Li et al. (2022), LID | Policy initialised from a pretrained LM, trained by imitation in BabyAI and VirtualHome | Found input format mattered little while LM pretraining mattered; relevant to RQ3. |
| Tan et al. (2024), TWOSOME | Log-probability scoring of valid actions with normalisation, trained with RL | Source of length normalisation for our scoring baseline (B2). |
| Zhou et al. (2024), LLM4Teach | Small student agent distils an LLM teacher, then shifts to RL loss | Related to the dropped RQ4 (§14). |
| Li et al. (2023), Othello-GPT | Probes recover board state from a sequence model's hidden states | Motivates the layer-wise analysis in RQ2. |
| Kim et al. (2025), OpenVLA-OFT | Parallel decoding and direct action regression in vision-language-action models | Evidence that bypassing token generation improves speed in practice. |
| Paglieri et al. (2025), BALROG | Benchmark of LLM and VLM agents on games including BabyAI | Reference for zero-shot LLM agent performance on related environments. |
| Perez et al. (2018), FiLM | Feature-wise conditioning of convolutions on language | Used in B3, as in the BabyAI baselines. |

## 6. Methodology

### 6.1 Environment and Tasks

We use **BabyAI-Text** (Carta et al., 2023), a text-only version of BabyAI (Chevalier-Boisvert et al., 2019). The agent receives a goal and a text description of its partial view, with object positions given in steps relative to the agent, and chooses one of six actions: *turn left, turn right, go forward, pick up, drop, toggle*. Each observation also carries MiniGrid's 7×7 symbolic view, which describes the same state as the text.

- **Tasks:** four single-room tasks with 8 distractors, in two tiers. Easy (64-step limit): *Go to <object>* and *Pick up <object>*. Hard (128-step limit, two named objects): *Put <object> next to <object>* and *Pick up <object>, then go to <object>*. Both hard tasks come from GLAM's training mix; the easy tier is the core, and the hard tier is used where time allows (§10).
- **Prompt:** the GLAM template, containing the goal, the last three observations, and the last two actions, used identically by every LLM-based condition (B1, B2 and the head). B3 reads the same window symbolically: the last three 7×7 views, the two actions between them, and the goal. No condition gets the agent's compass heading, which the prompt never contains.
- **Data splits:** environment seeds are partitioned into training (demonstrations), validation (model selection), and test (final evaluation, used once).
- **Shifted test sets:** (a) `test_dist16`: 16 distractors instead of 8; (b) `test_nonce`: object and colour names replaced by invented words in the goal and the descriptions (the symbolic view keeps its codes, so for B3 only the goal changes); (c) `test_heldout`: one named object is a held-out goal (yellow box, red key, green ball), never named in training. Shifted sets reuse the test seeds, so a drop comes from the shift, not from different rooms.
- **Implementation:** the BabyAI-Text package could not be installed on our compute, so, as planned in the submitted proposal, we wrote a text wrapper over Farama's MiniGrid. Tests check its descriptions and prompts against code transcribed from GLAM's repository.

### 6.2 Backbone

**Qwen3-0.6B** (Qwen Team, 2025; 28 layers, width 1024) in bfloat16, kept frozen throughout and shared by B1, B2 and the head. Its thinking mode is not used. Reading the head's input at layer *l* means the layers above *l* never run, so a lower layer is also cheaper per decision.

### 6.3 Experimental Conditions

| ID | Method | Trainable parameters | LLM passes per decision |
|---|---|---|---|
| Random | Uniform random actions; the chance floor reported next to every result | 0 | None |
| Expert | MiniGrid's BabyAI bot, reads the full state; the ceiling and the teacher for the demonstrations | 0 | None |
| B1 | Prompted generation; output mapped to the nearest valid action by sentence-embedding similarity | 0 | 1 prefill + decoding |
| B2 | Log-probability scoring of the six action strings, with length normalisation | 0 | One per action (shared prefix) |
| B3 | Symbolic behaviour cloning: frame-stacked CNN with FiLM conditioning on the goal, on the last three symbolic views and two actions, trained on the same demonstrations | < 1M | None |
| **P** | **Frozen LLM + action head (proposed).** Variants: layer ∈ {4, 8, 12, 16, 20, 24, 28}; pooling ∈ {last token, mean, learned attention}; MLP with 1–2 hidden layers | ~0.26–0.4M | 1 prefill, truncated at the layer read |

The P variants are run in two steps (§6.4): last-token and mean pooling at every layer, then learned attention pooling at the best layers of that sweep (on the easy tier, layers 12, 16 and 20, which tied), plus the final layer 28 as a control for H2. The hard tier is not swept: it trains B3 and the configuration chosen on the easy tier (layer 12, attention pooling, 1 hidden layer), unchanged, plus mean pooling at layer 12 as a pooling check and attention at layer 28 as the H2 control.

B3 isolates the value of LLM representations: it sees the same information and demonstrations without language pretraining.

### 6.4 Training

- **Demonstrations:** collected with BabyAI's built-in procedural expert, which solves every task on every eval set: 2000 training and 200 validation episodes per task.
- **Behaviour cloning:** cross-entropy loss on expert actions, with the same protocol for B3 and every head: the same demonstrations, seeds, batches of whole episodes, optimiser settings and selection rule (one shared training loop).
- **Cached hidden states.** Because the LLM is frozen, hidden states are computed once per demonstration step and cached for the layers under study (about 14 minutes for all 78k steps on one GPU), so training each head takes seconds to minutes and the layer sweep is inexpensive. Last-token and mean vectors are cached for every layer in the sweep; full token sequences (about 0.6 MB per step per layer) only for the layers used with attention pooling.
- **Schedule and selection.** AdamW with an initial learning rate of 0.003, halved when validation cross-entropy on the validation demonstrations has not improved for 3 epochs; training stops after 10 epochs without improvement (at most 200). The learning rate is shared by B3 and the head, chosen from {0.001, 0.003, 0.01} by validation cross-entropy on the easy tier, where 0.003 is the best for both (training README). Each run keeps the epoch with the lowest validation cross-entropy, then measures greedy validation success once on that epoch. Layers, poolings and other configurations are compared by that success. Test sets are never used for selection.
- **Sample efficiency:** B3 and the best head configuration are trained on increasing numbers of demonstration episodes (100 to 2000) to produce learning curves.
- **Compounding errors:** if behaviour cloning drifts off the expert's state distribution, we will add a DAgger-style step (Ross et al., 2011) in which the expert relabels states visited by the trained policy, applied identically to B3 and the head.

### 6.5 Evaluation

**Metrics**

- Success rate on held-out test episodes (primary metric), reported next to the random floor
- Episode length and BabyAI reward
- Invalid-output rate (B1)
- Inference latency per decision (median ms, batch size 1, same GPU for all conditions)
- Training cost (GPU-minutes, including hidden-state caching) and number of trainable parameters
- Generalisation gap: success on each shifted set minus in-distribution success
- Success as a function of the number of demonstrations (B3, best head)

**Protocol**

- At least 3 training seeds per head configuration and for B3, reported as mean with 95% confidence intervals.
- Layer, pooling, and hyperparameters are selected on the validation split only; the test and shifted sets are evaluated once.
- All LLM-based conditions share the same backbone, prompt, and episodes; all trained conditions share the same demonstrations and training protocol.
- For H2's clutter prediction, the last-token and mean-pooling heads at the best P's layer (P-L12-last-h1, P-L12-mean-h1) are also evaluated on `test` and `test_dist16`, next to the best P.
- For H2 on the hard tier, the final-layer attention head (P-L28-attn-h1) is also evaluated on all four test sets, next to the best P (layer 12).
- All conditions are reported, including negative results.

### 6.6 Failure Analysis

Failed test episodes will be categorised (for example looping or oscillating between actions, getting stuck behind distractors, approaching the wrong object, picking up or dropping at the wrong time, and drifting after an early mistake). We will report per-task breakdowns, action confusion matrices, and representative trajectories rendered with the environment's visualisation tools.

## 7. Expected Contributions

1. A controlled comparison of ways to read actions from the same frozen LLM under matched prompts and data, against a non-language agent trained on the same demonstrations.
2. A layer-wise and pooling analysis showing where decision-relevant information is accessible, directly testing GLAM's explanation for the slow learning of final-layer action heads.
3. A characterisation of the trade-off between inference cost and robustness to distribution shift.
4. A practical, single-GPU recipe for lightweight LLM-based game agents.

Negative results are also informative. For example, finding that heads collapse under name substitution, or that no layer outperforms the symbolic baseline, would directly answer the research questions.

## 8. Alignment with Assessment Criteria

| Criterion (weight) | How the project addresses it |
|---|---|
| Experiments and results (40%) | Baselines sharing the same backbone; a non-language baseline on the same demonstrations; layer and pooling ablations; multiple seeds with confidence intervals; strict validation/test separation; shifted test sets; failure analysis. |
| Report presentation (30%) | Clear question-driven structure; learning curves, layer-sweep plots, and cost-versus-success charts; all required sections, including the AI Use Statement. |
| Novelty (20%) | Frozen-backbone analysis across layers and pooling methods, which prior work did not isolate; explicit cost-versus-robustness trade-off. |
| Peer review (10%) | Clearly divided responsibilities, with every member able to explain the full pipeline. |

## 9. Risks and Mitigation

| Risk | Likelihood | Mitigation |
|---|---|---|
| Zero-shot baselines perform near random | Medium | Still a valid reference point; report against the random floor and B3. |
| Behaviour cloning suffers from compounding errors | Medium | DAgger-style relabelling for every trained condition; analyse drift explicitly in the failure analysis. |
| Validation rollouts dominate the cost: each decision runs the LLM (about 21 ms on an RTX 5080), and a failing episode runs to its 64- or 128-step limit | Medium | Validation success is measured once per run, on the selected epoch; attention pooling only at the best layers; easy tier first. |
| The final evaluation is long: 500 episodes per task and test set, and the zero-shot baselines fail most episodes, which then run to the step limit (B1 about 170 ms per decision: about 1.5 days of GPU time over both tiers; B2 about 9 hours) | High | The evaluation is resumable cell by cell and checks every run before the first episode; easy tier first, hard tier after; nothing else runs on the GPU meanwhile, so latencies stay comparable. |
| Mean pooling is dominated by a few tokens with very large activations | Low | LayerNorm on the head's input; reported as a finding if mean pooling still fails. |
| Time pressure near the deadline | High | Fixed milestones below; the layer sweep is cheap once hidden states are cached; hard tier only where time allows. |

## 10. Timeline

| Period | Milestones |
|---|---|
| Done by 4 Nov | Environment with four tasks and six eval sets, verified against GLAM; expert demonstrations collected; B1, B2, B3 and the action head implemented and tested; hidden-state caching and a shared behaviour-cloning trainer; run plan for every condition. |
| 4–7 Nov | Hidden states cached; B1 and B2 evaluated; B3 and the last-token / mean layer sweep on the easy tier, 3 seeds; attention pooling at the best layers (12, 16, 20) and at layer 28 as a control. **Milestone: core results by 7 Nov.** |
| 7–10 Nov | Hard tier; demonstration-count curves; DAgger only if behaviour cloning drifts. |
| 10–13 Nov | Shifted test sets and latency benchmarks for every condition; failure analysis. |
| Final days: 13–16 Nov | Report writing and figures; code clean-up and reproduction README; AI Use Statement; submission before 16 Nov, 11:59 PM. |

## 11. Team Roles

| Member | Primary responsibility |
|---|---|
| [Name 1] | Environment, expert demonstrations, baselines B1–B3 |
| [Name 2] | LLM feature extraction, caching, action-head training and sweeps |
| [Name 3] | Evaluation protocol, generalisation tests, failure analysis, figures |

All members contribute to the report and review each other's code so that everyone can explain every component.

## 12. Resources and Reproducibility

- **Compute:** a single GPU (measured on an RTX 5080: caching all demonstration steps about 14 minutes; 21 ms per decision for the full 28-layer model at batch size 1). Checkpoints are saved every epoch.
- **Software:** PyTorch, Hugging Face Transformers, Farama MiniGrid.
- **Reproducibility:** fixed random seeds; every setting in one configuration file, saved next to each dataset, cache and run, with the command line and git commit; a README covering dependencies, data preparation (demonstration collection and hidden-state caching), and commands to reproduce each table and figure.
- **Acknowledgement:** all reused code (GLAM's text description format and prompt template, MiniGrid's levels and expert bot) and pretrained models will be credited, and our own contributions identified.

## 13. AI Use Statement (draft)

*[Revise to reflect actual use before submission.]* We used Claude (Anthropic) to brainstorm project directions, search for and summarise related literature, draft this proposal, and write and test code. We verified every reference against the original paper, and the team chose the research question, designed the experiments, and will run and interpret all experiments.

## 14. Changes since the submitted proposal (23 Oct 2026)

The submitted version is in git history (commit `aa012f3`). The main question, research questions RQ1–RQ3 and the proposed method (P) are unchanged.

| # | Change | Reason |
|---|---|---|
| 1 | **Between 23 Oct and 4 Nov the project pursued a different design**: a small language model reading a learned spatio-temporal memory of past views as soft tokens (conditions M1–M4). On 4 Nov the team **returned to the submitted question**. | The memory design no longer answered the submitted question: it had no action head on the LLM's representation of the prompt, so the head could not be compared with B1 and B2 under matched inputs. Its code, tests and design and feasibility note are kept on git branch `archive/slm-memory-policy` (the note is in `report/architecture/` there). |
| 2 | Backbone: **Qwen2.5** (0.5–1.5B) became **Qwen3-0.6B**; the planned second model size is dropped (§6.2). | Integration-tested with Qwen3-0.6B; the schedule leaves no room for a second size. |
| 3 | **Two task tiers**: `putnext` and `pickup_then_goto` added to `goto` and `pickup` (§6.1). | Tests carrying an object, two goals and `drop`, over episodes about twice as long; both tasks come from GLAM's training mix. |
| 4 | The environment is our own text wrapper over MiniGrid (§6.1). | The fallback from the submitted proposal was taken: the BabyAI-Text package is hard to install on current Python and Windows. |
| 5 | **RQ4 and condition P-T dropped** (training on B2's decisions). | B2 is expected to be near the random floor, so its labels would teach little, and the schedule has no room for another training chain. |
| 6 | **Stretch condition S (PPO) dropped**; reinforcement learning is out of scope (§4). | The team decided on 4 Nov to keep all results to behaviour cloning. |
| 7 | **B3** is specified: a frame-stacked CNN with FiLM on the last three symbolic views and the two actions between them, without the heading (§6.3). | It sees the same window as the GLAM prompt, so B3 and the head differ in the model, not in the information. |
| 8 | RQ1 names **B3** explicitly, and H1 adds a hypothesis about it (§3). | B3 is the comparison for "effective and efficient": a policy that needs no LLM at all. |
| 9 | **Selection rule**: each run keeps the epoch with the lowest validation cross-entropy and measures validation success once on it; configurations are compared by that success (§6.4). | Every validation decision runs the LLM, so measuring success after every epoch of every configuration would take days of GPU time. The rule is the same for B3 and the head. |
| 10 | **Attention pooling** is run only at the best layers of the last-token / mean sweep (§6.3): planned as the two best; on the easy tier layers 12, 16 and 20 tied (best configurations 0.692–0.695 val success, within noise), so all three are used. Layer 28 is added as a control, so H2 (intermediate vs final layer) is also tested with attention pooling. | It needs every token's vector, which cannot be cached for all layers (about 45 GB per layer for all demonstrations); last-token and mean pooling are cached for every layer. Choosing two of three tied layers would have been arbitrary. |
| 11 | **Learning-rate decay and early stopping** for every trained condition, with one shared initial learning rate chosen on validation (§6.4). | A fixed schedule (20 epochs) under-trained both the head and B3; with early stopping every model trains until validation cross-entropy stops improving. One shared value keeps the protocol identical. |
| 12 | **The hard tier is not swept** (§6.3): it trains only B3 and the head configuration chosen on the easy tier, unchanged, plus the mean-pooling head at the same layer (P-L12-mean-h1) as a pooling check and the attention head at the final layer (P-L28-attn-h1) as the H2 control. | The sweep answers RQ2 on the easy tier; on the hard tier the question is whether the easy-tier comparison with B3 holds on longer tasks. Fixing the configuration on the easy tier also avoids re-tuning per tier. Each hard-tier head run costs about twice an easy one. |
| 13 | Risks, compute, timeline and team roles updated (§9–12). | Measured costs and the remaining schedule. |

## References

- Ahn, M., et al. (2022). Do As I Can, Not As I Say: Grounding Language in Robotic Affordances. *arXiv:2204.01691*.
- Carta, T., Romac, C., Wolf, T., Lamprier, S., Sigaud, O., & Oudeyer, P.-Y. (2023). Grounding Large Language Models in Interactive Environments with Online Reinforcement Learning. *ICML 2023*.
- Chevalier-Boisvert, M., et al. (2019). BabyAI: A Platform to Study the Sample Efficiency of Grounded Language Learning. *ICLR 2019*.
- Huang, W., Abbeel, P., Pathak, D., & Mordatch, I. (2022). Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents. *ICML 2022*.
- Kim, M. J., Finn, C., & Liang, P. (2025). Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success. *RSS 2025*.
- Li, K., Hopkins, A. K., Bau, D., Viégas, F., Pfister, H., & Wattenberg, M. (2023). Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task. *ICLR 2023*.
- Li, S., et al. (2022). Pre-Trained Language Models for Interactive Decision-Making. *NeurIPS 2022*.
- Paglieri, D., et al. (2025). BALROG: Benchmarking Agentic LLM and VLM Reasoning on Games. *ICLR 2025*.
- Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Courville, A. (2018). FiLM: Visual Reasoning with a General Conditioning Layer. *AAAI 2018*.
- Qwen Team (2025). Qwen3 Technical Report. *arXiv:2505.09388*.
- Ross, S., Gordon, G., & Bagnell, D. (2011). A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. *AISTATS 2011*.
- Tan, W., Zhang, W., Liu, S., Zheng, L., Wang, X., & An, B. (2024). True Knowledge Comes from Practice: Aligning LLMs with Embodied Environments via Reinforcement Learning. *ICLR 2024*.
- Zhou, Z., Hu, B., Zhao, C., Zhang, P., & Liu, B. (2024). Large Language Model as a Policy Teacher for Training Reinforcement Learning Agents. *IJCAI 2024*.

*Verify all references against the original sources before submission.*
