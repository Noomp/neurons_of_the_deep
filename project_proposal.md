# Project Proposal

## Reading Actions from Frozen Language Models: Lightweight Action Heads as Policies for Text-Based Gridworld Agents

**Course:** SC4001 Neural Networks and Deep Learning — Group Project
**Group:** [Group number]
**Members:** [Name 1], [Name 2], [Name 3]
**Date:** 23 October 2026
**Submission deadline:** 16 November 2026, 11:59 PM

---

## 1. Context

Large language models (LLMs) are increasingly used as agents: they observe an environment, choose actions, and pursue goals over many steps. Most current LLM agents act by *generating text*, such as a tool call or a command, and frontier systems are post-trained end to end with reinforcement learning (RL) so that the text they write leads to good outcomes. This approach is flexible but expensive. Every decision requires token-by-token decoding, the generated text does not always correspond to a valid action, and adapting the policy means updating billions of parameters.

Prior work has identified three ways to turn an LLM into a policy over a fixed set of actions (Carta et al., 2023):

1. **Generate and map:** let the LLM write free text, then map it to the closest valid action (Huang et al., 2022).
2. **Score each action:** compute the LLM's log-probability of each candidate action string and normalise over the action set (Ahn et al., 2022; Carta et al., 2023; Tan et al., 2024).
3. **Action head:** attach a new network that maps the LLM's internal representation directly to a distribution over actions in a single forward pass.

GLAM (Carta et al., 2023) chose option 2 and fine-tuned the entire model with PPO in BabyAI-Text, using substantial compute. In an ablation, a pretrained model with a new action head attached to its final layer learned slowly, which the authors attributed to the final layer being specialised for next-word prediction. Related work initialised policies from pretrained language models and fine-tuned them by imitation (Li et al., 2022), or distilled an LLM's decisions into a small specialist agent (Zhou et al., 2024).

A cheaper alternative remains under-explored: **keep the LLM frozen and train only a small head on its hidden states, choosing which layer to read from and how to summarise the tokens.** This mirrors frozen-backbone transfer learning in computer vision (a new classifier on a frozen ResNet) and is motivated by probing studies showing that game state can be decoded from a sequence model's intermediate representations (Li et al., 2023). In robotics, replacing token-by-token action generation with direct action prediction has produced large speed-ups (Kim et al., 2025), suggesting practical value beyond this project.

## 2. Problem Statement

Prompted LLM agents are slow and can produce invalid actions, while fully fine-tuning LLMs as policies is computationally expensive. There is little controlled evidence on three questions:

- whether a frozen LLM's hidden states already contain enough decision-relevant information for a small trained head to act well;
- where in the network that information is most accessible, and how token representations should be pooled;
- what a fixed action head gives up, compared with language-grounded methods that can exploit the meaning of object and action names, when the environment changes.

## 3. Research Questions and Hypotheses

**Main question:** Can a lightweight action head trained on the hidden states of a frozen LLM serve as an effective and efficient policy for a text-based gridworld, and what does it trade away compared with language-grounded action selection?

| ID | Research question | Hypothesis |
|---|---|---|
| RQ1 | How does a frozen-LLM action head compare with prompted generation and log-probability scoring in success rate and inference cost? | **H1:** After behaviour cloning, the head matches or exceeds both zero-shot methods in success rate while requiring a single forward pass per decision. |
| RQ2 | Which layer and which pooling method give the most decision-relevant representation? | **H2:** Intermediate layers outperform the final layer, and learned attention pooling outperforms last-token pooling, with the largest gain when observations contain many distractors. |
| RQ3 | How robust is the head under distribution shift compared with language-grounded methods? | **H3:** The head shows a larger relative drop than log-probability scoring when object names are replaced with invented words, but a smaller drop when the number of distractors increases. |
| RQ4 (secondary) | Does it matter whether the head learns from a scripted expert or from the LLM's own zero-shot decisions (task-specific distillation)? | Exploratory; no directional hypothesis. |

## 4. Objectives

- **O1.** Build a reproducible pipeline: environment, expert demonstrations, and cached hidden-state extraction from a frozen LLM.
- **O2.** Implement baselines that share the same backbone and prompt: prompted generation with action mapping, and log-probability action scoring, plus a non-LLM agent trained on symbolic observations.
- **O3.** Train and evaluate action heads across layers and pooling methods under a controlled protocol.
- **O4.** Measure generalisation under controlled distribution shifts and report inference and training cost.
- **O5.** Analyse failure cases qualitatively and quantitatively.
- **O6.** Deliver a report and commented, reproducible code that meet the course requirements.

**Out of scope:** pretraining or full fine-tuning of the LLM, multi-room or long-horizon tasks, and environments beyond BabyAI-Text.

## 5. Related Work

| Work | Approach | Relation to this project |
|---|---|---|
| Carta et al. (2023), GLAM | Flan-T5 as policy in BabyAI-Text; log-probability action scoring; full PPO fine-tuning | Closest work. Provides the environment and prompt template. Its action-head ablation used the final layer with full fine-tuning; we study frozen layers instead. |
| Huang et al. (2022) | Frozen LLM generates plans; phrases mapped to valid actions by sentence-embedding similarity | Basis for our generate-and-map baseline. |
| Li et al. (2022), LID | Policy initialised from a pretrained LM, trained by imitation in BabyAI and VirtualHome | Found input format mattered little while LM pretraining mattered; relevant to RQ3. |
| Tan et al. (2024), TWOSOME | Log-probability scoring of valid actions with normalisation, trained with RL | Source of length normalisation for our scoring baseline. |
| Zhou et al. (2024), LLM4Teach | Small student agent distils an LLM teacher, then shifts to RL loss | Related to RQ4 (LLM-teacher labels). |
| Li et al. (2023), Othello-GPT | Probes recover board state from a sequence model's hidden states | Motivates the layer-wise analysis in RQ2. |
| Kim et al. (2025), OpenVLA-OFT | Parallel decoding and direct action regression in vision-language-action models | Evidence that bypassing token generation improves speed in practice. |
| Paglieri et al. (2025), BALROG | Benchmark of LLM and VLM agents on games including BabyAI | Reference for zero-shot LLM agent performance on related environments. |

## 6. Methodology

### 6.1 Environment and Tasks

We use **BabyAI-Text** (Carta et al., 2023), a text-only version of BabyAI (Chevalier-Boisvert et al., 2019). The agent receives a goal and a text description of its partial view, with object positions given in steps relative to the agent, and chooses one of six actions: *turn left, turn right, go forward, pick up, drop, toggle*.

- **Tasks:** *Go to <object>* and *Pick up <object>* in a single room with 8 distractors.
- **Prompt:** the GLAM template, containing the goal, the last three observations, and the last two actions, used identically by every LLM-based condition.
- **Data splits:** environment seeds are partitioned into training (demonstrations), validation (model selection), and test (final evaluation, used once).
- **Shifted test sets:** (a) 16 distractors instead of 8; (b) object and colour names replaced by invented words; (c) goal objects held out from all training goals.
- **Fallback:** if the BabyAI-Text package cannot be installed on our compute, we will write an equivalent text wrapper over MiniGrid.

### 6.2 Backbone

A small, openly available instruction-tuned LLM (approximately 0.5–1.5B parameters, for example from the Qwen2.5 family) in half precision, kept frozen throughout. If time allows, a second model size will be tested to check whether conclusions hold across scale.

### 6.3 Experimental Conditions

| ID | Method | Trainable parameters | LLM passes per decision |
|---|---|---|---|
| B1 | Prompted generation; output mapped to the nearest valid action by sentence-embedding similarity | 0 | 1 prefill + decoding |
| B2 | Log-probability scoring of the six action strings, with length normalisation | 0 | One per action (shared prefix) |
| B3 | Symbolic behaviour cloning: small CNN/LSTM on BabyAI's symbolic grid observation, trained on the same demonstrations | < 1M | None |
| **P** | **Frozen LLM + action head (proposed).** Variants: layer ∈ {every 4th layer, final}; pooling ∈ {last token, mean, learned attention}; MLP with 1–2 hidden layers | ~0.1–2M | 1 prefill |
| P-T | Same as P, trained on B2's decisions instead of expert actions (RQ4) | ~0.1–2M | 1 prefill |
| S | *Stretch:* P fine-tuned with PPO from environment reward, LLM still frozen | ~0.1–2M | 1 prefill |

B3 isolates the value of LLM representations: it sees the same information and demonstrations without language pretraining.

### 6.4 Training

- **Demonstrations:** collected with BabyAI's built-in procedural expert, which solves these tasks reliably.
- **Behaviour cloning:** cross-entropy loss on expert actions. Because the LLM is frozen, hidden states are computed once per visited state and cached for the layers under study, so training each head takes minutes and the layer sweep is inexpensive. Full token sequences are cached only for the layers used with attention pooling.
- **Sample efficiency:** heads are trained on increasing numbers of demonstration episodes to produce learning curves.
- **Compounding errors:** if behaviour cloning drifts off the expert's state distribution, we will add a DAgger-style step (Ross et al., 2011) in which the expert relabels states visited by the trained head.

### 6.5 Evaluation

**Metrics**

- Success rate on held-out test episodes (primary metric)
- Episode length and BabyAI reward
- Invalid-output rate (B1)
- Inference latency per decision (median ms, batch size 1, same GPU for all conditions)
- Training cost (GPU-minutes) and number of trainable parameters
- Generalisation gap: success on each shifted set minus in-distribution success
- Success as a function of the number of demonstrations

**Protocol**

- At least 3 training seeds per head configuration, reported as mean with 95% confidence intervals.
- Layer, pooling, and hyperparameters are selected on the validation split only; the test and shifted sets are evaluated once.
- All LLM-based conditions share the same backbone, prompt, and demonstrations.
- All conditions are reported, including negative results.

### 6.6 Failure Analysis

Failed test episodes will be categorised (for example looping or oscillating between actions, getting stuck behind distractors, approaching the wrong object, picking up or dropping at the wrong time, and drifting after an early mistake). We will report per-task breakdowns, action confusion matrices, and representative trajectories rendered with BabyAI's visualisation tools.

## 7. Expected Contributions

1. A controlled comparison of ways to read actions from the same frozen LLM under matched prompts and data.
2. A layer-wise and pooling analysis showing where decision-relevant information is accessible, directly testing GLAM's explanation for the slow learning of final-layer action heads.
3. A characterisation of the trade-off between inference cost and robustness to distribution shift.
4. A practical, single-GPU recipe for lightweight LLM-based game agents.

Negative results are also informative. For example, finding that heads collapse under name substitution, or that no layer outperforms the symbolic baseline, would directly answer the research questions.

## 8. Alignment with Assessment Criteria

| Criterion (weight) | How the project addresses it |
|---|---|
| Experiments and results (40%) | Baselines sharing the same backbone; layer and pooling ablations; multiple seeds with confidence intervals; strict validation/test separation; shifted test sets; failure analysis. |
| Report presentation (30%) | Clear question-driven structure; learning curves, layer-sweep plots, and cost-versus-success charts; all required sections, including the AI Use Statement. |
| Novelty (20%) | Frozen-backbone analysis across layers and pooling methods, which prior work did not isolate; explicit cost-versus-robustness trade-off. |
| Peer review (10%) | Clearly divided responsibilities, with every member able to explain the full pipeline. |

## 9. Risks and Mitigation

| Risk | Likelihood | Mitigation |
|---|---|---|
| BabyAI-Text dependencies conflict with available environment | Medium | Write a minimal text wrapper over MiniGrid that follows the published description format. |
| GPU limits or Colab session interruptions | Medium | Small frozen model, cached hidden states, regular checkpoints. |
| Zero-shot baselines perform near random | Medium | Still a valid reference point; report against a random-action baseline and B3. |
| Behaviour cloning suffers from compounding errors | Medium | DAgger-style relabelling; analyse drift explicitly in the failure analysis. |
| PPO stretch goal is unstable | High | Kept optional; core results do not depend on it. |
| Time pressure near the deadline | Medium | Fixed milestones below; report outline started in week 1. |

## 10. Timeline

| Period | Milestones |
|---|---|
| Week 1: 23–29 Oct | Environment set up and verified; expert demonstrations collected; B1 and B2 evaluated; hidden-state caching pipeline working; report outline drafted. |
| Week 2: 30 Oct–5 Nov | B3 trained; action heads trained across layers and pooling methods; validation-based model selection; demonstration-count curves. **Milestone: core results by 5 Nov.** |
| Week 3: 6–12 Nov | Shifted test sets and latency benchmarks; RQ4 label-source ablation; stretch goal if on schedule; failure analysis. |
| Final days: 13–16 Nov | Report writing and figures; code clean-up and reproduction README; AI Use Statement; submission before 16 Nov, 11:59 PM. |

## 11. Team Roles

| Member | Primary responsibility |
|---|---|
| [Name 1] | Environment, expert demonstrations, baselines B1–B3 |
| [Name 2] | LLM feature extraction, caching, action-head training and sweeps |
| [Name 3] | Evaluation protocol, generalisation tests, failure analysis, figures |

All members contribute to the report and review each other's code so that everyone can explain every component.

## 12. Resources and Reproducibility

- **Compute:** course-provided GPUs or Google Colab, with checkpoints saved regularly.
- **Software:** PyTorch, Hugging Face Transformers, BabyAI-Text / MiniGrid.
- **Reproducibility:** fixed random seeds; configuration files for every experiment; a README covering dependencies, data preparation (demonstration collection and hidden-state caching), and commands to reproduce each table and figure.
- **Acknowledgement:** all reused code (BabyAI-Text, GLAM prompt template) and pretrained models will be credited, and our own contributions identified.

## 13. AI Use Statement (draft)

*[Revise to reflect actual use before submission.]* We used Claude (Anthropic) to brainstorm project directions, search for and summarise related literature, and draft this proposal. We verified every reference against the original paper, and the team chose the research question, designed the experiments, and will run and interpret all experiments.

## References

- Ahn, M., et al. (2022). Do As I Can, Not As I Say: Grounding Language in Robotic Affordances. *arXiv:2204.01691*.
- Carta, T., Romac, C., Wolf, T., Lamprier, S., Sigaud, O., & Oudeyer, P.-Y. (2023). Grounding Large Language Models in Interactive Environments with Online Reinforcement Learning. *ICML 2023*.
- Chevalier-Boisvert, M., et al. (2019). BabyAI: A Platform to Study the Sample Efficiency of Grounded Language Learning. *ICLR 2019*.
- Huang, W., Abbeel, P., Pathak, D., & Mordatch, I. (2022). Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents. *ICML 2022*.
- Kim, M. J., Finn, C., & Liang, P. (2025). Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success. *RSS 2025*.
- Li, K., Hopkins, A. K., Bau, D., Viégas, F., Pfister, H., & Wattenberg, M. (2023). Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task. *ICLR 2023*.
- Li, S., et al. (2022). Pre-Trained Language Models for Interactive Decision-Making. *NeurIPS 2022*.
- Paglieri, D., et al. (2025). BALROG: Benchmarking Agentic LLM and VLM Reasoning on Games. *ICLR 2025*.
- Ross, S., Gordon, G., & Bagnell, D. (2011). A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. *AISTATS 2011*.
- Tan, W., Zhang, W., Liu, S., Zheng, L., Wang, X., & An, B. (2024). True Knowledge Comes from Practice: Aligning LLMs with Embodied Environments via Reinforcement Learning. *ICLR 2024*.
- Zhou, Z., Hu, B., Zhao, C., Zhang, P., & Liu, B. (2024). Large Language Model as a Policy Teacher for Training Reinforcement Learning Agents. *IJCAI 2024*.

*Verify all references against the original sources before submission.*
