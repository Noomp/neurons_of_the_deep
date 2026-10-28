# very deep neurotica

SC4001 group project: *Reading Actions from Frozen Language Models* (see [project_proposal.md](project_proposal.md)).

## Layout

```
grp_proj/
├── project_proposal.md
├── config.py            all settings: environment, eval sets, data, evaluation, training, model sizes
├── environment.yml      conda env ("neurons") for the whole project
├── pyproject.toml       installs babytext, models and config (pip install -e .)
├── babytext/            the environment, self-contained: package, tools, tests, docs
│   └── README.md        environment documentation: tasks, eval sets, prompts, demos, tools
├── models/              experiment code
│   └── symbolic_agent.py  B3: frame-stacked CNN + FiLM on the symbolic grid
├── tests/               experiment tests
└── data/, outputs/      generated demos, figures and results (gitignored)
```

Environment code goes in `babytext/`; experiment code (models, training, evaluation of agents) goes in `models/` and
any new top-level folders, with tests in `tests/`.

## Setup

```bash
conda env create -f environment.yml   # creates the "neurons" env
conda activate neurons
pip install -e .                      # installs babytext, models and config (editable)
python -m pytest                      # environment + experiment tests, ~70 s
python -m babytext.tools.collect_demos   # expert demonstrations -> data/demos/
```

Always work inside the activated env (or `conda run -n neurons ...`). The conda PyTorch build loads its DLLs from the
env's `Library/bin`, which is only on the search path when the env is active; calling the env's `python.exe` directly
makes `import torch` fail with "DLL load failed". Run commands from the repo root: data and output paths in
`config.py` are relative to it.

## Configuration

All settings live in [config.py](config.py) (importable as `config`); edit the defaults there to change them project-wide.

| section | contents |
|---|---|
| `ENV` | room size, distractors, view size, time limit, held-out goals, invented words, prompt window |
| `EVAL_SETS` | the six eval sets: seed split and how each one's levels differ from `ENV` |
| `DATA` | demo counts per task, data and output folders |
| `EVAL` | episodes per evaluation |
| `TRAIN` | training protocol from the proposal: seeds, demo-count curve, backbone, layers, pooling, DAgger |
| `SYMBOLIC` | sizes of the B3 symbolic agent |

Changing `ENV` or `EVAL_SETS` changes the generated episodes, so re-collect demos and re-cache hidden states afterwards.
`data/demos/summary.json` records the full config each dataset was generated with.
