

# Reinforcement Learning for *Take Six!* (6 nimmt!)

Course project for **Reinforcement Learning**

This project studies Reinforcement Learning (RL) on the card game **Take Six!** (*6 nimmt!*). We build a reproducible Gymnasium-compatible environment for the game, benchmark several RL approaches, analyze the role of feature engineering, and introduce a permutation-invariant architecture (**Row DeepSets**) to better handle the symmetry of the game board. We further extend the setting by replacing random opponents with stronger frozen policies and study cross-environment generalization.

## Project summary

Our work proceeds in three main stages:

1. **Environment design**
   - We implement a custom Gymnasium environment for *Take Six!*
   - The environment uses a compact feature-based observation instead of raw images
   - The implementation is reproducible and supports action masking

2. **Algorithm comparison and representation analysis**
   - We compare **Maskable PPO**, **DQN**, **QRDQN**, and **Heuristic + RL**
   - We perform a full-factorial ablation study on three observation design choices:
     - continuous normalization
     - row sorting
     - difference features (`add_diff`)
   - We show that value-based methods outperform PPO in this task, and that representation design is critical

3. **Structural modeling and harder environment extension**
   - We introduce a **Row DeepSets** encoder to handle the row symmetry of the board in a principled way
   - We build a harder environment with **frozen stronger opponents**
   - We perform **cross-environment evaluation** to study generalization

## Main takeaways

- **Value-based RL is better suited to this task than PPO** in our current environment and training budget.
- **QRDQN** is the strongest baseline among the tested methods.
- The main gains come from **continuous normalization** and **row symmetry handling**.
- `add_diff` is useful only conditionally, especially when combined with normalized continuous input.
- A **Row DeepSets** permutation-invariant encoder improves over the best handcrafted QRDQN baseline.
- Policies trained against random opponents do **not** transfer well to stronger frozen opponents; retraining in the harder environment is necessary.

---

# Repository structure

```text
sixnimmt/
├─ checkpoints/
│  ├─ demo_models/
│  │  ├─ qrdqn_rowdeepsets_fixedopp_best/
│  │  └─ qrdqn_rowdeepsets_old_best/
│  ├─ frozen_opponents/
│  │  ├─ heuristic_seed1/
│  │  ├─ ppo_seed1/
│  │  └─ ppo_seed2/
│  └─ outputs/
│     ├─ qrdqn_rowdeepsets_fixedopp_seed0 ... seed4/
│     └─ qrdqn_rowdeepsets_seed0 ... seed4/
│
├─ results/
│  ├─ ablation/
│  ├─ cross_eval/
│  ├─ new_env/
│  └─ old_env/
│
├─ scripts/
│  ├─ train_qrdqn_row_deepsets.py
│  ├─ eval_qrdqn_row_deepsets.py
│  ├─ train_qrdqn_row_deepsets_fixed_opp.py
│  ├─ eval_qrdqn_row_deepsets_fixed_opp.py
│  ├─ eval_old_model_on_fixedopp_env.py
│  ├─ eval_fixedopp_model_on_old_env.py
│  ├─ train_maskable_ppo.py
│  ├─ eval_policy.py
│  ├─ train_heuristic_rl.py
│  ├─ eval_heuristic_rl.py
│  ├─ train_value_based.py
│  ├─ eval_value_based.py
│  └─ validate_invariants.py
│
├─ src/
│  └─ sixnimmt_env/
│     ├─ __init__.py
│     ├─ continuous.py
│     ├─ core.py
│     ├─ env.py
│     ├─ env_fixed_opponents.py
│     └─ opponent_policies.py
│
├─ requirements.txt
├─ pyproject.toml
├─ project_demo.ipynb
└─ report.pdf
```

## Directory guide

### `src/sixnimmt_env/`

Core implementation of the environments and helper modules.

- `core.py`
  Pure game logic: cards, deck, table, players, placement rules, bullhead computation.
- `env.py`
  Original environment with **3 random opponents**.
- `env_fixed_opponents.py`
  Harder environment with **3 frozen stronger opponents**.
- `continuous.py`
  Observation wrappers used by PPO / heuristic-related experiments.
- `opponent_policies.py`
  Frozen policy wrappers used in the fixed-opponent environment.
- `__init__.py`
  Package exports.

### `scripts/`

Training and evaluation entry points.

- Original environment:
  - `train_qrdqn_row_deepsets.py`
  - `eval_qrdqn_row_deepsets.py`
- Harder fixed-opponent environment:
  - `train_qrdqn_row_deepsets_fixed_opp.py`
  - `eval_qrdqn_row_deepsets_fixed_opp.py`
- Cross-environment evaluation:
  - `eval_old_model_on_fixedopp_env.py`
  - `eval_fixedopp_model_on_old_env.py`
- Baselines:
  - `train_maskable_ppo.py`
  - `eval_policy.py`
  - `train_heuristic_rl.py`
  - `eval_heuristic_rl.py`
  - `train_value_based.py`
  - `eval_value_based.py`

### `checkpoints/`

Saved model weights.

- `frozen_opponents/`
  Pretrained PPO / Heuristic+PPO opponents used in the harder environment.
- `demo_models/`
  Compact demo checkpoints for quick evaluation and notebook demonstrations.
- `outputs/`
  Per-seed best checkpoints used for the reported 5-seed results and cross-evaluation.

### `results/`

Final summarized results used in the report and the notebook.

- `old_env/`
  Main results in the original random-opponent environment.
- `ablation/`
  Full-factorial observation ablation results.
- `new_env/`
  Results for the harder fixed-opponent environment.
- `cross_eval/`
  Cross-environment evaluation results.

------

# Installation

## Option 1: minimal installation

From the repository root:

```bash
pip install -r requirements.txt
pip install -e .
```

The editable installation uses the provided `pyproject.toml` and allows imports such as:

```python
from sixnimmt_env import SixQuiPrendEnv
```

## Python version

We recommend **Python 3.9+**.

------

# Notebook

The main notebook is:

```text
project_demo.ipynb
```

This notebook is intended as a **lightweight demo and results viewer**. It does **not** retrain every model by default. Instead, it:

- demonstrates the environments,
- loads the final summarized results from `results/`,
- visualizes the main comparisons and cross-environment evaluation,
- shows how to reproduce the main experiments through the provided scripts.

The notebook is organized into the following sections:

1. Title & project overview
2. Setup
3. Environment demo
4. Main results in the original environment
5. Ablation study and representation design
6. Harder fixed-opponent environment and cross-environment evaluation
7. Reproducibility and how to run

------

# Environment design

## Original environment

The original environment models a single-hand *Take Six!* game as a fixed-length RL episode:

- 4 players in total
- player 0 is the learning agent
- players 1–3 are random opponents
- 10 cards per player
- 4 rows on the table
- 10 steps per episode

### Observation

The observation is a structured dictionary containing:

- `player_hand`
  padded vector of the agent’s current hand
- `last_value_of_rows`
  tail value of each row
- `length_of_rows`
  current length of each row
- `table_bulls`
  total bullheads in each row

### Action

The action space is:

```python
Discrete(10)
```

The action corresponds to the index of the card to play from the sorted hand.

Because the hand shrinks over time, legal actions are handled through:

```python
env.action_masks()
```

### Reward

At each step:

```text
reward = - agent_penalty
```

So maximizing cumulative reward is equivalent to minimizing accumulated bullheads.

### Simplification

To keep the environment compact and stable for single-agent RL, we use a simplified version of the “forced eat” rule:

- when the played card is smaller than all four row tails,
- the environment automatically selects the row with the minimum bullhead sum.

This removes an additional sparse branching choice while preserving the main strategic structure.

------

## Harder fixed-opponent environment

We also provide a second environment where the three random opponents are replaced by stronger **frozen policies**:

- PPO(seed1)
- PPO(seed2)
- Heuristic+PPO(seed1)

This environment keeps:

- the same game rules,
- the same reward,
- the same agent interface,

and only changes the **opponent distribution**.

This allows us to study how the optimal learned policy changes when the environment becomes harder.

------

# Main experimental results

## 1. Original environment: algorithm comparison

We first compare several methods in the original random-opponent environment:

- Maskable PPO
- DQN
- QRDQN
- Heuristic + RL

The aggregated 5-seed summary is stored in:

```text
results/old_env/all_methods_5seed_summary.csv
```

Our main conclusion is:

- **value-based methods outperform PPO**
- **QRDQN is the strongest baseline**

------

## 2. Observation ablation study

We perform a full-factorial ablation on three observation design choices:

- continuous normalization
- row sorting
- difference features (`add_diff`)

The corresponding result files are:

```text
results/ablation/ablation_aggregate.csv
results/ablation/ablation_main_effects.csv
results/ablation/ablation_pairwise_interactions.csv
```

Main conclusions:

- continuous normalization is the most important single factor
- row sorting is consistently beneficial
- difference features are not universally helpful by themselves
- the usefulness of `add_diff` depends strongly on normalized input

------

## 3. Row DeepSets

To go beyond handcrafted row sorting, we introduce a **Row DeepSets** encoder.

Each row is represented by a compact tuple of row features and processed through a shared encoder, followed by permutation-friendly pooling. This better respects the symmetry of the board than treating rows as a fixed ordered vector.

The final original-environment result for the best model is stored in:

```text
results/old_env/qrdqn_rowdeepsets_5seed_aggregate.json
results/old_env/qrdqn_rowdeepsets_5seed_summary.csv
```

This is our strongest model in the original environment.

------

## 4. Harder fixed-opponent environment

We retrain our best backbone (**QRDQN + Row DeepSets**) in the harder fixed-opponent environment.

The final 5-seed summary is stored in:

```text
results/new_env/qrdqn_rowdeepsets_fixedopp_5seed_aggregate.json
results/new_env/qrdqn_rowdeepsets_fixedopp_5seed_summary.csv
```

------

## 5. Cross-environment evaluation

To study generalization, we evaluate:

- policies trained in the original environment on the harder environment
- policies trained in the harder environment on the original environment

The final cross-evaluation summaries are stored in:

```text
results/cross_eval/old_on_new_5seed_aggregate.json
results/cross_eval/new_on_old_5seed_aggregate.json
results/cross_eval/summary_2x2.csv
```

The 2×2 summary table is:

| train environment       | test environment        | mean return |
| ----------------------- | ----------------------- | ----------- |
| old random-opponent env | old random-opponent env | -7.3976     |
| old random-opponent env | new fixed-opponent env  | -8.5408     |
| new fixed-opponent env  | new fixed-opponent env  | -6.8140     |
| new fixed-opponent env  | old random-opponent env | -7.8450     |

Interpretation:

- policies trained against random opponents do **not** transfer well to stronger opponents
- retraining in the harder environment is necessary and effective
- the gain is mainly **environment-specific adaptation**, not universal improvement

------

# How to run the main experiments

## Original environment: QRDQN + Row DeepSets

Train:

```bash
python scripts/train_qrdqn_row_deepsets.py ...
```

Evaluate:

```bash
python scripts/eval_qrdqn_row_deepsets.py ...
```

------

## Harder fixed-opponent environment

Train:

```bash
python scripts/train_qrdqn_row_deepsets_fixed_opp.py \
  --ppo1_run_dir checkpoints/frozen_opponents/ppo_seed1 \
  --ppo2_run_dir checkpoints/frozen_opponents/ppo_seed2 \
  --heuristic_run_dir checkpoints/frozen_opponents/heuristic_seed1
```

Evaluate:

```bash
python scripts/eval_qrdqn_row_deepsets_fixed_opp.py ...
```

------

## Cross-environment evaluation

### Old model on fixed-opponent environment

```bash
python scripts/eval_old_model_on_fixedopp_env.py \
  --run_dir checkpoints/outputs/qrdqn_rowdeepsets_seed4 \
  --ppo1_run_dir checkpoints/frozen_opponents/ppo_seed1 \
  --ppo2_run_dir checkpoints/frozen_opponents/ppo_seed2 \
  --heuristic_run_dir checkpoints/frozen_opponents/heuristic_seed1
```

### Fixed-opponent-trained model on original environment

```bash
python scripts/eval_fixedopp_model_on_old_env.py \
  --run_dir checkpoints/outputs/qrdqn_rowdeepsets_fixedopp_seed2
```

------

# Baseline scripts

For completeness, the repository also includes the baseline scripts used in the report:

- PPO:
  - `scripts/train_maskable_ppo.py`
  - `scripts/eval_policy.py`
- Heuristic + RL:
  - `scripts/train_heuristic_rl.py`
  - `scripts/eval_heuristic_rl.py`
- Value-based baselines:
  - `scripts/train_value_based.py`
  - `scripts/eval_value_based.py`

------

# Demo checkpoints

For quick use in the notebook or for lightweight evaluation, we provide compact demo checkpoints:

- `checkpoints/demo_models/qrdqn_rowdeepsets_old_best/`
- `checkpoints/demo_models/qrdqn_rowdeepsets_fixedopp_best/`

These contain:

- `config.json`
- `best_model.pt`

------

# Reproducibility notes

This project intentionally separates:

- reusable environment modules in `src/`,
- experiment pipelines in `scripts/`,
- summarized outputs in `results/`,
- and presentation / visualization in `project_demo.ipynb`.

This keeps the repository easier to understand and avoids duplicating heavy training logic inside the notebook.

If you only want to inspect the final conclusions, the notebook and `results/` directory are sufficient.
If you want to rerun experiments, use the corresponding scripts in `scripts/`.

------

# Acknowledgement

All external libraries and frameworks used in this project are clearly acknowledged in the report and code. In particular, the implementation relies on:

- Gymnasium
- NumPy
- PyTorch
- Stable-Baselines3
- SB3-Contrib

------

# Final conclusion

This project shows that for *Take Six!*:

- careful **environment design** is essential,
- **representation design** matters as much as, or more than, the RL algorithm choice itself,
- and the learned policy is highly sensitive to the **opponent distribution**.

Our final conclusion is that **QRDQN + Row DeepSets** is the strongest model among all tested approaches, while stronger fixed opponents reveal an important generalization gap that requires environment-specific retraining.