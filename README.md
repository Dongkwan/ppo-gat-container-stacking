# PPO-GAT: Container Stacking Benchmark and Results

Benchmark instances and complete experimental results for:

> **A Tight Guarantee for Online Container Stacking: A Strong Baseline for Reinforcement Learning**
> Dong Kwan Kim, Young-Wuk Lee, Jung Sik Jeong — *IEEE Access*, 2026.

This repository is a **data release**. It contains the held-out evaluation
instances every number in the paper was measured on and the per-run results
behind every table and figure. Every value in the paper's tables can be
recomputed from `results_summary.csv` and the files in `analysis/`; the
per-voyage statistics need the raw JSON files in `results_raw.zip`.

**Numbering.** Table and figure numbers in this README and in the `paper_ref`
column follow an earlier version of the manuscript. The manuscript calls
PPO-MLP-Pointer PPO-PerNode, PPO-MLP-FlatHead PPO-FlatHead, and PPO-GAT
PPO-PerNode-GAT.

---

## Quick start

Every headline number lives in one file:

```python
import pandas as pd

df = pd.read_csv("results_summary.csv")

# Table 5, two-block yard: PPO-GAT vs. the flat baselines
tbl = df[(df.experiment == "main") & (df.config == "B2K4")]
print(tbl.groupby("paper_label").rehandling_mean
         .agg(["mean", lambda s: s.std(ddof=0), "count"])
         .rename(columns={"<lambda_0>": "std"}).round(3))
```

```
                            mean    std  count
paper_label
DQN                       43.816  1.733      3
Greedy rule                2.007  0.000      1
H-PPO-GAT (hierarchical)  34.629  2.287      3
PPO-Flat                  43.327  1.543      5
PPO-GAT (ours)             0.850  0.458     10
PPO-MLP-Pointer            6.339  6.148     10
```

That is the `B2K4` column of Table 5, reproduced exactly.

The `paper_ref` column names the table or figure each row backs, so you can
go from any printed number straight to its runs.

**Terminology.** The paper's measure is *blocking placements per voyage*, a
placement-time proxy for rehandling and a lower bound on relocations under
heavier-first retrieval. The column and JSON key names keep their
original form (`rehandling_mean`, `rehandling_ma50`) and hold this measure.

---

## Layout

```
results_summary.csv        233 rows, one per run or budget milestone — every table and figure
robustness_summary.csv     60 rows (4 methods x 3 seeds x 5 perturbations) — earlier manuscript only
learning_curves_ma50.csv   123 training curves, 50-episode moving average
behavior_summary.csv       behavior measures of the saved policies
analysis/                  rule evaluations and decision analyses (no training involved)
scenario.json              yard scenario parameters and their provenance
benchmark_instances/       the 300 held-out evaluation voyages per configuration
results_raw.zip            246 raw per-run JSON files (the source of the CSVs)
```

The three CSVs are the working interface and contain every number in the
paper. `results_raw.zip` is the unaggregated provenance behind them — one
JSON per run, each carrying its full unsmoothed training curve:

```python
import json, zipfile

with zipfile.ZipFile("results_raw.zip") as z:
    run = json.load(z.open("results/PPO-GAT_B2K4_s42.json"))
print(run.keys())
# rehandling_mean, rehandling_std, travel_mean, travel_std,
# final_train_rehandling, curve, train_seconds, n_steps, n_eval
```

### `results_summary.csv`

| column | meaning |
|---|---|
| `experiment` | `main`, `ablation_v5`, `topology`, `sensitivity`, or `budget` |
| `method` | raw run identifier |
| `paper_label` | the label this run appears under in the paper |
| `paper_ref` | the table or figure it backs |
| `config` | `B1K1`, `B1K3`, `B2K4`, `B3K4` (blocks x bays) |
| `blocks`, `bays` | `B` and `K` for that configuration |
| `seed` | training seed: 42, 123, or 7; on `B2K4`, also 1 and 2 for PPO-Flat, and 1, 2, 3, 5, 11, 13, 17 for PPO-MLP-Pointer and PPO-GAT (empty for the deterministic greedy rule) |
| `beta1`, `beta2`, `beta3` | reward weights — populated only for the sensitivity sweep |
| `rehandling_mean`, `rehandling_std` | blocking placements per voyage over 300 evaluation voyages |
| `travel_mean`, `travel_std` | normalized RTG travel distance per voyage; empty where a method does not model travel |
| `n_steps`, `n_eval` | training budget and evaluation voyage count |
| `train_seconds` | wall-clock training time of the first run of that cell. A later re-run reproduces its values but not its time, so the `retr_*` files can differ, and times depend on batch load |
| `source_file` | the run file inside `results_raw.zip` this row came from |
| `relocation_mean`, `relocation_std` | realized relocations per voyage under the retrieval rule below; filled for the rows of Table 6 |
| `cross_block_moves_total` | relocations that had to leave their block, summed over 300 voyages; zero in every run |

Two different standard deviations are easy to confuse here:

- `rehandling_std` in the CSV is the spread **within one run, across the 300
  evaluation voyages**.
- The `±` values printed in the paper are the spread **across the training
  seeds** (as stated in the caption of Table 5: ten or five for some `B2K4` entries, otherwise three) — the standard deviation of `rehandling_mean` over a
  seed group, computed with `ddof=0`.

Use `ddof=0` as in the Quick start; pandas defaults to `ddof=1`, which
inflates a 3-seed standard deviation by a factor of `sqrt(3/2)` ≈ 1.22 and
will not match the paper.

### Added runs

The `graph_random` and `graph_full` runs replace the crane graph with a random
graph (62 of 780 stack pairs, a different draw per seed) or a fully connected
graph, with edge weights set to zero. Their raw JSON files record `graph_kind`,
`graph_seed`, and a `graph` object with edge counts, connected components,
isolated stacks, and cross-block pairs. Added runs also record library
`versions` and `torch_seeded`.

**PPO-Flat.** The original runs at seeds 42, 123, and 7 did not seed PyTorch. The
CSVs now hold the seeded re-runs (`flatseed_*`). The unseeded JSON files stay in
`results_raw.zip` for comparison (`PPO-Flat_<config>_s<seed>.json`). `B2K4` seeds 1
and 2 were seeded from the start.

**Ten seeds.** The `retr_*` files hold ten seeds of PPO-GAT and PPO-MLP-Pointer on
`B2K4`. The first five repeat stored runs and reproduce them exactly. Each file adds
`relocation_mean`, `blocking_per_voyage`, and `relocations_per_voyage`.

**Budget.** The `budget3_*` files continue PPO-GAT, PPO-MLP-Pointer, and PPO-Flat on
`B3K4` to 10⁶ steps, with evaluations at 3×10⁵, 6×10⁵, and 10⁶ under the `evals` key. The partial voyage at each
milestone is dropped. The 3×10⁵ values equal the stored runs. Their curves are in
`learning_curves_ma50.csv` as methods `PPO-GAT-1M`, `PPO-MLP-Pointer-1M`, and
`PPO-Flat-1M`. DQN and H-PPO-GAT were not extended. The `budget2_*` files do the same for the
ten PPO-MLP-Pointer seeds on `B2K4`, whose curves are under `PPO-MLP-Pointer-1M`, config `B2K4`.
PPO-GAT was not extended on `B2K4`.

**Behavior of saved models.** `behavior_summary.csv` has one row per saved model (108 rows, including greedy), on the
300 held-out voyages without training. `weight_response` is the share of decisions whose chosen stack changes if the
arriving class is replaced (mean over the other four classes). `open_loop` is the share of voyages that take the modal
stack at each time step. The focused manuscript calls PPO-MLP-Pointer PPO-PerNode, PPO-MLP-FlatHead PPO-FlatHead, and
PPO-GAT PPO-PerNode-GAT.

**Rules and decision analyses.** `analysis/` holds, with no training involved: `bestfit.json` and `bestfit2.json` (best-fit and
tie-break rules, the check of the guarantee, class error, two-axis cost, the skewed mix, and the proportions of Lee and Cho),
`decisions.json` (forced and avoidable blocking, score tables), `rules.json` (the preference test on common states), and
`dead.json` (active units of the output layers).

**Flat-output control.** The `flathead_*` files keep the stack-wise encoder of PPO-MLP-Pointer and
use a flat output, on `B2K4` (ten seeds) and `B3K4` (three), to 10⁶ steps. Rows have `experiment` `flathead`.

**Two-axis cost.** The `twoaxis_*` files retrain on `B2K4` (seeds 42, 123, 7) with crane cost
(S|Δk|+|Δj|)/(KS-1) in the reward and the travel measure. Rows have `experiment` `twoaxis`, and curves
use the method suffix `-TwoAxis`. Their `travel_mean` is not comparable with the linear runs.

**Other rules and utilization.** The `relcheck_*` files, in `results_raw.zip` only, evaluate saved models
without retraining under destination rules `avoid` (the paper's), `nearest`, and `emptiest`, at
utilization 0.6 and 0.8, under the `by_util` key.

**Retrieval rule.** Containers leave heaviest class first across the yard. Within a
class, the one with the fewest containers above it goes first, with ties to the lowest
`(b, k, j)`. Each container above it moves, top first, to a non-full stack in the same
block, preferring a stack where the move creates no new blocking, then the nearest by
`|dk| + |dj|`, then the lowest index. Each move is one relocation; a move with no room
in the block goes to another block and is counted separately. Under heavier-first
retrieval, blocking placements are a lower bound on relocations, and under this rule the
two are equal whenever a non-blocking destination exists, so their correlation is not
independent evidence for the proxy.

---

## Benchmark instances

`benchmark_instances/eval_<config>.json` holds the 300 held-out voyages that
every method in the paper was evaluated on. Each voyage is the ordered
sequence of container weight classes (1–5, higher is heavier) as they arrive
at the yard; a solver must assign each arrival to a stack.

| file | blocks | bays | yard capacity | containers/voyage | size |
|---|---|---|---|---|---|
| `eval_B1K1.json` | 1 | 1 | 25 | 15 | 10 KB |
| `eval_B1K3.json` | 1 | 3 | 75 | 45 | 27 KB |
| `eval_B2K4.json` | 2 | 4 | 200 | 120 | 71 KB |
| `eval_B3K4.json` | 3 | 4 | 300 | 180 | 107 KB |

```python
import json

inst = json.load(open("benchmark_instances/eval_B2K4.json"))
inst["voyages"][0][:10]   # [2, 3, 4, 1, 5, ...] first 10 arrivals of voyage 0
```

**Yard model.** A yard has `B` blocks; each block has `K` bays, each bay `S=5`
stacks, each stack `T=5` tiers. One RTG crane serves each block and travels
only within it. A lighter container placed above a heavier one is a **blocking
placement**, since the heavier container is loaded first and the lighter one must
be moved to reach it. The objective is to minimize blocking placements per voyage,
with a secondary term for crane travel distance. This counts placements; relocations
from a simulated retrieval are reported separately.

**Reporting on this benchmark.** Report mean blocking placements per voyage over
all 300 voyages of a configuration, and state your training budget. The paper's
comparison point is `B2K4`: the greedy rule scores 2.01, and PPO-GAT scores
0.85 ± 0.46 across ten seeds at 3×10⁵ environment steps.

---

## Experimental protocol

| | |
|---|---|
| Yard | `S = 5` stacks/bay, `T = 5` tiers/stack, `W = 5` weight classes |
| Utilization | 0.6 of yard capacity per voyage |
| Reward weights | `(β₁, β₂, β₃) = (0.6, 0.3, 0.1)` |
| Training budget | 3×10⁵ environment steps (MC Q-learning: 2×10⁷ episodes); PPO-GAT, PPO-MLP-Pointer, and PPO-Flat on `B3K4`, and PPO-MLP-Pointer on `B2K4`, also to 10⁶ |
| Training seeds | 42, 123, 7; on `B2K4`, ten seeds for PPO-GAT and PPO-MLP-Pointer and five for PPO-Flat |
| Evaluation | 300 held-out voyages, generated with seed 99, disjoint from training |

Every method sees the identical 300 evaluation voyages for a given
configuration. The evaluation set is fixed across methods and seeds.

The paper's Table 3 lists the PPO and GSSE settings, which are shared by
PPO-Flat, PPO-MLP-Pointer, PPO-GAT and H-PPO-GAT. DQN is the one
baseline with a separate configuration:

| DQN setting | Value |
|---|---|
| Q-network | 2 hidden layers, width 256, ReLU |
| Optimizer | Adam, learning rate 1×10⁻³ |
| Discount γ | 0.99 |
| Replay buffer | 50,000 transitions |
| Batch size | 512 |
| Target network update | every 1,000 steps |
| Exploration | ε-greedy, 1.0 → 0.05, exponential decay over 2×10⁵ steps |
| Training budget | 3×10⁵ environment steps (as for the PPO agents) |

---

## Scope of these results

The paper reports a proof of concept at a representative yard scale, and the
data here carries the same limits:

- **Few training seeds.** Most cells have three seeds, including the ablation,
  topology, and budget runs. Several
  comparisons are not statistically separable at these seed counts; the
  paper says so where it applies, and the per-seed rows are in the CSV so you
  can check any comparison yourself.
- **Three-block yards at the common budget.** No method reaches the greedy
  reference on `B3K4` at 3×10⁵ steps. PPO-GAT and PPO-MLP-Pointer do at 10⁶ steps, at
  longer travel, over three seeds, without a clear difference in blocking. PPO-Flat
  does not improve up to 10⁶ steps.
- **Two-block budget.** All ten PPO-MLP-Pointer seeds beat greedy on `B2K4` by 6×10⁵ steps,
  against five at 3×10⁵. PPO-GAT was not extended there, so these are not same-step comparisons.
- **Fixed budget between methods.** Results compare sample efficiency at 3×10⁵
  steps, and only three methods on `B3K4` and PPO-MLP-Pointer on `B2K4` were extended, to 10⁶ steps, so these
  numbers do not show that the flat baselines cannot converge given more experience.
- **One retrieval rule.** Relocations were measured under one rule at utilization
  0.6, without crane travel or time during retrieval.

## What is not included

- **Source code.** The environment, agents, and training scripts are not part
  of this release, and neither are the trained models. The benchmark instances above are self-contained, so a new
  solver can be evaluated on the same voyages without them.
- **A superseded ablation run.** An earlier ablation set (15 runs, built on a
  hierarchical base whose "full" configuration scores 34.63 rather than 0.64)
  was excluded to avoid shipping two contradictory "full" rows. The paper's
  Table 7 comes from the `ablation_v5` runs included here.
- **Single-bay training-curve arrays** from early runs, superseded by the
  curves in `results_raw.zip`.

## Citation

GitHub reads `CITATION.cff` in this repository, so the "Cite this repository"
button produces the entry below in BibTeX or APA.

```bibtex
@article{Kim2026PPOGAT,
  author  = {Kim, Dong Kwan and Lee, Young-Wuk and Jeong, Jung Sik},
  title   = {A Tight Guarantee for Online Container Stacking: A Strong Baseline for Reinforcement Learning},
  journal = {IEEE Access},
  year    = {2026}
}
```

## License

[CC BY 4.0](LICENSE) — free to share and adapt with attribution.
