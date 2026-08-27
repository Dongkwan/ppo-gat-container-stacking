# PPO-GAT: Container Stacking Benchmark and Results

Benchmark instances and complete experimental results for:

> **Graph-Structured Deep Reinforcement Learning with a Per-Node Policy for
> Container Stacking Optimization in Smart Port Yards**
> Dong Kwan Kim, Young-Wuk Lee, Jung Sik Jeong — *IEEE Access*, 2026.

This repository is a **data release**. It contains the held-out evaluation
instances every number in the paper was measured on, the per-run results
behind every table and figure, and the figures themselves. Every value in
the paper can be recomputed from `results_summary.csv` alone.

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
PPO-Flat                  43.348  0.636      3
PPO-GAT (ours)             0.637  0.130      3
PPO-MLP-Pointer            7.131  6.239      3
```

That is the `B2K4` column of Table 5, reproduced exactly.

The `paper_ref` column names the table or figure each row backs, so you can
go from any printed number straight to its runs.

---

## Layout

```
results_summary.csv        121 runs, one row each — every table and figure
robustness_summary.csv     60 rows (4 methods x 3 seeds x 5 perturbations) — Fig. 3
learning_curves_ma50.csv   63 training curves, 50-episode moving average — Fig. 2
scenario.json              yard scenario parameters and their provenance
benchmark_instances/       the 300 held-out evaluation voyages per configuration
figures/                   the four figures as published (vector PDF)
results_raw.zip            133 raw per-run JSON files (the source of the CSVs)
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
| `experiment` | `main`, `ablation_v5`, or `sensitivity` |
| `method` | raw run identifier |
| `paper_label` | the label this run appears under in the paper |
| `paper_ref` | the table or figure it backs |
| `config` | `B1K1`, `B1K3`, `B2K4`, `B3K4` (blocks x bays) |
| `blocks`, `bays` | `B` and `K` for that configuration |
| `seed` | training seed: 42, 123, or 7 (empty for the deterministic greedy rule) |
| `beta1`, `beta2`, `beta3` | reward weights — populated only for the sensitivity sweep |
| `rehandling_mean`, `rehandling_std` | rehandlings per voyage over 300 evaluation voyages |
| `travel_mean`, `travel_std` | normalized RTG travel distance per voyage; empty where a method does not model travel |
| `n_steps`, `n_eval` | training budget and evaluation voyage count |
| `train_seconds` | wall-clock training time |
| `source_file` | the run file inside `results_raw.zip` this row came from |

Two different standard deviations are easy to confuse here:

- `rehandling_std` in the CSV is the spread **within one run, across the 300
  evaluation voyages**.
- The `±` values printed in the paper are the spread **across the three
  training seeds** — the standard deviation of `rehandling_mean` over a
  seed group, computed with `ddof=0`.

Use `ddof=0` as in the Quick start; pandas defaults to `ddof=1`, which
inflates a 3-seed standard deviation by a factor of `sqrt(3/2)` ≈ 1.22 and
will not match the paper.

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
only within it. A container placed on top of a lighter one causes a
**rehandling** during vessel loading, since the heavier container must be
loaded first. The objective is to minimize rehandlings per voyage, with a
secondary term for crane travel distance.

**Reporting on this benchmark.** Report mean rehandlings per voyage over all
300 voyages of a configuration, and state your training budget. The paper's
comparison point is `B2K4`: the greedy rule scores 2.01, and PPO-GAT scores
0.64 ± 0.13 across three seeds at 3×10⁵ environment steps.

---

## Experimental protocol

| | |
|---|---|
| Yard | `S = 5` stacks/bay, `T = 5` tiers/stack, `W = 5` weight classes |
| Utilization | 0.6 of yard capacity per voyage |
| Reward weights | `(β₁, β₂, β₃) = (0.6, 0.3, 0.1)` |
| Training budget | 3×10⁵ environment steps (MC Q-learning: 2×10⁷ episodes) |
| Training seeds | 42, 123, 7 |
| Evaluation | 300 held-out voyages, generated with seed 99, disjoint from training |

Every method sees the identical 300 evaluation voyages for a given
configuration. The evaluation set is fixed across methods and seeds.

---

## Scope of these results

The paper reports a proof of concept at a representative yard scale, and the
data here carries the same limits:

- **Three training seeds per cell.** Several comparisons in the reward
  sensitivity sweep are not statistically separable at this seed count; the
  paper says so where it applies, and the per-seed rows are in the CSV so you
  can check any comparison yourself.
- **Three-block yards are unsolved.** No method in the paper, including
  PPO-GAT, reaches the greedy floor on `B3K4`. The runs are included.
- **Fixed budget.** Results compare sample efficiency at 3×10⁵ steps. No
  budget sweep was run, so these numbers do not show that the flat baselines
  cannot converge given more experience.

## What is not included

- **Source code.** The environment, agents, and training scripts are not part
  of this release. The benchmark instances above are self-contained, so a new
  solver can be evaluated on the same voyages without them.
- **A superseded ablation run.** An earlier ablation set (15 runs, built on a
  hierarchical base whose "full" configuration scores 34.63 rather than 0.64)
  was excluded to avoid shipping two contradictory "full" rows. The paper's
  Table 6 comes from the `ablation_v5` runs included here.
- **Single-bay training-curve arrays** from early runs, superseded by the
  curves in `results_raw.zip`.

## Citation

GitHub reads `CITATION.cff` in this repository, so the "Cite this repository"
button produces the entry below in BibTeX or APA.

```bibtex
@article{Kim2026PPOGAT,
  author  = {Kim, Dong Kwan and Lee, Young-Wuk and Jeong, Jung Sik},
  title   = {Graph-Structured Deep Reinforcement Learning with a Per-Node
             Policy for Container Stacking Optimization in Smart Port Yards},
  journal = {IEEE Access},
  year    = {2026}
}
```

## License

[CC BY 4.0](LICENSE) — free to share and adapt with attribution.
