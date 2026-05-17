# dbemu-bench

Four concrete benchmarks derived from a single underlying design.
Motivation and the broader strategy spectrum are in
[CONTINUOUS_AGENT.md](CONTINUOUS_AGENT.md).

| Name           | Audience                    | Shape                                  | Metric                 |
| -------------- | --------------------------- | -------------------------------------- | ---------------------- |
| `dbemu-mini`   | broad / Kaggle-style        | one dataset, one task, one metric      | cell-F1                |
| `dbemu-static` | conference, model compare   | frozen subtasks (LongBench / ARC)      | bits recovered + macro |
| `dbemu-env`    | conference, robustness      | procedural, seeded, hparam grid        | mean ± std             |
| `dbemu-train`  | from-scratch model research | env + curriculum + reward (SFT + RL)   | training library       |

## Shared core

All four benchmarks share:

- One **operation grammar** — DDL / DML / TX / QUERY.
- One **oracle** — SQLite executes the canonical state.
- One **wire format** — JSON
  `{schema, initial_state, ops, queries, answers?}`.

Anything below is what each benchmark layers on top.

---

## 1. `dbemu-mini` — Kaggle-style

**Purpose.** Drop-in eval that anyone can run in five minutes.

- **Input format.** JSON, one instance per file. Fixed schema:
  three tables, five columns each, INTEGER + TEXT only.
- **Initial state.** 100 rows per table.
- **Op stream length.** 1 000 (fixed).
- **Queries.** 50 per instance — point-read and small range-read only.
- **Output.** `predictions.json` — list of answers in query order.
- **Metric.** Cell-F1 over `(table, row_id, col) → value`, averaged
  over queries, averaged over instances. **One scalar.**
- **Dataset.** 10 000 instances; train (5 k) / public test (2.5 k) /
  private test (2.5 k). Frozen forever.
- **Env.** Pure offline. One Python scoring script, stdlib only.

No regime split. No spectrum. No strategy slot. Just one number per
submission. This is the entry-level benchmark; everything else is
optional.

---

## 2. `dbemu-static` — conference, frozen subtasks

**Purpose.** Citable comparison of off-the-shelf models across
distinct capability slices. Same shape as LongBench / ARC: a frozen
test set, a known taxonomy, a macro-average headline.

### Subtasks (5)

Picked so each isolates one failure mode:

| Subtask     | Probes                                |
| ----------- | ------------------------------------- |
| `T1-read`   | read-after-write fidelity             |
| `T5-long`   | long-range dependency (lag ≥ 100 k)   |
| `T6-hash`   | random-key hash-table stress          |
| `T7-count`  | counter-array stress                  |
| `T10-mvcc`  | snapshot read at past op index        |

### Protocol

- **Splits.** 800 train / 100 dev / 100 test per subtask. Frozen.
- **Stream lengths.** Mixed within each subtask: 1 k, 10 k, 100 k.
- **Format.** JSON only (multi-format deferred to `dbemu-env`).
- **Metric.** Normalized bits recovered per subtask; macro-average is
  the leaderboard scalar.
- **Two leaderboards.**
  - *Limited context* (8 k tokens). Models must adopt a memory
    strategy; the strategy is named as part of the submission.
  - *Unlimited context / recurrent.* No eviction; results additionally
    reported per recurrent state bit.

---

## 3. `dbemu-env` — conference, dynamic procedural env

**Purpose.** Robustness, ablation, and cliff-finding studies. The
dataset is implied by the generator + seed list, not shipped as files
(closer in spirit to Procgen than to ARC).

### Generator

Seeded procedural sampler over the full op grammar. Reproducible from
`(seed, hparams)` alone.

### Hparam grid

| Axis                  | Values                                   |
| --------------------- | ---------------------------------------- |
| stream length         | 10², 10³, 10⁴, 10⁵, 10⁶                  |
| schema width          | 1, 5, 25, 100                            |
| keyspace              | 10, 10³, 10⁵, 10⁷                        |
| churn per key         | 1, 10, 100, 1000                         |
| read/write ratio      | 0.01, 0.1, 1, 10                         |
| format                | sqlite, json, csv, flatbuffers           |
| **idle ticks / op ρ** | 0, 1, 4, 16, 64, 256                     |

The last axis is the continuous-agent axis from
[CONTINUOUS_AGENT.md](CONTINUOUS_AGENT.md).

### Protocol

- 50 seeds per (subtask, hparam point).
- Report mean ± std of bits recovered.
- Standard outputs:
  - per-axis slice plots (e.g. accuracy vs. stream length),
  - Pareto frontiers (accuracy vs. cost),
  - the continuous-agent derivative Δ_ρ.

---

## 4. `dbemu-train` — training framework (no leaderboard)

**Purpose.** Let a model learn to *be* a database emulator from
scratch — without depending on a pretrained LLM's emergent SQL skill.
This is the half of the design space that pure leaderboards cannot
measure.

### Env API (gym-like)

```python
env = DBEmuEnv(seed, hparams)
obs = env.reset()              # initial_state observation
for op in stream:
    obs, info = env.step(op)   # advances oracle, returns next obs
ans = env.query(q)             # returns ground-truth answer
```

The env wraps the SQLite oracle in a deterministic stepping interface.
`obs` is the configured state representation: full table, last op
only, idle tick, or empty.

### Data generators

- Infinite procedural stream from the same grammar as `dbemu-env`.
- **Curriculum scheduler** — staged hparams:
  - small schema → wide schema,
  - short stream → long stream,
  - sorted ops → adversarial shuffled.

### SFT recipe

- Targets: per-cell tokens of the post-stream table.
- Loss: cross-entropy per cell; optionally curriculum-weighted by
  difficulty.
- Auxiliary objective: predict-next-op self-supervision.

### RL recipe

- Episode = (stream, queries).
- Action = emitted answer token sequence.
- Reward = bits recovered on the answer, with optional shaping:
  - *dense* — cell-accuracy delta after each query;
  - *sparse* — end-of-episode aggregate.
- Off-policy data from SFT generations is supported.

### Diagnostics (the architecture-purity signals)

- **State probe.** Train a linear head on the hidden state to predict
  the ground-truth table; the probe accuracy upper-bounds any read-out
  head. Indicates whether *representational capacity* or *read-out* is
  the bottleneck.
- **Capacity curve.** Sweep hidden-state size; report bits recovered.
  This is the clean architecture signal — the answer to "how good is
  this sequence model, per bit of state?".

### Reference training scripts

- Small Mamba / RNN / decoder-only transformer trained from random
  init on the curriculum.
- Establishes that the domain is *learnable from scratch*, not just
  amenable to large pretrained-LLM prompting.

---

## Why the four-way split

- **`dbemu-mini`** — low barrier, broad participation, single number.
- **`dbemu-static`** — stable comparison, citation-friendly, lasts.
- **`dbemu-env`** — research-grade ablation, robustness,
  continuous-agent regime.
- **`dbemu-train`** — enables architecture and training-method work
  that does *not* depend on frontier-model API access.

The four share one grammar, one oracle, and one wire format, so
progress on any one transfers to the others.

---

## Repo layout

```
dbemu-bench/
  core/         grammar + sqlite oracle + json wire format (shared)
  mini/         dbemu-mini: frozen 10k instances + scorer
  static/      dbemu-static: 5 subtasks, frozen splits, two leaderboards
  env/          dbemu-env: generator, seed runner, plot helpers
  train/        dbemu-train: gym env, curriculum, SFT/RL recipes, probes
  paper/        LaTeX writeup
```

## Build order

1. `core/` — grammar, SQLite oracle, JSON wire format, cell-F1 and
   bits-recovered scorers.
2. `mini/` — 10 k instance generator, scoring script, sample
   submission. Ship first; it is the smallest complete deliverable.
3. `static/` — five subtask generators, frozen splits, two-leaderboard
   harness.
4. `env/` — full hparam grid, seed runner, plotting.
5. `train/` — env wrapper, curriculum, SFT loop, RL loop, probe.
