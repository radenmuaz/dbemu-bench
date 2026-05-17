# dbemu-bench — v3

Four shippable benchmarks built on one design. Supersedes
[PLAN_V2.md](PLAN_V2.md); preserves the design surface from
[PLAN_V1.md](PLAN_V1.md) that v2 dropped too aggressively. Motivation
in [CONTINUOUS_AGENT.md](CONTINUOUS_AGENT.md).

## Changes from v2

1. **Strategy spectrum (L0–L11) promoted back** into the main doc;
   submissions self-declare a level for post-hoc slicing.
2. **Static subtasks 5 → 8** — added T3 (schema), T4 (transactions),
   T13 (constraints). Cheap to add, cover famous LLM failure modes.
3. **`static` test size 100 → 1000 per subtask** — enables statistical
   discrimination between close models.
4. **`static` formats: JSON + CSV** — multi-format insight partially
   restored.
5. **`env` hparam grid → fractional-factorial** sparse plan (~500 runs)
   instead of an intractable 7.5 M-run cross-product.
6. **Headline metric unified to normalized bits recovered** across all
   four benchmarks. Cell-F1 is reported as a diagnostic only.
7. **`train` serialization specified** explicitly so the recipe is
   reproducible.
8. **Baseline-numbers slot** added with a defined fill-in protocol.

---

## Shared core

All four benchmarks share:

- One **operation grammar** — DDL / DML / TX / QUERY (full set in
  PLAN_V1 §2).
- One **oracle** — SQLite executes canonical state.
- One **wire format** — JSON
  `{schema, initial_state, ops, queries, answers?}`.
- One **headline metric** — **normalized bits recovered**
  $\tilde{I} = I(T; P) / H(T) \in [0,1]$.
  Cell-F1 and per-column accuracy reported as diagnostics.
- One **serialization** for `train` — see §4 below; the same
  serialization is also the canonical output format for the other
  three benchmarks, so a single model can play across all of them.

---

## Strategy spectrum (L0–L11), promoted

Submissions to any benchmark may declare a level `L<n>`. The level is
not enforced — it is used for post-hoc slicing of the leaderboard,
because the cost denominator changes per level (KV bytes for L1–L6,
tool-call count for L7–L10).

| L  | Name              | State medium     | Cost unit          |
| -- | ----------------- | ---------------- | ------------------ |
| 0  | Parametric        | weights          | params             |
| 1  | In-context        | KV (full)        | tokens             |
| 2  | Recurrent         | hidden state     | state bits         |
| 3  | Evicting KV       | bounded KV       | KV bytes           |
| 4  | Self-summary      | KV of digests    | KV bytes           |
| 5  | Hierarchical      | layered codes    | KV bytes           |
| 6  | Text scratchpad   | emitted tokens   | tokens             |
| 7  | KV tool           | external bytes   | tool calls         |
| 8  | Typed tool        | external typed   | tool calls         |
| 9  | SQL tool          | real DB          | tool calls         |
| 10 | Planner/executor  | DB + plan trace  | tool calls + plan  |
| 11 | Pure DB (oracle)  | DB               | —                  |

Full level definitions in PLAN_V1 §4.

---

## Tasks

Eight subtasks shipped in `static`; the remaining six from PLAN_V1 §5
are still generatable by `env` and `train`.

### In `static` (8)

| ID         | Probes                                |
| ---------- | ------------------------------------- |
| `T1-read`  | read-after-write fidelity             |
| `T3-schema`| schema induction from DML stream      |
| `T4-tx`    | transactional consistency (savepoints)|
| `T5-long`  | long-range dependency (lag ≥ 10⁵)     |
| `T6-hash`  | random-key hash-table stress          |
| `T7-count` | counter-array stress                  |
| `T10-mvcc` | snapshot read at past op index        |
| `T13-cons` | UNIQUE / FK / CHECK enforcement       |

### Available in `env` and `train` only

`T2-agg`, `T8-join`, `T9-ooo`, `T11-type`, `T12-tok`, `T14-bulk`.

---

## 1. `dbemu-mini` — Kaggle-style

**Purpose.** Drop-in eval; five minutes to a number.

- **Format.** JSON, fixed schema (3 tables × 5 cols, INTEGER+TEXT).
- **Initial state.** 100 rows per table.
- **Stream.** 1 000 ops; 50 queries per instance (point-read +
  small-range only).
- **Output.** `predictions.json` — list of answers in query order.
- **Metric.** Normalized bits recovered, instance-mean. **One scalar.**
- **Optional submission field.** `"strategy": "L<n>"` for spectrum
  slicing.
- **Dataset.** 10 000 instances; 5 k train / 2.5 k public test /
  2.5 k private test. Frozen forever.
- **Env.** Pure offline; one Python scoring script, stdlib only.

---

## 2. `dbemu-static` — frozen subtasks

**Purpose.** Citable comparison across distinct capability slices.

- **Subtasks.** Eight, listed above.
- **Splits.** 4 000 train / 1 000 dev / 1 000 test per subtask. Frozen.
- **Stream lengths.** Mixed within each subtask: 10³, 10⁴, 10⁵.
- **Formats.** JSON and CSV. Per-instance the format is fixed; the
  per-subtask leaderboard reports both format-conditional means and an
  overall mean.
- **Metric.** Normalized bits recovered per subtask; macro-average over
  subtasks is the leaderboard scalar.
- **Two leaderboards.**
  - *Limited context (8 k tokens).* Memory strategy must be declared.
  - *Unlimited context / recurrent.* Results additionally reported per
    recurrent state bit.
- **Per-submission disclosure.** Spectrum level, context length, total
  compute (FLOPs), and total wall-time. Used for cost-normalized
  views.

---

## 3. `dbemu-env` — dynamic procedural env

**Purpose.** Robustness, ablation, cliff-finding, continuous-agent
studies.

### Sparse-grid sample plan

Full cross-product of seven axes is 7.5 M runs. Instead, we use a
**fractional-factorial** design (resolution V) over the six structural
axes plus a full sweep on the continuous-agent axis $\rho$:

| Axis                  | Levels                                    |
| --------------------- | ----------------------------------------- |
| stream length         | 10², 10⁴, 10⁶                             |
| schema width          | 1, 25                                     |
| keyspace              | 10, 10⁵                                   |
| churn per key         | 1, 100                                    |
| r/w ratio             | 0.1, 10                                   |
| format                | json, sqlite                              |
| **idle ticks / op ρ** | 0, 1, 4, 16, 64, 256 (full sweep)         |

Structural design: $2^{5-0}$ full factorial on the 2-level axes
× 3 stream lengths = 96 structural points. With 5 seeds each
= 480 runs per subtask. ρ adds a 6× multiplier when continuous-agent
analysis is requested, *not* by default.

A user opting into the full design can request the dense grid by flag.

### Protocol

- 5 seeds per structural point; report mean ± std of bits recovered.
- Standard outputs: per-axis slice plots, Pareto frontiers
  (accuracy vs. cost), continuous-agent derivative $\Delta_\rho$.
- Closer to Procgen than ARC: dataset is the seeded generator, not a
  shipped corpus.

---

## 4. `dbemu-train` — training framework (no leaderboard)

**Purpose.** Let a model learn to *be* a database emulator from
scratch, independent of pretrained LLM SQL priors.

### Serialization (canonical for all four benchmarks)

Tables, ops, and answers are serialized into a single token stream
with a fixed reserved vocab. This is what the model reads and writes;
the cell-F1 / bits-recovered scorers parse the same format.

Reserved tokens: `<bos>`, `<eos>`, `<table>`, `</table>`,
`<row>`, `</row>`, `<col>`, `<val>`, `<null>`, `<op>`, `<query>`,
`<answer>`. Identifiers and string values use a 32 k BPE; integers are
emitted as digit streams with `<sign>`; floats as `<sign>digits.digits`.

Table snapshot:

```
<table name=users>
  <row><col=id><val=1><col=name><val=alice></row>
  <row><col=id><val=2><col=name><val=bob></row>
</table>
```

Op:

```
<op kind=update table=users where=id:1 set=name:carol>
```

Answer (point read of a row):

```
<answer><row><col=name><val=carol></row></answer>
```

This format is grammar-checked by the harness so structurally invalid
outputs score 0 cleanly rather than crashing the scorer.

### Env API (gym-like)

```python
env = DBEmuEnv(seed, hparams)
obs = env.reset()              # initial_state (serialized)
for op in stream:
    obs, info = env.step(op)   # advances oracle; returns next serialized obs
ans = env.query(q)             # ground-truth answer (serialized)
```

`obs` mode is configurable: full-table snapshot, last-op only, idle
tick (empty op), or nothing.

### SFT recipe

- Targets: serialized post-stream table tokens.
- Loss: per-token cross-entropy summed over the answer / target span.
- Auxiliary objective: predict-next-op self-supervision, weighted at
  0.1.
- Curriculum: schema width → keyspace size → stream length →
  adversarial ordering, each ramped over 10 k–100 k steps.

### RL recipe

- Episode = (stream, queries).
- Action = serialized answer token sequence.
- Reward = normalized bits recovered on the answer.
- Shaping options:
  - *dense* — per-query cell-accuracy delta;
  - *sparse* — end-of-episode aggregate only.
- Off-policy data from SFT generations is supported (PPO with
  importance correction).

### Diagnostics

- **State probe.** Linear head trained on frozen hidden state →
  ground-truth cell tokens. Probe accuracy upper-bounds any read-out
  head and isolates "is the state representational, or is the
  read-out the bottleneck?".
- **Capacity curve.** Sweep hidden-state size; plot bits recovered.
  The clean architecture signal.

### Reference training scripts

Small Mamba / RNN / decoder-only transformer trained from random init
on the curriculum, all reaching ≥ 0.8 bits recovered on T1 at stream
length 10³ as the smoke-test. Establishes the domain is learnable
from scratch.

---

## Baselines (numbers to fill in)

The bench is incomplete without reference points. Slot to be filled by
the first release; protocol below so anyone can reproduce.

| L  | Baseline                            | Mini | Static (macro) |
| -- | ----------------------------------- | ---- | -------------- |
| 11 | SQLite oracle                       | 1.00 | 1.00           |
|  9 | LLM-with-SQL-tool                   | TBD  | TBD            |
|  6 | LLM with scratchpad table           | TBD  | TBD            |
|  4 | LLM with summary checkpoints @ 100  | TBD  | TBD            |
|  3 | LLM with sliding window             | TBD  | TBD            |
|  2 | Mamba-130M trained on `train`       | TBD  | TBD            |
|  2 | LSTM-50M trained on `train`         | TBD  | TBD            |
|  1 | LLM zero-shot, full stream          | TBD  | TBD            |
|  – | Random answer (uniform over types)  | ≈0   | ≈0             |

Fill protocol: each baseline ships with a runnable script under
`baselines/L<n>_<name>.py`; numbers above must be reproducible from a
single `make baselines` invocation.

---

## Repo layout

```
dbemu-bench/
  core/         grammar + oracle + JSON wire + serialization + scorers
  mini/         frozen 10 k instances + scorer + sample submission
  static/      8 subtasks, frozen splits, two-leaderboard harness
  env/          generator, fractional-factorial runner, plot helpers
  train/        gym env, curriculum, SFT/RL recipes, probe, ref scripts
  baselines/    L<n>_<name>.py — every row in the baseline table
  paper/        LaTeX writeup
```

## Build order

1. `core/` — grammar, oracle, wire format, serialization,
   bits-recovered + cell-F1 scorers.
2. `mini/` — 10 k generator, scoring script, sample submission, README.
3. `baselines/L1`, `L3`, `L9` — minimum set to populate the table.
4. `static/` — eight subtask generators, frozen splits, two-leaderboard
   harness.
5. `env/` — fractional-factorial runner, seed sweep, plotting.
6. `train/` — env wrapper, curriculum, SFT loop, RL loop, state probe,
   reference Mamba/RNN/transformer scripts.
7. Remaining baselines fill in.

---

## Why v3 over v2

v2's instinct (carve into shippable products) was right; v2's
execution dropped too much. v3 keeps the four-bench skeleton and
restores the spectrum, broadens the static subtask set, fixes the
`env` grid blowup, unifies the metric, and pins down the
serialization that `train` needs to actually be reproducible. The
result is one shippable artifact per audience without losing the
design coherence that made the original plan defensible as research.
