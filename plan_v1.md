# dbemu-bench

A benchmark for stress-testing sequence models and LLMs on **emulating a
database**: ingest an initial state, apply a long stream of operations,
answer queries. Designed to be informative for architecture research (RNN /
SSM / transformer) *and* for context-management strategy research (eviction,
summarization, retrieval, tool-memory).

---

## 1. Inputs (state representations)

The same logical database is rendered four ways, so format-sensitivity is
measurable:

- **SQLite**: schema + rows; also the canonical ground-truth executor.
- **JSON**: documents per table, or single nested object; tests nested-state
  tracking.
- **CSV**: header + rows; tests flat tabular recall.
- **FlatBuffers**: schema-bound binary; tests strict-typed,
  offset-addressed recall.

Each instance ships as `{schema, initial_state, op_stream, queries}`. The op
stream is format-agnostic at the semantic level; the harness renders it into
each surface form per run.

---

## 2. Operation grammar

- **DDL**: `CREATE / ALTER / DROP TABLE`, `CREATE INDEX`
- **DML**: `INSERT`, `UPDATE`, `DELETE`, `UPSERT`
- **TX**: `BEGIN`, `SAVEPOINT`, `ROLLBACK [TO sp]`, `COMMIT`
- **Query** (interleaved or terminal): `SELECT` with filter / join /
  group-by / agg / order / limit, `EXISTS`, `COUNT`
- **Time-travel**: `AS OF <op_idx>` (MVCC snapshot reads)
- **Bulk**: `IMPORT csv`, `MERGE`

The op stream is the model's input after the initial state; queries are the
eval points.

---

## 3. Two model regimes

### Regime A — context-limited transformer

Stream length >> context window. Model must adopt a memory strategy:

| Strategy            | Mechanism                                                |
| ------------------- | -------------------------------------------------------- |
| Sliding-KV          | drop oldest KV blocks                                    |
| Summary-checkpoint  | every N ops, emit a state digest; replace evicted KV    |
| Hierarchical summary| multi-level digests (per 10 / 100 / 1k ops)              |
| Retrieval (RAG)     | embed past ops/digests; recall at query time             |
| Tool-memory         | model writes to an external KV store, reads back         |
| Learned compressor  | column-wise quantized state encoded into tokens          |
| Hybrid              | combinations of the above                                |

Each strategy is a plug-in; the benchmark scores the *strategy*, not just
the model.

### Regime B — context-unlimited / fixed-state

For RNN, SSM/Mamba, RWKV, or any transformer with full window. Same tasks,
no eviction. Adds a critical axis: **state-size-normalized metrics**
(accuracy per bit of recurrent state). This is what makes the bench
informative for architecture research, not just for prompting.

---

## 4. Strategy spectrum: full-tensor → agentic-DB

Eleven points, ordered by how much state leaves the model's tensors and
lives in external software. Each is a valid `strategy` in the bench; the
Pareto sweep across them is the real research artifact.

```
weights → hidden → KV → text scratchpad → opaque KV → typed tool → SQL → planner+executor → pure DB
[ tensor-native ──────────────────][ hybrid ────────────────────][ agentic ────────────────────]
```

Along the axis:

- **Where state lives**: parameters → activations → KV-cache → emitted
  tokens → external bytes.
- **Who interprets it**: the network → the network + prompt → external
  software.
- **What the bench measures**: architectural capacity → context-management
  capability → tool-use / planning capability.
- **Failure mode**: forgetting → eviction lossiness → tool-call errors →
  planning errors.

### Levels

| L  | Name                | State medium       | Bench signal isolates             |
| -- | ------------------- | ------------------ | --------------------------------- |
| 0  | Parametric          | weights            | training-time memorization        |
| 1  | In-context          | KV (full)          | long-context attention            |
| 2  | Recurrent           | hidden state       | architecture / state bits         |
| 3  | Evicting KV         | bounded KV         | eviction policy                   |
| 4  | Self-summary        | KV of digests      | model's compression sense         |
| 5  | Hierarchical/learned| layered codes      | budgeted compression              |
| 6  | Text scratchpad     | emitted tokens     | format discipline                 |
| 7  | KV tool             | external bytes     | retrieval discipline              |
| 8  | Typed tool          | external typed     | schema + planning                 |
| 9  | SQL tool            | real DB            | SQL competence                    |
| 10 | Planner/executor    | DB + plan trace    | meta-reasoning                    |
| 11 | Pure DB (oracle)    | DB                 | ceiling                           |

#### L0 — Pure parametric
State baked into weights. No runtime update. Tests memorization /
generalization across schemas. Breaks on anything dynamic.

#### L1 — Pure in-context
Full op-stream re-fed every call. State is the activation pattern induced
by the prompt. Cost is O(n²) attention; breaks at stream length > context.

#### L2 — Fixed-state recurrent (RNN / SSM / Mamba)
State is the hidden vector. Single pass, no re-read. Tests pure
state-tracking capacity per bit of state — the architecture signal. Report
metrics normalized by state bits.

#### L3 — KV-cache with eviction (sliding / attention-score / H2O-style)
Bounded KV managed by a policy. Locality/recency profile diagnoses the
policy.

#### L4 — Self-summarization
Model emits a natural-language state snapshot at checkpoints; old tokens
dropped, snapshot stays. Tests the model's ability to factor state
(separate variable from constant, hot from cold).

#### L5 — Hierarchical / learned compressor
Multi-level digests, or a learned encoder packing rows into a fixed
token-budget. Tests lossy compression at a fixed budget.

#### L6 — Structured in-prompt scratchpad
Model maintains an ASCII/Markdown/JSON table inside its prompt and rewrites
it each turn. Tests discipline, format adherence, size control.

#### L7 — Opaque external KV tool
`kv.put(k, v)`, `kv.get(k)`, `kv.scan(prefix)`. No schema enforcement.
Tests when to call, with what keys, retrieval recall@k.

#### L8 — Typed table tool
`table.create/insert/update/select` over a schema. Engine enforces types,
uniqueness. Tests schema design + query planning, not state-tracking.

#### L9 — Full SQL tool
Model emits SQL against a real engine (SQLite). The engine *is* the state.
Tests SQL competence, schema understanding, transactional reasoning.

#### L10 — Planner / executor split
Model is the planner only; decomposes intent into sub-tool calls (parser,
optimizer, executor, verifier). Tests meta-reasoning and error recovery.

#### L11 — Pure DB passthrough (oracle)
No model in the loop. Ground-truth upper bound.

### Why the spectrum matters

Each level collapses different metrics:

- **L0 – L2**: state-bit-normalized accuracy is load-bearing; tool-call
  counts irrelevant.
- **L3 – L6**: locality/recency profile + cost-per-KV-byte dominate.
- **L7 – L10**: tool-call count and correctness, plan length dominate; raw
  recall approaches 1.0 because the DB does the remembering.
- **L11**: zero variance; pins the ceiling.

A complete bench run reports **one Pareto frontier across all 11 levels**
for each task family. The interesting research questions live in the gaps:

- Where does L2 overtake L3 per byte? — architectural advantage of recurrence.
- Where does L7 beat L4 per token? — externalization vs. compression.
- Where does L9 stop helping? — when planning errors eat externalization gains.

---

## 5. Task taxonomy

Grouped by what they break.

| ID  | Name                          | Breaks                       |
| --- | ----------------------------- | ---------------------------- |
| T1  | Read-after-write fidelity     | weak recall                  |
| T2  | Aggregate fidelity            | lossy summarization noise    |
| T3  | Schema induction              | no schema-tracking           |
| T4  | Transactional consistency     | no tentative/committed split |
| T5  | Long-range dependency         | sliding window (cliff)       |
| T6  | Hash-table stress             | summary + RAG                |
| T7  | Counter array                 | both window and summary      |
| T8  | Cross-table join after stream | weak cross-state binding     |
| T9  | Out-of-order replay           | naive autoregressive replay  |
| T10 | MVCC snapshot read            | history discarded            |
| T11 | Type coercion across formats  | non-invariant representations|
| T12 | Adversarial tokenization      | near-duplicate key handling  |
| T13 | Constraint enforcement        | no UNIQUE/FK/CHECK tracking  |
| T14 | Bulk import + diff            | mix of wide-load and update  |

---

## 6. Metrics (theoretically grounded)

### 6.1 Lossless primitives

- **Exact match** — sanity only.
- **Cell-F1** over `(table, row_id, col) → value` tuples, decomposed:
  row precision/recall, column precision/recall, value accuracy given
  row+col matched.
- **Schema-F1** — `(name, type)` pairs.
- **Order metric** (when `ORDER BY`): Kendall-τ on returned sequence.

### 6.2 Lossy / aggregate primitives

- **Relative error** for scalar aggregates: `|p − t| / max(|t|, ε)`.
- **sMAPE** (bounded symmetric variant).
- **Earth Mover's Distance** for histograms / group-by distributions.
- **Jaccard / overlap@k** for set- and top-k–valued answers.
- **Bucketed accuracy** with tolerance bands (1 / 5 / 10 / 25 %).

### 6.3 Partial-credit primitives

- **Tree edit distance** (Zhang–Shasha) over JSON / nested rows.
- **Row-tuple edit distance** for tables.
- **Coverage × accuracy decomposition**:
  `coverage = cells_attempted / cells_truth`,
  `accuracy = cells_correct / cells_attempted`. Report both, plus harmonic
  mean. Prevents "answer-only-what-you-know" gaming.

### 6.4 Information-theoretic composite (load-bearing)

- **Bits-recovered**: `I(T; P) = H(T) − H(T | P)`. Estimated empirically.
  For deterministic tables, reduces to
  `Σ_c entropy(c) · cell_accuracy(c)`. Single scalar, comparable across
  tasks, additive across cells.
- **Normalized**: `bits-recovered / H(T) ∈ [0, 1]` — task-comparable
  "information fidelity".
- **Calibration** (Brier, ECE) when the model emits a per-cell
  distribution. Separates wrong-and-confident from wrong-and-uncertain.
- **Log-likelihood under teacher-forced replay** for generative scoring.

### 6.5 Cost-aware (for the Pareto)

- bits-recovered per **prompt token**
- bits-recovered per **KV byte** (Regime A) / **state bit** (Regime B)
- bits-recovered per **wall-second** and per **FLOP**

### 6.6 Locality / recency profiles

For T5/T6/T10, plot accuracy vs. write-to-query lag. The *shape*
diagnoses memory strategies: cliff = sliding window; slow decay =
summarization lossiness; flat = real memory.

---

## 7. Stress dimensions

| Axis                   | Values                                   |
| ---------------------- | ---------------------------------------- |
| Stream length          | 10², 10³, 10⁴, 10⁵, 10⁶                  |
| Schema width           | 1, 5, 25, 100 cols                       |
| Schema depth           | 0, 1, 3, 5                               |
| Keyspace size          | 10, 10³, 10⁵, 10⁷                        |
| Write churn per key    | 1, 10, 100, 1000                         |
| Read/write ratio       | 0.01, 0.1, 1, 10                         |
| Query complexity       | scan, 1-join, 2-join, agg-group, subq    |
| Format                 | sqlite, json, csv, flatbuffers           |
| Op ordering            | sorted, random, adversarial-shuffled     |

A single benchmark run is a seeded sparse grid sample.

---

## 8. Reference baselines (shipped)

1. Zero-shot LLM, full stream in context (when it fits).
2. Sliding-window LLM.
3. Summary-checkpoint LLM.
4. RAG-over-ops LLM.
5. Tool-memory LLM (write/read to scratch KV).
6. Hand-written stateful agent (oracle upper bound for that strategy).
7. RNN baseline (LSTM).
8. SSM baseline (Mamba).
9. SQLite ground-truth (perfect-score reference).

---

## 9. Auto-research API

Single entry point:

```
evaluate(strategy_fn, regime, axes_subset) -> {
  scalar,          # normalized bits-recovered (default leaderboard metric)
  breakdown,       # per-task, per-axis dataframe
  pareto_points    # (cost, accuracy) pairs for plotting
}
```

This lets a research agent propose a new memory strategy, run it, and get
a single comparable number plus a diagnostic profile — which is what makes
the bench useful for automated capability search rather than just
leaderboarding.

---

## 10. Repo layout (proposed)

```
dbemu-bench/
  spec/         format-agnostic op grammar, schema DSL
  gen/          generators per task family (T1..T14)
  render/       sqlite | json | csv | flatbuffers serializers
  oracle/       sqlite-backed ground truth + queries
  metrics/      cell_f1, emd, ted, bits_recovered, calibration, cost
  strategies/   sliding, summary, hier, rag, tool, learned
  models/       LLM adapters, RNN, mamba baselines
  harness/      runner, scheduler, sweep config
  reports/      pareto + heatmap rendering
  paper/        LaTeX writeup
```

---

## 11. Build order

1. Spec + SQLite oracle + JSON/CSV/FB renderers.
2. T1 + T2 with cell-F1 and bits-recovered.
3. Sliding + summary strategies + zero-shot baseline.
4. T5/T6/T7 (the Regime-A breakers) with locality profiles.
5. Mamba/RNN adapters + state-normalized metrics.
6. Remaining tasks T8 – T14.
7. Pareto / heatmap reports, strategy API freeze.
