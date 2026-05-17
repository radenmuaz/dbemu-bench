# Continuous agents and why full-tensor-native DB emulation matters

A position note attached to `dbemu-bench`.

## 1. Two clocks: event-driven vs. continuous

Almost every LLM agent today is **event-driven**:

```
user_input ──► prompt build ──► forward pass ──► tool calls ──► reply ──► idle
```

Between events the model does nothing. It has no internal clock. It does
not think while nobody is typing. It cannot consolidate, anticipate,
garbage-collect, or rehearse. Its state lives in a transcript that is
replayed verbatim each turn; the model itself is a pure function
re-invoked on demand.

A recurrent / state-space model is structurally different. It runs a
forward pass every tick:

$$h_t = f(h_{t-1}, x_t), \qquad x_t \in \mathcal{X} \cup \{\varnothing\}$$

The input $x_t$ is *allowed to be empty*. The state still advances. The
agent has an **internal clock decoupled from input arrival**. This is the
substrate on which a *continuous agent* becomes possible:

- **Idle compute**: forward passes consume time even when nothing
  arrived; the model can use them to reorganize state.
- **Anticipation**: it can predict the next operation and pre-compute
  partial answers before being asked.
- **Consolidation**: it can replay-and-compress recently-seen ops into a
  more compact internal representation (sleep / dream analog).
- **Background maintenance**: it can re-derive invariants, refresh
  derived aggregates, evict stale facts — exactly the work a real
  database does in the background.
- **Adaptive compute**: hard queries get more ticks; trivial ones get
  fewer. The decoupling of wall-clock from input is what makes adaptive
  compute *naturally expressible*.

An event-driven transformer can imitate some of this with scratchpads
and tool loops, but only by re-entering the prompt at every step. The
continuous version is cheaper, lower-latency, and fundamentally
different in kind: **the agent has interior time.**

## 2. What a continuous DB-emulating agent looks like

Strip everything to the essentials:

```
loop t = 0, 1, 2, ...
    x_t  ← next_op_or_none()
    h_t  ← f(h_{t-1}, x_t)
    y_t  ← g(h_t)              # maybe a prediction, maybe nothing
    emit(y_t) if salient
end
```

`f` is the recurrent update; `g` is the read-out head. The hidden state
$h_t$ *is* the database. When no op arrives, $f$ still runs:
$h_t = f(h_{t-1}, \varnothing)$. That step is not wasted — it is where
internal reorganization happens.

Concrete uses of no-op ticks in this setting:

| Idle-tick activity            | DB analog                                  |
| ----------------------------- | ------------------------------------------ |
| Re-compress hot keys          | rebuild B-tree pages, vacuum               |
| Refresh derived aggregates    | refresh materialized views                 |
| Pre-cache likely queries      | query plan cache warm-up                   |
| Detect / repair drift         | constraint check, fsck                     |
| Rehearse rare facts           | anti-forgetting replay                     |
| Predict next op               | speculative execution                      |

None of these require a tool call. None of them require a user prompt.
They happen because the agent has an interior life.

## 3. Why full tensor-native DB emulation matters

The strategy spectrum in `PLAN.md` runs from L0 (parametric) through L2
(recurrent / fixed-state) up to L11 (pure external DB). The *agentic*
levels (L7 – L11) are easier and, in a real engineering sense,
preferable for production. But they are uninformative as a benchmark for
intelligence, for three reasons:

### 3.1 Externalization launders the hard part

If the model writes `INSERT INTO users ...` to SQLite, the *remembering*
is done by SQLite. The model only has to (i) parse intent, (ii) generate
syntactically valid SQL, (iii) read the result back. None of those test
state-tracking; all of them are tests of translation. A 7B model with a
SQL tool can score perfectly on a task that a 70B model without one
cannot touch. The tool, not the model, did the work.

That is fine for products. It is a category error for benchmarks of
sequence-model capability.

### 3.2 Continuous agents need internal state, not RPC

A continuous agent makes a forward pass every tick. If its state lives
behind an RPC (`kv.get`, `SELECT`), then every tick that touches state
is at least one network round-trip. That cannot be the substrate of an
agent that ticks at, say, 1 kHz or per-token. Continuous agency
*requires* state in the tensors. Tool-memory is fine for episodic
recall; it is wrong for moment-to-moment cognition.

If we want continuous agents, we need architectures that can hold
substantial structured state in their hidden vectors, evolve it over
long horizons, and read from it at the speed of one matmul. That is
exactly the L2 problem. Nothing else measures it.

### 3.3 The bench should pay for what is hard

The information-theoretic scalar in the bench — bits recovered
normalized to $H(T)$ — is the same number for L2 and L9. But the *cost*
denominator changes:

- At L9, an extra bit costs roughly one extra SQL round-trip.
- At L2, an extra bit costs *more recurrent state* — i.e., a larger,
  better-trained model.

So the L2 numbers are the ones that move when architecture moves. The
L9 numbers are the ones that move when prompting moves. Both are
useful; only L2 is a signal for new sequence-model research.

A benchmark that does not separate these axes will mis-rank progress.
`dbemu-bench` separates them by construction.

## 4. The continuous-agent regime in `dbemu-bench`

Add one axis to the existing stress dimensions: **idle ticks per op**,
$\rho \in \{0, 1, 4, 16, 64, 256\}$.

For every op, the model receives $\rho$ no-op forward passes before the
next op arrives. Reads/queries are evaluated *after* the trailing idle
window. This gives the continuous agent a chance to:

- consolidate the just-arrived op into its state,
- anticipate the upcoming query,
- repair invariants disturbed by the op.

The benchmark reports two derivatives:

- $\Delta_\rho$ = accuracy gain per unit idle tick. A purely reactive
  architecture has $\Delta_\rho = 0$. A genuinely continuous one has
  $\Delta_\rho > 0$ and eventually saturates.
- Idle-compute efficiency = bits recovered per idle-tick FLOP. This
  distinguishes architectures that use idle compute well from ones that
  spin.

Baselines added for this regime:

- ACT / PonderNet halting policies (adaptive idle compute).
- Mamba / RWKV with looped feedback on $\varnothing$ input.
- A "rumination" wrapper over an LLM that lets it emit hidden
  scratchpad tokens during $\rho$ idle ticks.

## 5. Auto-research implications

If we believe AI research should increasingly be done by AI research
agents, the bench should produce signals those agents can act on:

- **Sparse, named failure modes.** Each task isolates one. The agent
  can read a per-task heatmap and propose a targeted modification.
- **A single comparable scalar.** Normalized bits recovered. Lets the
  agent decide whether a proposed change won or lost without
  task-specific judgment.
- **A Pareto frontier instead of a single number.** Lets the agent
  reason about cost/capability tradeoffs, not just leaderboard rank.
- **A continuous-agent axis.** Lets the agent investigate the most
  open and least-measured part of the design space: what to do with
  idle compute.

This is why tensor-native database emulation, evaluated under both
event-driven and continuous regimes, is the right shape of benchmark for
the next round of sequence-model research. It is the smallest setup
that:

1. forces all state into the substrate under study,
2. exposes architecture capacity as a measurable quantity,
3. admits agentic strategies as one end of a spectrum rather than the
   only option, and
4. provides a place for "thinking while idle" to be either rewarded or
   exposed as decorative.

## 6. Summary

- Real cognition is continuous. Event-driven transformer agents are not.
- Recurrent / SSM forward passes happen every tick, even on $\varnothing$
  input, which is the right substrate for continuous agency.
- Externalizing DB state to a SQL tool makes products work and makes
  benchmarks lie: the tool does the remembering.
- Full tensor-native DB emulation puts the hard part — long-horizon
  structured state tracking — back inside the model, where the bench
  can measure it.
- An "idle ticks per op" axis turns the bench into the first test of
  continuous agency that has a ground truth.
