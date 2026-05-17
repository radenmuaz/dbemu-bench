# Table of contents

Index of all design documents for `dbemu-bench`, with a reading order
and a suggested mapping to LaTeX sections for a unified paper writeup.

---

## Documents (canonical reading order)

| #  | File                                               | One-liner                                                                   |
| -- | -------------------------------------------------- | --------------------------------------------------------------------------- |
| 1  | [CONTINUOUS_AGENT.md](CONTINUOUS_AGENT.md)         | Position paper: continuous agents, idle-tick compute, why tensor-native     |
| 2  | [PLAN_V3.md](PLAN_V3.md)                           | **Current canonical plan** — four shippable benchmarks                       |
| 3  | [MODELS.md](MODELS.md)                             | Architectures, hparams, training recipes, hidden-state and KV pressure       |
| 4  | [RELATED_WORK.md](RELATED_WORK.md)                 | Memory architectures, agentic memory, long-context benchmarks, System 1/2    |
| 5  | [MULTIMODAL_EXT.md](MULTIMODAL_EXT.md)             | Extension to perception/control: tolerance bands, multi-rate streams         |
| 6  | [PLAN_V2.md](PLAN_V2.md)                           | Predecessor — early four-bench carve-up (historical)                         |
| 7  | [PLAN_V1.md](PLAN_V1.md)                           | Original maximalist design — full spectrum L0–L11 and 14-task taxonomy       |

### Quick-start paths

- **"Just tell me the benchmark"** → [PLAN_V3.md](PLAN_V3.md).
- **"Why does this matter?"** → [CONTINUOUS_AGENT.md](CONTINUOUS_AGENT.md), then [PLAN_V3.md](PLAN_V3.md).
- **"I want to submit a model"** → [PLAN_V3.md](PLAN_V3.md) §1 (`dbemu-mini`), then [MODELS.md](MODELS.md) §4.2 (pretrained zero-shot).
- **"I want to train from scratch"** → [PLAN_V3.md](PLAN_V3.md) §4 (`dbemu-train`), then [MODELS.md](MODELS.md) §4.1 and §5.
- **"I want the full research story"** → 1 → 7 → 2 → 3 → 4 → 5 (in that order).

---

## Document summaries

### 1. [CONTINUOUS_AGENT.md](CONTINUOUS_AGENT.md)
Position paper. Argues that event-driven LLM agents are structurally
limited and that recurrent / SSM forward passes on $\varnothing$ input
are the substrate of *continuous agency*. Introduces the
**idle-ticks-per-op** axis $\rho$ and motivates why tensor-native
database emulation is the right benchmark substrate.
Key sections: two clocks; idle-tick activities; why externalization
launders the hard part; the continuous-agent regime; auto-research
implications.

### 2. [PLAN_V3.md](PLAN_V3.md)  *(current canonical plan)*
Four shippable benchmarks on one shared core (grammar, SQLite oracle,
JSON wire format, serialization, normalized bits-recovered metric).
Re-promotes the strategy spectrum L0–L11 into the main doc and
includes a changelog from v2.

- §0 Changes from v2
- §1 Shared core
- §2 Strategy spectrum (promoted)
- §3 Tasks (8 in `static`; 14 generatable in `env`/`train`)
- §4 `dbemu-mini` (Kaggle)
- §5 `dbemu-static` (frozen LongBench-style)
- §6 `dbemu-env` (procedural / seeded; fractional-factorial)
- §7 `dbemu-train` (gym-like env, curriculum, SFT + RL, probes)
- §8 Baselines table
- §9 Repo layout & build order

### 3. [MODELS.md](MODELS.md)
Per-task pressure profile; minimal and recommended architectures for
three regimes (from-scratch, pretrained zero-shot, pretrained +
finetuned); concrete hyperparameter tables; explicit recipes for
**pressuring hidden state** (capacity sweeps, anti-leak conventions,
blind snapshots) and **pressuring KV cache** (overflow, silent-think
budget, attention-mask "no-peek" trick that forces latent reasoning).
Three named tracks: state-capacity, KV-strategy, latent-reasoning.

### 4. [RELATED_WORK.md](RELATED_WORK.md)
Surveys four adjacent literatures and positions `dbemu-bench` in
each. Includes citation links through 2025.

- §1 Memory in sequence models (architectural, KV mgmt, retrieval, agentic)
- §2 Long-context benchmarks (RULER, BABILong, LongBench, NIAH)
- §3 System 1 / System 2 reasoning (o1, R1, ToT, Reflexion, overthinking)
- §4 Continuous / always-on agents and adaptive compute (ACT, PonderNet, FR-Ponder)
- §5 Where `dbemu-bench` sits (alignment table)

### 5. [MULTIMODAL_EXT.md](MULTIMODAL_EXT.md)
Extension blueprint. Maps the spectrum and bits-recovered scoring
onto perception and control where lossy is the norm and
correctness has thresholds.

- §1–§2 Why and the five axes of mismatch
- §3 Tolerance-banded scoring $I_\sim(T;P)$ (the load-bearing primitive)
- §4 Multi-timescale memory and per-stream $\rho_*$
- §5 Spectrum L0–L11 reinterpreted for control
- §6 New task families `Tp_*` (perception) and `Tc_*` (control)
- §7–§8 Overlaps and gaps
- §9 Concrete proposal: `dbemu-perc` as a fifth benchmark
- §10 Open research questions

### 6. [PLAN_V2.md](PLAN_V2.md)  *(historical)*
Earlier four-bench carve-up. Kept for the
self-critique vs. v3 (see end of v3 §0). Superseded by PLAN_V3.

### 7. [PLAN_V1.md](PLAN_V1.md)  *(historical / design doc)*
Maximalist plan: full operation grammar, the eleven-point strategy
spectrum, fourteen-task taxonomy, full metric stack, full stress-axis
grid, auto-research API. Still the most complete *design surface* and
the right thing to cite for the *vision*; PLAN_V3 is the right thing
to cite for the *shippable artifacts*.

---

## Suggested LaTeX paper structure (mapping)

A single unified paper covering everything. Section numbers match a
target `\section`/`\subsection` outline; the "Source" column points
to the markdown document(s) that supply the content.

| LaTeX section                                          | Source documents                                                |
| ------------------------------------------------------ | --------------------------------------------------------------- |
| 1. Abstract                                            | new — distilled from PLAN_V3 §0 + CONTINUOUS_AGENT §1            |
| 2. Introduction                                        | CONTINUOUS_AGENT §1–§3                                          |
| 3. Inputs and operation grammar                        | PLAN_V1 §1–§2                                                   |
| 4. Two model regimes                                   | PLAN_V1 §3                                                      |
| 5. The strategy spectrum (L0–L11)                      | PLAN_V1 §4 + PLAN_V3 §2                                         |
| 6. Task taxonomy                                       | PLAN_V1 §5 + PLAN_V3 §3                                         |
| 7. Metrics                                             | PLAN_V1 §6                                                      |
| 8. Stress dimensions                                   | PLAN_V1 §7 + PLAN_V3 §6 (sparse-grid)                           |
| 9. Continuous agents and tensor-native emulation       | CONTINUOUS_AGENT §1–§6                                          |
| 10. Four concrete benchmarks                           | PLAN_V3 §4–§7                                                   |
|    10.1 `dbemu-mini`                                   | PLAN_V3 §4                                                      |
|    10.2 `dbemu-static`                                 | PLAN_V3 §5                                                      |
|    10.3 `dbemu-env`                                    | PLAN_V3 §6                                                      |
|    10.4 `dbemu-train`                                  | PLAN_V3 §7                                                      |
| 11. Models and training recipes                        | MODELS.md §1–§5                                                 |
|    11.1 Per-task pressure profile                      | MODELS §1                                                       |
|    11.2 Architectures across three regimes             | MODELS §3–§4                                                    |
|    11.3 Hyperparameter tables                          | MODELS §5                                                       |
| 12. Pressuring state and KV cache                      | MODELS.md §6–§8                                                 |
|    12.1 Hidden-state pressure recipes                  | MODELS §6                                                       |
|    12.2 KV-cache pressure and latent reasoning         | MODELS §7                                                       |
|    12.3 Task variants and three tracks                 | MODELS §8                                                       |
| 13. Baseline targets and predictions                   | PLAN_V3 §8 + MODELS §9                                          |
| 14. Auto-research API                                  | PLAN_V1 §9                                                      |
| 15. Multimodal extension                               | MULTIMODAL_EXT §1–§10                                           |
|    15.1 Five axes of mismatch                          | MULTIMODAL_EXT §2                                               |
|    15.2 Tolerance-banded scoring                       | MULTIMODAL_EXT §3                                               |
|    15.3 Multi-timescale memory                         | MULTIMODAL_EXT §4                                               |
|    15.4 Spectrum reinterpreted                         | MULTIMODAL_EXT §5                                               |
|    15.5 Proposed `dbemu-perc`                          | MULTIMODAL_EXT §9                                               |
| 16. Related work                                       | RELATED_WORK §1–§5                                              |
|    16.1 Memory in sequence models                      | RELATED_WORK §1                                                 |
|    16.2 Long-context benchmarks                        | RELATED_WORK §2                                                 |
|    16.3 System 1 / System 2 reasoning                  | RELATED_WORK §3                                                 |
|    16.4 Continuous agents and adaptive compute         | RELATED_WORK §4                                                 |
|    16.5 Where `dbemu-bench` sits                       | RELATED_WORK §5                                                 |
| 17. Conclusion                                         | new — distilled                                                  |
| References                                             | RELATED_WORK source list                                         |

### Notes for the LaTeX writer

- **Avoid duplicating PLAN_V1 wholesale.** Use it for §3–§8 (the
  design surface); reserve §10 for the v3 shippable carve-up so the
  reader sees both vision and product without redundancy.
- **The strategy-spectrum table is referenced from many sections.**
  Define it once in §5 with a `\label{tab:spectrum}` and use
  `\Cref{tab:spectrum}` everywhere else.
- **Bits-recovered scalar appears in §7, §13, and §15.** Define once
  in §7; reuse the symbol $\tilde{I}$ in §13 and the equivalence-
  modulated form $I_{\sim}$ in §15.3.
- **Continuous-agent axis $\rho$ appears in §8, §9, §12.** Define
  in §9; reuse elsewhere.
- **Three pressure tracks (§12.3) are the natural place** to motivate
  the per-architecture predictions in §13 — keep them adjacent so the
  reader sees claim → measurement → prediction in order.
- **MULTIMODAL_EXT can be cut for a shorter paper** (a focused
  benchmarks-track submission would stop at §14 and treat §15 as
  future work). The full version belongs in a longer venue.

### Existing LaTeX skeleton

The current [paper/dbemu-bench.tex](paper/dbemu-bench.tex) covers
sections 2–10 and 14, plus a §9 continuous-agents section. It
predates MODELS, RELATED_WORK, and MULTIMODAL_EXT. The remaining
work to align it with the structure above:

- Add §11 *Models and training recipes* (new).
- Add §12 *Pressuring state and KV cache* (new).
- Add §13 *Baseline targets and predictions* (extend the existing
  baselines section).
- Add §15 *Multimodal extension* (new, optional for short version).
- Add §16 *Related work* (new) with proper `\bibliography`.
- Refresh §17 *Conclusion* to mention all five contributions.
