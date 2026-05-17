# Models for dbemu-bench

Architectures, hyperparameters, and training recipes to excel at the
tasks in [PLAN_V3.md](PLAN_V3.md). Covers three regimes — random
init, pretrained-zero-shot, pretrained-finetuned — and includes
concrete recipes for **pressuring hidden state and KV cache** so each
task isolates a specific capacity bottleneck.

Reading order: [PLAN_V3.md](PLAN_V3.md) → this file →
[CONTINUOUS_AGENT.md](CONTINUOUS_AGENT.md) for the recurrent /
idle-compute angle.

---

## 1. Per-task pressure profile

Each PLAN_V3 subtask exerts pressure on a *specific* resource. The
table is what motivates the architecture choices in §3 and the
"pressure variants" in §8.

| Task        | Dominant pressure                | Information needed at readout              | Min state to saturate (est.)              |
| ----------- | -------------------------------- | ------------------------------------------ | ----------------------------------------- |
| `T1-read`   | local state / cell recall        | current value of queried cells             | $H(\text{queried cells})$ bits            |
| `T3-schema` | structural memory                | column names + types of all tables         | $\sim 10^2$–$10^3$ bits                   |
| `T4-tx`     | branched / tentative state       | current value + savepoint stack            | $k \cdot H(\text{table})$ for $k$ saves   |
| `T5-long`   | memory horizon                   | one value written far in the past          | $H(\text{value})$ bits; recall over lag   |
| `T6-hash`   | associative capacity             | $K$ independent key→value pairs            | $K \cdot v$ bits ($v$ = value bits)       |
| `T7-count`  | factored counter capacity        | $K$ independent integer counters           | $K \cdot \log_2 \max\text{count}$ bits    |
| `T10-mvcc`  | history retention                | $S$ snapshots, each $H(\text{table})$ bits | $S \cdot H(\text{table})$ bits            |
| `T13-cons`  | invariant tracking               | uniqueness sets, FK closures               | $\sim H(\text{indexes})$ bits             |

### Worked example — T6 hash, T7 counter, T10 MVCC

These three are the "capacity breakers" because the pressure scales
linearly with a task parameter.

- **T6 hash**, $K = 10^3$ uint32 values:
  $\approx 32{,}000$ bits $\approx 4$ kB of *non-redundant* state.
- **T7 counter**, $K = 10^3$, max count $256$:
  $\approx 8{,}000$ bits $\approx 1$ kB.
- **T10 MVCC**, $S = 10$ snapshots of a $1$ kB table:
  $\approx 10$ kB.

A 256-dim hidden state at fp16 is $\sim 0.5$ kB and cannot saturate
T6 at $K \geq 100$. The *capacity curve* (PLAN_V3 §4 diagnostics) is
exactly this measurement — sweep hidden-state size, plot bits
recovered, locate the cliff.

---

## 2. How each architecture responds to pressure

| Arch                       | State medium                                  | Saturates first on              | Scales by                            |
| -------------------------- | --------------------------------------------- | ------------------------------- | ------------------------------------ |
| LSTM / GRU                 | $h, c \in \mathbb{R}^{d_h}$                   | T6 / T7 (associative)           | wider $d_h$, more layers             |
| Mamba (S6)                 | $h \in \mathbb{R}^{d_s \times d_i}$ per layer | T6 at high $K$                  | larger $d_s$, more layers            |
| Mamba-2 (SSD)              | larger $d_s$, head-grouped                    | T6 at very high $K$             | head dim, head count                 |
| DeltaNet                   | matrix $S$ (key-value outer products)         | length extrapolation (no decay) | $d_\text{state}$; conv. helps        |
| Gated DeltaNet             | matrix $S$ + gating + delta update            | T10 (extreme retention)         | $d_\text{state}$, head count         |
| TTT-Linear / TTT-MLP       | hidden state is a *learned inner model*       | continues to improve past 16 k  | inner-model capacity, inner lr       |
| Atlas / Titans-style       | high-capacity associative memory module       | beyond standard MQAR ceilings   | memory module size, window size      |
| RWKV / Hyena / xLSTM       | mixed                                         | variant-specific                | variant-specific                     |
| Hybrid (linear + softmax)  | linear-attn KV + sparse full-attn KV          | T5 at the linear-only layers    | full-attn layer ratio (e.g. 3:1)     |
| Transformer (full ctx)     | KV-cache (grows with seq)                     | nothing until ctx overflow      | longer context, more heads           |
| Transformer (evict KV)     | bounded KV                                    | T5 (cliff at window edge), T6   | bigger window, better eviction       |
| Transformer + tool         | external store                                | nothing — tool does the work    | tool reliability                     |

The DB benchmark is built to *distinguish* these. The architecture-purity
signal from PLAN_V3 (capacity curve) is the cleanest way to do it.

---

## 2.5 Capacity on exact copying and precision memory

The post-Mamba RNN literature has converged on **exact copying / multi-query
associative recall (MQAR) / S-NIAH** as the canonical micro-benchmarks for
precision memory. Empirical findings from those benchmarks map directly
onto our T1 (read), T6 (hash), and T10 (MVCC) tasks. Summary of what the
literature establishes:

- **Pure linear attention fails precision recall.** Linear-attention RNNs
  (RetNet, GLA, RWKV-4) lack the precision to perform local token shifts
  and exact comparisons (Arora et al., *Based*, 2024). This is the
  baseline failure mode on T6.
- **State size is the recurrent ceiling.** A fundamental tradeoff between
  recurrent state size and MQAR accuracy holds within and across
  architecture classes — increasing state size almost always increases
  accuracy. This is the same statement as our capacity curve for T6.
- **DeltaNet ≫ Mamba on MQAR per state.** Replacing the additive update
  with the **delta rule** (Schlag et al. 2021; Yang et al. 2024) minimizes
  per-step prediction MSE and yields perfect MQAR in the hardest setting
  even *without* convolution. Caveat: original DeltaNet has weak length
  extrapolation because it lacks an explicit decay.
- **Gated DeltaNet fixes the extrapolation + retention gap.** Combining
  the delta rule (precise targeted updates) with a gating term (rapid
  erasure) yields the best small-RNN performance to date on real-world
  in-context retrieval: 30.6 average accuracy vs. Mamba2 29.8 vs. DeltaNet
  26.2 (Yang et al., ICLR 2025). On S-NIAH-3 (complex pattern
  memorization), Mamba2 drops off quickly while Gated DeltaNet remains
  strong — directly relevant to T10-blind-snapshot pressure.
- **TTT-Linear / TTT-MLP scale beyond Mamba in long context.** Making
  the hidden state itself a *learned mini-model* updated by SGD at test
  time (Sun et al. 2024) lets the model keep reducing perplexity past
  16 k tokens, where Mamba's perplexity plateaus. The mini-model is the
  state; expressiveness comes from inner-model capacity, not from a
  larger fixed vector.
- **Atlas / Titans-style memory beats both per memory byte.** Atlas
  (Behrouz et al., 2025) treats memory as an optimizable module updated
  to memorize *the current window*, not just the last token, and reports
  the best MQAR per memory size of any tested architecture plus +80 %
  accuracy at 10 M-token BABILong. This is the strongest published
  signal that the right move for T10 is an explicitly-optimized memory
  module, not a passively-accumulated state.
- **Hybrids ship.** Production-grade open-weight models have converged
  on a small-fraction full-attention hybrid: **Qwen3-Next-80B-A3B**
  and **Kimi Linear** both use a **3:1 ratio** of Gated DeltaNet to
  Gated Attention. Pure linear handles the bulk; sparse full-attention
  layers restore the precision needed for shifts/comparisons. Native
  262 k context with no eviction.

### What this implies for `dbemu-bench`

| PLAN_V3 task    | Closest published benchmark      | Predicted best arch (a priori)                 |
| --------------- | -------------------------------- | ---------------------------------------------- |
| `T1-read`       | exact copy / S-NIAH-1            | hybrid (precision from sparse full-attention)  |
| `T5-long`       | MQAR with long distractor span   | Gated DeltaNet, TTT-Linear, hybrid             |
| `T6-hash`       | MQAR at high $K$                 | Gated DeltaNet ≫ Mamba2 ≫ Mamba ≫ linear-attn  |
| `T7-count`      | counter array; not in MQAR canon | factored representations (Mamba2, GDN)         |
| `T10-mvcc`      | S-NIAH-3 + window-memory         | Atlas / Titans-style; Gated DeltaNet runner-up |
| `T13-cons`      | constraint tracking              | hybrid (attention for set-membership)          |

Two consequences for the bench:

1. **Add Gated DeltaNet, TTT, Atlas, and Qwen3-Next-style hybrid to the
   baseline grid.** Without them the bench under-represents the actual
   2024–2025 frontier of precision-memory RNNs.
2. **Pressure variants in §6–§8 should sweep the dimensions that
   distinguish these architectures** — $K$ for capacity (separates
   Mamba from GDN), $L$ for extrapolation (separates DeltaNet from GDN),
   $S$ for snapshot retention (separates Atlas from everything else),
   and $M_\text{silent}$ for latent reasoning depth (separates TTT from
   classical SSM).

---

## 3. Minimal architectures (lower bound to play)

Smallest model that should be able to saturate a *baseline* slice
($K = 100$ for capacity tasks, stream length $10^3$, 8 k context):

| Family              | Min size      | Hidden / state                  | Layers | Notes                                          |
| ------------------- | ------------- | ------------------------------- | ------ | ---------------------------------------------- |
| LSTM                | $\sim 5$ M    | $d_h = 512$                     | 4      | enough for T1/T3/T5 at small scale             |
| Mamba (S6)          | $\sim 15$ M   | $d_\text{model}=384, d_s=16$    | 12     | early reference target                          |
| Mamba-2             | $\sim 25$ M   | $d_\text{model}=384, d_s=64$    | 12     | better T6/T7 capacity                          |
| DeltaNet            | $\sim 25$ M   | $d_\text{model}=384$, matrix $S$| 12     | best MQAR per state pre-gating; weak extrap.   |
| Gated DeltaNet      | $\sim 30$ M   | $d_\text{model}=384$, matrix $S$| 12     | strongest small RNN on precision tasks         |
| TTT-Linear          | $\sim 30$ M   | $d_\text{model}=384$, inner-LM  | 12     | hidden state = mini-model; scales past 16 k    |
| Hybrid GDN + Attn   | $\sim 30$ M   | as GDN, 3:1 with 8-head attn    | 12 / 4 | small Qwen3-Next-style hybrid                  |
| Tiny tx             | $\sim 25$ M   | $d_\text{model}=384, 8$ heads   | 12     | ctx $\leq 4$ k; will fail T5                   |
| Pretrained tx       | $\geq 1.3$ B  | (Phi-1.5, Qwen2.5-1.5B)         | —      | zero-shot baseline                              |

These are the *floor*. A 15 M Mamba is what `dbemu-train`'s
"learnable from scratch" smoke test (PLAN_V3 §4) targets.

---

## 4. Recommended architectures (per regime)

### 4.1 Regime A — random init (the architecture-purity track)

**Goal.** Establish that the domain is learnable from scratch and
produce a *capacity curve* per architecture.

| Arch                       | $d_\text{model}$  | $d_\text{state}$  | layers | heads | Params | Why                                                  |
| -------------------------- | ----------------- | ----------------- | ------ | ----- | ------ | ---------------------------------------------------- |
| LSTM-50M                   | 1024 ($\times 2$) | n/a               | 6      | —     | 50 M   | classical baseline; bad at T6 by design              |
| Mamba-130M                 | 768               | 16                | 24     | —     | 130 M  | early baseline; saturates T7 to $K=1000$             |
| Mamba2-130M                | 768               | 64                | 24     | —     | 130 M  | strong on T6                                          |
| DeltaNet-130M              | 768               | matrix $S$        | 24     | —     | 130 M  | best pre-gated MQAR per state                        |
| **Gated DeltaNet-130M**    | 768               | matrix $S$        | 24     | —     | 130 M  | **recommended default RNN** (precision + decay)      |
| TTT-Linear-130M            | 768               | inner-LM (linear) | 24     | —     | 130 M  | scales past 16 k where Mamba flatlines               |
| TTT-MLP-130M               | 768               | inner-LM (2-MLP)  | 24     | —     | 140 M  | best inner-model capacity per outer state            |
| **Hybrid GDN+Attn-130M**   | 768               | matrix $S$; 3:1   | 24 / 6 | 12    | 150 M  | **recommended default hybrid** (Qwen3-Next analog)   |
| Mamba2-370M                | 1024              | 128               | 32     | —     | 370 M  | should clear T6/T10 at $K=10^4$                      |
| Gated DeltaNet-370M        | 1024              | matrix $S$        | 32     | —     | 370 M  | best small RNN at precision                          |
| Hybrid GDN+Attn-370M       | 1024              | matrix $S$; 3:1   | 32 / 8 | 16    | 420 M  | precision + capacity ceiling for from-scratch tier   |
| Tx-370M (rope)             | 1024              | —                 | 24     | 16    | 370 M  | reference transformer; needs strategy                |
| Hybrid Mamba-MHA           | 1024              | 64                | 24 / 8 | 16    | 400 M  | recurrence + local attention for T1/T4               |
| Atlas-style 130M           | 768               | learned mem mod   | 24     | —     | 130 M  | dedicated memory module; targets T10 ceiling         |

**Recommended default for the bench.** A four-architecture grid that
exposes the most informative contrasts on one plot:

- **Gated DeltaNet-130M** — delta-rule recurrent ceiling at small scale
- **Mamba2-130M** — additive-update SSM at matched scale
- **Hybrid GDN+Attn-130M** — Qwen3-Next-style hybrid at matched scale
- **Mamba2-370M** + **Gated DeltaNet-370M** — capacity-scaling pair

This separates *architecture class* (delta-rule vs. additive vs. hybrid)
from *capacity scaling* on the same axes.

### 4.2 Regime B — pretrained, zero/few-shot

**Goal.** Measure emergent capability of frontier open-weight models
without modification.

| Model                       | Why                                                                       |
| --------------------------- | ------------------------------------------------------------------------- |
| Qwen2.5-3B                  | strong instruction-following at small scale                                |
| Llama-3.2-3B                | widely-used baseline                                                       |
| Phi-3.5-mini                | strong reasoning per param                                                 |
| Mamba-2.8B                  | pretrained SSM, cleanest L2 spectrum slot (additive update)                |
| Codestral-Mamba             | pretrained SSM with code priors (helps SQL-style)                          |
| Mistral-7B-Instr            | the "default" 7B for honest comparisons                                    |
| Qwen2.5-Coder-7B            | code priors help; tests whether SQL-skill leaks                            |
| **Gated DeltaNet-1.5B**     | open pretrained delta-rule RNN; cleanest L2 precision-memory slot          |
| **OLMo-Hybrid**             | open hybrid (Gated DeltaNet + attention); independent of frontier vendors  |
| **Qwen3-Next-80B-A3B**      | 3:1 Gated DeltaNet : Gated Attention hybrid; MoE 3 B active; native 262 k |
| **Kimi Linear**             | independent 3:1 hybrid; cross-vendor comparison point                      |
| TTT-1.3B (research release) | reference test-time-training pretrained model                              |

Submission disclosure (PLAN_V3 spectrum) is mandatory: this regime is
implicitly L1 (full in-context) unless the strategy says otherwise.

### 4.3 Regime C — pretrained + finetuned

**Goal.** Combine language priors with specialized state-tracking;
land on or near the L11 oracle.

Recipe:
1. Take a Regime B base.
2. SFT on `dbemu-train` curriculum, ~1–10 B tokens.
3. Optional: RL with bits-recovered reward (PPO + dense shaping).
4. Optional: distill into a smaller recurrent student.

| Base                    | Adapter                | Tokens               | Notes                                                        |
| ----------------------- | ---------------------- | -------------------- | ------------------------------------------------------------ |
| Qwen2.5-3B              | LoRA r=64              | 2 B SFT              | cheap; first to ship                                          |
| Mamba-2.8B              | full-FT                | 5 B SFT + 0.5 B RL   | additive-update L2 candidate                                  |
| **Gated DeltaNet-1.5B** | full-FT                | 5 B SFT + 0.5 B RL   | **best small-RNN L2 candidate** (precision + retention)       |
| Llama-3.2-3B            | LoRA r=128             | 5 B SFT              | reference; broad applicability                                |
| Mistral-7B              | QLoRA r=64             | 2 B SFT              | budget option                                                 |
| Phi-3.5-mini            | full-FT                | 2 B SFT              | strong per-param, easy to fine-tune                           |
| **Qwen3-Next-80B-A3B**  | LoRA r=64 on GDN + attn| 2 B SFT (+ 0.5 B RL) | **highest-ceiling hybrid**; native 262 k for long-stream tasks|
| OLMo-Hybrid             | LoRA r=64              | 2 B SFT              | open hybrid; clean ablation against pure GDN                  |
| TTT-1.3B                | full-FT                | 2 B SFT              | tests whether test-time inner update helps T10                |

---

## 5. Concrete hyperparameters

### 5.1 From-scratch Mamba2-130M (the reference)

| Param                  | Value                                  |
| ---------------------- | -------------------------------------- |
| $d_\text{model}$       | 768                                    |
| $d_\text{state}$       | 64                                     |
| layers                 | 24                                     |
| head dim               | 64                                     |
| vocab                  | 32 k (per PLAN_V3 §4 serialization)    |
| sequence length        | 8 k → 32 k (curriculum)                |
| optimizer              | AdamW, $\beta_1=0.9, \beta_2=0.95$     |
| lr                     | $3 \times 10^{-4}$, cosine to $1e{-5}$ |
| warmup                 | 2 k steps                              |
| weight decay           | 0.1                                    |
| dropout                | 0                                      |
| batch (tokens)         | 256 k                                  |
| total tokens           | 10 B (smoke); 50 B (full)              |
| precision              | bf16 + fp32 master                     |
| state init             | zero each episode                      |

### 5.2 From-scratch LSTM-50M (sanity baseline)

| Param                  | Value                          |
| ---------------------- | ------------------------------ |
| $d_h$                  | 1024 (h and c)                 |
| layers                 | 6                              |
| dropout (intra)        | 0.1                            |
| optimizer              | AdamW                          |
| lr                     | $1 \times 10^{-3}$             |
| clip                   | 1.0                            |
| batch                  | 256 k tokens                   |
| total tokens           | 10 B                           |

### 5.3 SFT recipe on pretrained Mamba-2.8B

| Param                  | Value                          |
| ---------------------- | ------------------------------ |
| precision              | bf16                           |
| lr                     | $5 \times 10^{-5}$, cosine     |
| weight decay           | 0                              |
| batch                  | 64 k tokens                    |
| sequence              | 16 k (truncated curriculum)    |
| steps                  | 80 k (~5 B tokens)             |
| loss weight (next-op)  | 0.1                            |
| loss weight (target)   | 1.0                            |
| RL stage               | PPO, KL coef 0.05, 5 k steps   |
| reward                 | bits-recovered + 0.1 × shape   |

### 5.4 LoRA on pretrained transformer (Qwen2.5-3B)

| Param        | Value                          |
| ------------ | ------------------------------ |
| rank         | 64                             |
| alpha        | 128                            |
| target       | q,k,v,o + mlp gates            |
| precision    | bf16, base in 4-bit (QLoRA)    |
| lr           | $2 \times 10^{-4}$             |
| batch        | 32 k tokens                    |
| sequence     | 8 k                            |
| steps        | 60 k                           |

### 5.5 From-scratch Gated DeltaNet-130M

| Param                  | Value                                            |
| ---------------------- | ------------------------------------------------ |
| $d_\text{model}$       | 768                                              |
| heads (delta)          | 16                                               |
| head dim (key, value)  | 48, 48                                           |
| layers                 | 24                                               |
| short-conv on q,k,v    | width 4 (per Yang et al. 2024)                   |
| gate                   | sigmoid $\alpha$, softplus $\beta$               |
| optimizer              | AdamW, $\beta_1=0.9, \beta_2=0.95$               |
| lr                     | $3 \times 10^{-4}$, cosine to $1 \times 10^{-5}$ |
| warmup                 | 2 k steps                                        |
| weight decay           | 0.1                                              |
| batch (tokens)         | 256 k                                            |
| sequence length        | 8 k → 32 k (curriculum)                          |
| total tokens           | 10 B (smoke); 50 B (full)                        |
| precision              | bf16 + fp32 master                               |
| state init             | zero each episode                                |

### 5.6 From-scratch TTT-Linear-130M

| Param                  | Value                                       |
| ---------------------- | ------------------------------------------- |
| $d_\text{model}$       | 768                                         |
| inner model            | linear, $W \in \mathbb{R}^{d_k \times d_v}$ |
| $d_k = d_v$ (inner)    | 64                                          |
| inner update           | SGD with learnable per-layer lr             |
| chunked update size    | 16 (per Sun et al. 2024)                    |
| outer layers           | 24                                          |
| outer optimizer        | AdamW                                       |
| outer lr               | $1 \times 10^{-3}$, cosine                  |
| batch (tokens)         | 256 k                                       |
| sequence length        | 8 k → 32 k                                  |
| total tokens           | 10 B                                        |
| precision              | bf16                                        |

### 5.7 From-scratch Hybrid GDN+Attn-130M (Qwen3-Next-style)

| Param                  | Value                                                |
| ---------------------- | ---------------------------------------------------- |
| $d_\text{model}$       | 768                                                  |
| layer pattern          | 3 × Gated DeltaNet, 1 × Gated Attention (repeat × 6) |
| GDN heads / head dim   | 16 / 48                                              |
| Attn heads / head dim  | 12 / 64                                              |
| attn variant           | gated attention with RoPE                            |
| attn window            | full (within native ctx)                             |
| total layers           | 24 (18 GDN + 6 attn)                                 |
| optimizer              | AdamW                                                |
| lr                     | $2.5 \times 10^{-4}$, cosine                         |
| batch (tokens)         | 256 k                                                |
| sequence length        | 8 k → 32 k                                           |
| total tokens           | 10 B                                                 |
| precision              | bf16                                                 |

### 5.8 LoRA on Qwen3-Next-80B-A3B (hybrid finetune)

| Param      | Value                                                              |
| ---------- | ------------------------------------------------------------------ |
| rank       | 64 (separate adapters for GDN and attention blocks)                |
| alpha      | 128                                                                |
| target     | GDN q/k/v/o + attn q/k/v/o + MLP gates                             |
| precision  | bf16, base in 4-bit                                                |
| lr         | $1 \times 10^{-4}$ (lower than for pure-attention LoRA)            |
| batch      | 32 k tokens                                                        |
| sequence   | 32 k (selectively trains on the native 262 k window)               |
| steps      | 60 k                                                               |
| notes      | freeze MoE router; train only experts active in the curriculum mix |

### 5.9 Full-FT on Gated DeltaNet-1.5B (open pretrained)

| Param        | Value                          |
| ------------ | ------------------------------ |
| precision    | bf16                           |
| lr           | $5 \times 10^{-5}$, cosine     |
| weight decay | 0                              |
| batch        | 64 k tokens                    |
| sequence     | 16 k                           |
| steps        | 80 k                           |
| RL stage     | PPO, KL coef 0.05, 5 k steps   |
| reward       | bits-recovered + 0.1 × shape   |

---

## 6. Pressuring hidden state — concrete recipes

The architecture-purity signal requires forcing the model to *use*
its state instead of relying on surface patterns or oracle leakage.

### 6.1 Capacity-pressure variants (state)

For each capacity task, sweep one parameter to the breaking point:

- **T6-hash sweep.** Fix value width $v = 32$, sweep $K$ over
  $\{10, 30, 100, 300, 1{,}000, 3{,}000, 10{,}000, 30{,}000\}$.
  Each model's bits-recovered curve flattens at its own
  $K_{\max} \approx \text{state bits} / v$.
- **T7-counter sweep.** Fix max-count $= 256$ ($\log_2 = 8$),
  sweep $K$ the same way.
- **T10-mvcc sweep.** Fix table size, sweep snapshot count
  $S \in \{1, 3, 10, 30, 100\}$.

Reporting these three curves on one plot is the capacity diagnostic.

### 6.2 Anti-leak conventions

To stop the model from "cheating" by re-reading earlier ops:

- **Single-pass evaluation.** The op stream is presented once;
  queries arrive only after `<eos-stream>`. No replay.
- **Query-far-from-write window.** Queries reference cells whose
  last write is at least $L_{\min}$ ops in the past (configurable;
  default 100).
- **Prompt scrub.** For transformers, the harness optionally drops
  KV for the op stream before reading queries, so only the model's
  *summary* (or its recurrent state) can answer. Equivalent to L4
  (self-summary) at the harness level.
- **Distractor ops.** Insert ops on non-queried rows interleaved
  with the writes the query depends on, so a heuristic "remember
  the last few writes" baseline fails.

### 6.3 Snapshot-history pressure (T10 specifically)

T10 is the cleanest state-pressure task because **the snapshot
count $S$ is what scales the requirement**.

Two variants:

- **T10-fixed-snapshot.** All $S$ snapshots are pre-declared at
  stream construction; the model knows which op indices it will be
  asked about. Tests targeted retention — model can compress
  selectively.
- **T10-blind-snapshot.** Snapshot indices are revealed only at
  query time. Tests *uniform* retention — the model must keep all
  history. This is the strict capacity test.

For a transformer with context $C$ tokens and a stream of $N \gg C$
tokens, blind-snapshot at depth $S$ requires the model to encode
~$S$ full table states into the *final* $C$ tokens of its window.
For a recurrent model, the same requirement falls on $h_N$.

---

## 7. Pressuring KV cache — concrete recipes

The user's specific scenario: *KV is built by attending to DB tokens
during the stream; the model must then reason from those latents
about the last several ops before emitting the final state.*

### 7.1 Context-overflow pressure

- **Long-stream variant.** Stream length $L$ chosen so that
  serialized tokens $\geq 2 C$ (twice the context).
- Forces eviction; cliff is diagnostic.
- Combine with T5 (long-range) to get the "needle past the window
  boundary" failure mode that StreamingLLM / H2O were designed to
  fix and that the bench *measures*.

### 7.2 Latent-reasoning pressure

This is the "reason from latents over the last several ops" setup:

```
[ DB op tokens × N_stream ]  ←  consume context, build KV
[ <eos-stream> ]
[ <think> × M_silent ]       ←  no new info; model may emit hidden
                                 scratchpad tokens (PonderNet-style)
[ <query> ... <answer> ]     ←  readout
```

The window must be sized so that after `<eos-stream>` and the silent
slot, the model can no longer re-attend to the *earliest* ops.
Equivalently: the answer must be encoded in the **last $W$ KV slots**
of the stream, where $W \ll N_\text{stream}$.

Two knobs:

- **$N_\text{stream}$** — pressures KV capacity (must compress).
- **$M_\text{silent}$** — pressures latent reasoning depth (must
  *use* what is compressed).

Sweep both. The diagnostic plot is bits-recovered as a function of
$M_\text{silent}$ at fixed $N_\text{stream}$. A model that does no
latent reasoning has a flat line. A model that does only
shallow-latent reasoning has a curve that rises and then plateaus.

This is the perception/reasoning analogue of the idle-tick axis
$\rho$ from PLAN_V3 §3 / [CONTINUOUS_AGENT.md](CONTINUOUS_AGENT.md):
$M_\text{silent}$ on the *output side*, $\rho$ on the *input side*.

### 7.3 Attention-budget pressure

For a transformer:

- **KV-byte budget.** Cap KV at a target size (e.g. 64 MB) regardless
  of context; harness ejects per declared strategy (sliding,
  H2O-score, summary).
- **Per-op KV quota.** Force the model to summarize: each op is
  followed by exactly $k$ "digest" tokens; only those persist.
  Strict version of L4 (self-summary).

These reproduce the conditions a `dbemu-static` "limited-context"
submission must operate in.

### 7.4 Forcing the latent path (the "no peeking" trick)

If a transformer keeps the full op stream in KV, it doesn't have to
do latent reasoning — it just re-attends. To *force* latent
reasoning, mask attention from the answer region back to the op
region. The model must rely on whatever it folded into the
intermediate hidden states between `<eos-stream>` and `<answer>`.

This is implemented at the attention-mask level:

```
mask[i, j] = 0 if j ≤ eos_stream_idx and i > eos_stream_idx + small_W
```

Models trained or evaluated under this mask are scored under the
strictest L1/L4 boundary. This is the natural transformer analogue
of an RNN-only evaluation; it lets a transformer compete in the L2
column of the spectrum.

---

## 8. Task variants designed to pressure specific limits

A compact catalog of task variants, each annotated with which limit
it pressures and which architecture excels.

| Variant            | Recipe                                                | Pressures                | Best arch (a priori)        |
| ------------------ | ----------------------------------------------------- | ------------------------ | --------------------------- |
| `T1-recall-far`    | T1 with query lag $\geq 100$ ops, distractors         | state                    | Mamba2 / large LSTM         |
| `T5-cliff`         | T5 with lag $\in \{0.5C, C, 1.5C\}$ tokens            | KV eviction              | summary-checkpoint LLM      |
| `T6-cap`           | T6 sweep on $K$                                       | state capacity           | Mamba2 (big $d_s$)          |
| `T7-cap`           | T7 sweep on $K$                                       | state capacity           | Mamba2 / LSTM               |
| `T10-blind-S`      | T10 with $S$ unrevealed until query                   | state retention          | Mamba2 (large state)        |
| `Tn-overflow`      | any task at $L > 2C$ tokens                           | KV strategy              | RAG / summary               |
| `Tn-no-peek`       | answer masked from op KV                              | latent reasoning         | recurrent + ponder          |
| `Tn-silent-think`  | $M_\text{silent}$ tokens between stream and query     | latent compute budget    | reasoning LLM (o1-style)    |
| `Tn-stream-rate`   | idle ticks $\rho$ between ops                         | continuous-agent compute | recurrent + adaptive halt   |

The first three rows are the **state-capacity track**; the next two
are the **KV-strategy track**; the last three are the
**latent-reasoning track**. Each track has a clean "best a priori
arch" prediction that the bench can confirm or refute.

---

## 9. Baseline targets (what "good" looks like)

Numbers below are *targets to fill in* during the first release of
the bench, set as initial sanity goals. Bits recovered, normalized.

| Model                                            | `mini` | `static` macro | `T6-cap` $K=1000$ | `T10-blind` $S=10$ |
| ------------------------------------------------ | ------ | -------------- | ----------------- | ------------------ |
| Random baseline                                  | 0.00   | 0.00           | 0.00              | 0.00               |
| Pretrained LLM zero-shot (L1, 8 k ctx)           | 0.6 ?  | 0.4 ?          | 0.2 ?             | 0.1 ?              |
| Pretrained + sliding window (L3)                 | 0.5 ?  | 0.3 ?          | 0.1 ?             | 0.05 ?             |
| Pretrained + summary checkpoint (L4)             | 0.65 ? | 0.45 ?         | 0.3 ?             | 0.2 ?              |
| Pretrained + SQL tool (L9)                       | 0.95 ? | 0.9 ?          | 0.9 ?             | 0.9 ?              |
| LSTM-50M from scratch (L2)                       | 0.6 ?  | 0.35 ?         | 0.15 ?            | 0.1 ?              |
| Mamba2-130M from scratch (L2)                    | 0.8 ?  | 0.55 ?         | 0.5 ?             | 0.3 ?              |
| Mamba2-370M from scratch (L2)                    | 0.9 ?  | 0.7 ?          | 0.8 ?             | 0.5 ?              |
| **Gated DeltaNet-130M from scratch (L2)**        | 0.85 ? | 0.65 ?         | 0.75 ?            | 0.45 ?             |
| **Gated DeltaNet-370M from scratch (L2)**        | 0.93 ? | 0.78 ?         | 0.9 ?             | 0.6 ?              |
| **TTT-Linear-130M from scratch (L2)**            | 0.83 ? | 0.6 ?          | 0.65 ?            | 0.5 ?              |
| **Hybrid GDN+Attn-130M from scratch (L1/L2 mix)**| 0.88 ? | 0.72 ?         | 0.8 ?             | 0.5 ?              |
| Mamba-2.8B SFT + RL (L2)                         | 0.97 ? | 0.85 ?         | 0.95 ?            | 0.7 ?              |
| **Gated DeltaNet-1.5B SFT + RL (L2)**            | 0.97 ? | 0.88 ?         | 0.97 ?            | 0.78 ?             |
| **Qwen3-Next-80B-A3B SFT (L1/L2 mix)**           | 0.98 ? | 0.93 ?         | 0.97 ?            | 0.85 ?             |
| SQLite oracle (L11)                              | 1.00   | 1.00           | 1.00              | 1.00               |

The pattern matters more than the numbers. Five predictions the bench
should confirm:

1. **L9 tool-use dominates `mini` and `static` macro** but collapses
   on tasks where the tool can't see history (it can; predict L9
   wins everywhere except possibly latent-reasoning variants).
2. **Mamba-2.8B-FT beats Pretrained-LLM-zero-shot** even at L2 vs L1,
   *because* the bench rewards state tracking; this is the central
   architecture claim.
3. **T6-cap at $K=10^4$ separates Mamba2-370M from Mamba2-130M
   cleanly** along the capacity-curve cliff; LSTM-50M is far below
   both.
4. **Gated DeltaNet beats Mamba2 at matched params on T6 and T10**,
   confirming the delta-rule + gating advantage observed on MQAR and
   S-NIAH-3 in the open literature.
5. **The 3:1 hybrid (Qwen3-Next-style) beats either pure-recurrent
   or pure-transformer at matched params on T1 + T10 jointly** —
   precision from the few attention layers; retention from the
   linear bulk. If this fails to hold, the hybrid design choice is
   over-claimed.

---

## 10. Practical advice

- **For benchmark builders.** Ship Mamba2-130M from-scratch and one
  Mamba-2.8B SFT checkpoint with the first release. These two cover
  the most informative L2 points on every plot.
- **For submitters.** Disclose: spectrum level, context length,
  total FLOPs, total wall-time, state bits (if recurrent). The
  cost-normalized leaderboards become more informative than raw
  bits-recovered.
- **For architecture researchers.** Run only the capacity-curve and
  latent-reasoning tracks first. They give the highest information
  per training run about a new architecture's strengths.
- **For training-method researchers.** SFT-then-RL on Mamba-2.8B is
  the highest-ceiling configuration. The bits-recovered reward is
  smooth and decomposable, so RL converges; dense shaping is
  optional but helps early.
- **For prompt / strategy researchers.** All the action is on
  `dbemu-static`'s limited-context leaderboard with declared
  strategies. The interesting comparisons are L3 vs L4 vs L7.
