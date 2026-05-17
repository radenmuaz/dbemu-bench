# Related work

Positions `dbemu-bench` against four adjacent literatures: memory in
sequence models, long-context benchmarks, dual-process (System 1 /
System 2) reasoning, and continuous / always-on agents with adaptive
compute. Sources at the bottom.

---

## 1. Memory in sequence models

The strategy spectrum (L0–L11) in [PLAN_V3.md](PLAN_V3.md) collapses
several previously-separate lines of research onto a single axis. Each
of them is dense in its own right.

### 1.1 Architectural memory (L0–L2)

**SSM lineage.** Mamba [Gu & Dao 2023] introduced a selective
state-space model that scales linearly in sequence length and exposes
a fixed-size hidden state. Mamba-2 [Dao & Gu 2024] reformulated SSMs
and attention as two views of a structured-matrix class ("state-space
duality") and grew state size up to 16×, yielding substantially better
multi-query associative recall (MQAR). RWKV, Hyena, Mega, and
S4 / S5 sit in the same family.

**Delta-rule lineage.** A parallel line of work replaces the additive
SSM update with a **delta rule** that minimizes per-step prediction
MSE. **DeltaNet** [Yang, Schlag et al., 2024; building on Schlag et
al. 2021] makes the linear-attention RNN parallelizable while
delivering MQAR performance that surpasses Mamba per state. **Gated
DeltaNet** [Yang et al., ICLR 2025] adds gating for rapid memory
erasure on top of the delta rule's targeted updates, outperforming
both Mamba-2 and DeltaNet across language modeling, in-context
retrieval, and S-NIAH-3 (where Mamba-2 collapses on complex pattern
memorization while Gated DeltaNet remains strong). The Hazy Research
**Based** paper [Arora et al. 2024] earlier established that *pure*
linear attention cannot perform the precise local shifts and
comparisons required for associative recall — motivating the gating
and delta-rule machinery that followed.

**Test-time training (TTT).** Sun et al. [2024] reframe the recurrent
hidden state as a *learned mini-model* updated by SGD on the test
sequence itself; the layer is a step of self-supervised learning.
**TTT-Linear** and **TTT-MLP** instantiate the inner model as a
linear map or a 2-layer MLP, and continue to reduce perplexity past
16 k tokens where Mamba's perplexity plateaus. This is the cleanest
recent answer to the question "what does an RNN's hidden state want
to *be*?": an optimizer over an inner predictor.

**Optimizable long-term memory modules.** **Atlas** [Behrouz et al.
2025] and the Titans family treat memory as an explicitly-optimized
module updated to memorize the *current window*, not just the last
token, and report best-in-class MQAR per memory size plus +80 %
accuracy at 10-million-token BABILong. This is the strongest
published signal that targeted memory optimization beats passive
state accumulation for long-horizon retention.

**Hybrid architectures (shipped at scale).** Production frontier
open-weight models have converged on a small-fraction full-attention
hybrid. **Qwen3-Next-80B-A3B** and **Kimi Linear** both use a
**3:1 ratio** of Gated DeltaNet to Gated Attention layers, with a
native 262 k context and no eviction. The hybrid pattern restores
the precision needed for shifts/comparisons via the sparse
full-attention layers while keeping linear cost on the bulk of the
model. **OLMo-Hybrid** offers a fully open-weights version of the
same recipe. Gated DeltaNet itself has been incorporated into
Qwen3.5, Qwen3-Next, and OLMo-Hybrid as the linear-attention layer.

**Why this matters here.** L2 (recurrent / fixed-state) of the
spectrum is the most informative regime for *architecture* research:
metrics are reported per recurrent state bit, exposing how efficiently
an architecture packs structured state. `dbemu-bench`'s `dbemu-train`
diagnostic — the **capacity curve** — is the same shape of measurement
the Mamba-2, DeltaNet, Gated DeltaNet, and Atlas papers apply to
MQAR / S-NIAH, generalized from a single recall task to an entire
DB-emulation taxonomy. The bench's `T6-hash` and `T10-mvcc` are the
direct stress tests this lineage already implicitly competes on.

**Augmented memory (historical).** Neural Turing Machines and
Differentiable Neural Computers proposed external memory banks
controlled by a recurrent network. The agentic-memory work
(§1.4 below) and the Atlas / Titans line are contemporary,
LLM-shaped reincarnations of that research program.

### 1.2 KV management for transformers (L3–L5)

A large 2023–2025 literature treats the KV-cache as a managed
resource. The standard moves:

- **StreamingLLM**: keep a fixed prefix ("attention sinks") plus a
  sliding window of recent KV. Fast and hardware-friendly but
  discards semantically critical middle-context tokens.
- **H2O (Heavy-Hitter Oracle)**: rank tokens by historical attention
  mass and keep the heaviest. ~29× throughput improvement at 20%
  retention on OPT-30B.
- **SnapKV**: predict prefill-time importance from an end-of-prompt
  observation window.
- **RefreshKV** (2025): periodically refresh which tokens are kept
  rather than relying on monotone eviction.
- **PagedEviction** (2025): block-wise eviction integrated with
  PagedAttention.
- A taxonomy of methods divides into **dropping** (H2O, SnapKV),
  **quantization** (CacheGen, KIVI), **merging** (HOMER, Look-M), and
  **prompt compression** (LLMLingua, LongLLMLingua).

Recent meta-analyses ([Pitfalls of KV Cache Compression], [Taming
Fragility]) report that aggressive compression silently degrades
multi-hop and arithmetic tasks — exactly the failure modes
`dbemu-bench`'s T5 (long-range) and T7 (counter array) are designed
to surface.

### 1.3 External / retrieval-based memory (L6–L7)

REALM, RAG, RETRO, and Memorizing Transformers externalize knowledge
to a retrieved set of passages or KV chunks. In our spectrum these are
L6 (text scratchpad) and L7 (opaque external KV); they trade quadratic
attention for tool-call latency, and they outsource the "remembering"
to a retriever whose quality dominates end-task performance.

### 1.4 Agentic memory (L7–L10 in practice)

The current frontier of LLM-agent memory frames the context window as
the smallest tier of a memory hierarchy.

- **MemGPT** [Packer et al. 2023] models the context window as RAM
  and an external store as disk; the LLM issues read/write calls to
  page between them.
- **A-MEM** [Xu et al. 2025] organizes memory as a Zettelkasten-style
  network of notes with contextual descriptions, keywords, and tags
  that are dynamically linked.
- **AgeMem** [2025] integrates long- and short-term memory operations
  directly into the agent's policy, exposing store / retrieve /
  update / summarize / discard as actions.
- **Letta** is the production successor to MemGPT; their 2025
  benchmark suite finds that on temporal queries and multi-hop
  reasoning recent agentic-memory systems deliver +20–30 point gains
  over flat RAG baselines.
- **Mem0**'s 2026 state-of-agent-memory survey catalogues the
  production gap between research and deployed memory systems.

**Relation to `dbemu-bench`.** Agentic-memory papers benchmark on
ad-hoc multi-session dialogue corpora; the failure modes that show up
(temporal queries, multi-hop, fact updates) map directly to
`dbemu-bench`'s T1 (read-after-write), T5 (long-range), T8 (join), and
T10 (MVCC snapshot). The bench provides a *grammar* and a *ground
truth* for these capabilities that dialogue corpora cannot.

---

## 2. Long-context benchmarks (the prior art `dbemu-bench` is closest to)

- **Needle-in-a-Haystack (NIAH).** Insert a short string into a long
  document; recover it. The minimal long-context probe.
- **RULER** [NVIDIA, 2024]. Extends NIAH with multiple needles,
  multi-hop tracing, and aggregation tasks; introduced the "real
  context size" framing that decouples claimed window size from
  effective recall.
- **BABILong** [NeurIPS 2024]. Embeds bAbI reasoning chains in
  arbitrarily long haystacks (up to 10 M tokens). Tests
  reasoning-across-facts in long documents.
- **LongBench / LongBench v2.** Real and synthetic mixed-domain tasks
  with documents up to ~40 k tokens; the canonical
  general-long-context benchmark cited by most 2024–2025 papers.
- **InfiniteBench, LongGenBench, Counting-Stars.** Variants targeting
  long *outputs*, counting, and aggregation.

**Where `dbemu-bench` differs.** All of the above use natural-language
documents. The "ground truth" is whatever a human annotator wrote.
`dbemu-bench` replaces the document with a typed operation stream and
the annotator with a SQLite executor, so the ground truth is a
mechanical function of the input — making partial-credit metrics like
*bits recovered* and the *coverage × accuracy* decomposition
well-defined. It also adds two axes those benchmarks lack: the
**strategy spectrum** (so an agent with a SQL tool can compete on the
same task as a raw transformer) and **idle ticks per op** for
continuous-agent evaluation.

---

## 3. System 1 / System 2 in LLMs

The Kahneman dual-process framing — fast intuitive (System 1) vs.
slow deliberate (System 2) — has been imported wholesale into the
LLM literature.

### 3.1 The claim, simplified

Stock pretrained LLMs are commonly characterized as
**System-1-shaped**: fast, fluent, and bad at the kind of careful
multi-step reasoning that requires backtracking, verification, and
search. The 2024–2025 wave of "reasoning models" — OpenAI's o1 / o3,
DeepSeek R1, QwQ — is framed as a shift toward **System-2-shaped**
inference: long internal chains of thought, self-verification, and
inference-time search.

### 3.2 Methods labelled "System 2"

- **Chain-of-Thought (CoT)**, **Self-Consistency**, **Tree of
  Thoughts (ToT)** — promote step-wise reasoning at inference time.
- **ReAct**, **Reflexion**, **Self-Refine** — interleave reasoning
  with action and self-critique.
- **Process reward models (PRM)** and search-based decoding —
  evaluate intermediate steps, not just final answers.
- **Latent / continuous-thought reasoning** [2025] — proposes hidden
  scratchpad tokens that aren't decoded but still incur compute,
  closer in spirit to PonderNet (§4).
- **Reasoning on a Spectrum** [Feb 2025] — argues against binary
  System-1-vs-System-2 framing and aligns LLMs to a continuum.

### 3.3 The "overthinking" problem

A growing 2025 sub-literature documents that explicit System-2
prompting *over-applied* yields longer, more expensive, and
sometimes *worse* answers — "overthinking on easy problems,
underthinking on hard ones." This is the negative result that
motivates the next section: instead of "more System 2," prefer
*adaptive* compute that allocates more thought where it is actually
useful.

### 3.4 Relation to `dbemu-bench`

DB emulation forces System-2-style state tracking by construction:
a long operation stream cannot be answered by surface-form pattern
matching. But it also rewards System-1-style amortization — a
recurrent model that has internalized the update rule pays no
deliberation cost at all. The **bits-recovered / cost** Pareto
frontier across the spectrum is therefore a direct measurement of how
each model balances the two modes per unit work. The benchmark does
not take a side; it gives both modes a place to win.

---

## 4. Continuous / always-on agents and adaptive compute

This is the section closest to [CONTINUOUS_AGENT.md](CONTINUOUS_AGENT.md).

### 4.1 Adaptive computation

- **ACT** [Graves 2016]. RNNs learn how many internal steps to take
  per input.
- **PonderNet** [Banino et al. 2021]. Reformulates halting as a
  probabilistic decision; trains a halting head that emits a Bernoulli
  per step.
- **FR-Ponder** [2025]. Reports 30–50 % token / FLOP reduction on
  GSM8K, MATH-500, GPQA at matched or improved accuracy by
  instance-adaptive halting.
- **Awesome-Adaptive-Computation** — community-curated reading list.
- **Adaptive Test-Time Compute Allocation** [2025] — uses constrained
  policy optimization to budget compute per problem.
- **Input-adaptive allocation of LM computation** [ICLR 2025] —
  generalizes the per-token compute decision to LLM serving.

A 2025 survey identifies systemic inefficiencies in current
test-time compute strategies: *underthinking on hard problems,
overthinking on easy ones, limited sensitivity to task complexity*.
This is exactly what an *idle-ticks-per-op* axis ($\rho$ in
`dbemu-env`) lets one measure directly.

### 4.2 Background / unprompted agent behavior

A 2025 NeurIPS workshop paper, **What Do LLM Agents Do When Left
Alone?**, instruments idle agents and reports the spontaneous
emergence of structured reflection-planning loops in continuous ReAct
architectures with persistent memory and self-feedback. This is the
empirical counterpart to the continuous-agent thesis: idle time is
not wasted, and reasonable architectures do non-trivial things in it.

**Always-On OS-tuning agents** (NeurIPS ML4Sys 2025) and **Thought
Management Systems** for long-horizon goal-driven agents (Sci. Direct
2025) explore the engineering side: how to keep an LLM agent
productive over hours or days without explicit human-issued tasks.

### 4.3 Relation to `dbemu-bench`

`dbemu-env` includes $\rho \in \{0, 1, 4, 16, 64, 256\}$ idle ticks
per op and reports two derivatives:

- $\Delta_\rho$ — accuracy gain per unit idle tick (a purely reactive
  architecture has $\Delta_\rho = 0$).
- **Idle-compute efficiency** — bits recovered per idle-tick FLOP.

To our knowledge no existing benchmark exposes a continuous-agent
axis with a ground-truth scoring function. The closest analogues
(NIAH / RULER / BABILong) all evaluate a *single forward pass* over a
long context; none scores what happens between operations.

---

## 5. Where `dbemu-bench` sits

| Adjacent area                    | Closest works                            | What `dbemu-bench` adds                                                  |
| -------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------ |
| Long-context benchmarks          | RULER, BABILong, LongBench, NIAH         | typed-op streams, executor-grounded truth, bits-recovered scoring        |
| KV / memory compression          | StreamingLLM, H2O, SnapKV, RefreshKV     | per-task locality/recency profile diagnoses failure mode                  |
| Agentic memory                   | MemGPT, A-MEM, Letta, Mem0               | grammar + oracle for the capabilities those systems claim to enable      |
| State-space / recurrent          | Mamba, Mamba-2, RWKV, S4/S5              | capacity curve and state-bit-normalized metrics across many tasks         |
| Delta-rule / TTT / Atlas         | DeltaNet, Gated DeltaNet, TTT, Atlas     | T6 / T10 as direct extensions of MQAR / S-NIAH-3 / window-memory         |
| Hybrid linear + softmax          | Qwen3-Next, Kimi Linear, OLMo-Hybrid     | per-task precision vs. retention attribution between layer types         |
| System-1 / System-2 reasoning    | o1, R1, QwQ; CoT, ToT, ReAct, Reflexion  | one task family that rewards System-1 amortization *and* System-2 search |
| Adaptive compute                 | ACT, PonderNet, FR-Ponder                | $\rho$-axis with a ground-truth target for "thinking while idle"          |
| Cognitive architectures (older)  | Soar, ACT-R, NTM, DNC                    | a measurable proving ground for the same intuitions in modern stack      |

The aim is not to displace any of the above. It is to provide one
benchmark where progress on **any** of them lands on a comparable
scalar (normalized bits recovered) at a comparable cost denominator
(per token / per state bit / per tool call), so the field can finally
plot a single (cost, fidelity) Pareto frontier across the entire
spectrum from in-weights memorization to full SQL delegation.

---

## Sources

### Memory and KV management
- [Top 10 KV Cache Compression Techniques (MarkTechPost, 2026)](https://www.marktechpost.com/2026/04/29/top-10-kv-cache-compression-techniques-for-llm-inference-reducing-memory-overhead-across-eviction-quantization-and-low-rank-methods/)
- [Reformulating KV Cache Eviction Problem for Long-Context LLM Inference](https://arxiv.org/html/2605.07234v1)
- [Hierarchical KV (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/de7dc701a2882088f3136139949e1d05-Paper-Conference.pdf)
- [The Pitfalls of KV Cache Compression](https://arxiv.org/pdf/2510.00231)
- [Taming the Fragility of KV Cache Eviction](https://arxiv.org/html/2510.13334v1)
- [EvicPress: Joint KV-Cache Compression and Eviction](https://arxiv.org/html/2512.14946)

### Agentic memory
- [A-MEM: Agentic Memory for LLM Agents (arXiv)](https://arxiv.org/abs/2502.12110)
- [A-MEM PDF](https://arxiv.org/pdf/2502.12110)
- [Letta: Agent Memory blog](https://www.letta.com/blog/agent-memory)
- [Letta: Benchmarking AI Agent Memory](https://www.letta.com/blog/benchmarking-ai-agent-memory)
- [Letta / MemGPT docs](https://docs.letta.com/concepts/memgpt/)
- [Mem0: State of AI Agent Memory 2026](https://mem0.ai/blog/state-of-ai-agent-memory-2026)
- [Memory for Autonomous LLM Agents: Mechanisms, Evaluation, Frontiers](https://arxiv.org/html/2603.07670v1)
- [AgeMem: Unified Long- and Short-Term Memory (arXiv)](https://arxiv.org/abs/2601.01885)
- [Agent Memory Paper List (Shichun-Liu, GitHub)](https://github.com/Shichun-Liu/Agent-Memory-Paper-List)

### State-space / recurrent
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- [Mamba-2 (Goomba Lab, Part I)](https://goombalab.github.io/blog/2024/mamba2-part1-model/)
- [Mamba-2 (Goomba Lab, Part II — Theory)](https://goombalab.github.io/blog/2024/mamba2-part2-theory/)
- [Mamba-2 (Tri Dao blog)](https://tridao.me/blog/2024/mamba2-part1-model/)
- [Mamba SSM (GitHub)](https://github.com/state-spaces/mamba)

### Delta-rule and test-time-training RNNs
- [Parallelizing Linear Transformers with the Delta Rule over Sequence Length (DeltaNet, NeurIPS 2024)](https://arxiv.org/abs/2406.06484)
- [DeltaNet Explained — Songlin Yang's blog (Part I)](https://sustcsonglin.github.io/blog/2024/deltanet-1/)
- [Gated Delta Networks: Improving Mamba2 with Delta Rule (ICLR 2025)](https://arxiv.org/abs/2412.06464)
- [Gated DeltaNet (NVlabs, GitHub)](https://github.com/NVlabs/GatedDeltaNet)
- [FG2-GDN: Doubly Fine-Grained Gated Delta Networks](https://arxiv.org/html/2604.19021v1)
- [Based: Simple linear attention language models balance the recall-throughput tradeoff (Arora et al.)](https://arxiv.org/html/2402.18668v1)
- [Gated Linear Attention with Hardware-Efficient Training (GLA)](https://arxiv.org/pdf/2312.06635)
- [Learning to (Learn at Test Time): RNNs with Expressive Hidden States (TTT, Sun et al. 2024)](https://arxiv.org/abs/2407.04620)
- [TTT-LM (PyTorch, GitHub)](https://github.com/test-time-training/ttt-lm-pytorch)
- [TTT-LM (JAX, GitHub)](https://github.com/test-time-training/ttt-lm-jax)
- [ATLAS: Learning to Optimally Memorize the Context at Test Time](https://arxiv.org/abs/2505.23735)

### Hybrid linear-attention + softmax (production-scale)
- [Qwen3-Next-80B-A3B-Instruct (Hugging Face)](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)
- [Qwen3-Next official page](https://qwen3-next.com/)
- [vLLM: Qwen3-Next hybrid architecture](https://blog.vllm.ai/2025/09/11/qwen3-next.html)
- [NVIDIA Tech Blog: Qwen3-Next hybrid MoE](https://developer.nvidia.com/blog/new-open-source-qwen3-next-models-preview-hybrid-moe-architecture-delivering-improved-accuracy-and-accelerated-parallel-processing-across-nvidia-platform/)
- [Qwen3.5: Nobody Agrees on Attention Anymore (Hugging Face blog)](https://huggingface.co/blog/mlabonne/qwen35)
- [Gated DeltaNet (Sebastian Raschka, LLMs-from-scratch)](https://sebastianraschka.com/llms-from-scratch/ch04/08_deltanet/)
- [Beyond Standard LLMs (Sebastian Raschka)](https://magazine.sebastianraschka.com/p/beyond-standard-llms)
- [qwen3.5-gated-deltanet-analysis (gist)](https://gist.github.com/justinchuby/0213aa253664fb72e9adb0089816de15)

### Long-context benchmarks
- [RULER (GitHub, NVIDIA)](https://github.com/NVIDIA/RULER)
- [RULER (OpenReview)](https://openreview.net/pdf?id=kIoBbc76Sy)
- [BABILong (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/c0d62e70dbc659cc9bd44cbcf1cb652f-Paper-Datasets_and_Benchmarks_Track.pdf)
- [BABILong (GitHub)](https://github.com/booydar/babilong)
- [LongBench v2 (ACL 2025 findings)](https://aclanthology.org/2025.findings-acl.903.pdf)
- [LongGenBench (OpenReview)](https://openreview.net/forum?id=3A71qNKWAS)

### System 1 / System 2 reasoning
- [Reasoning on a Spectrum: Aligning LLMs to System 1 and System 2](https://arxiv.org/pdf/2502.12470)
- [Exploring System 1 / System 2 communication for latent reasoning](https://arxiv.org/html/2510.00494v1)
- [System 1 / System 2 in LLMs (WaterCrawl blog)](https://watercrawl.dev/blog/Unlocking-the-Mind-of-AI-System-1-and-System-2)
- [System 1 vs System 2 in Modern AI (Gloqo)](https://www.gloqo.ai/insights/combining_system_1_and_system_2_thinking/)

### Continuous / always-on agents
- [What Do LLM Agents Do When Left Alone? (arXiv HTML)](https://arxiv.org/html/2509.21224v1)
- [What Do LLM Agents Do When Left Alone? (PDF)](https://arxiv.org/pdf/2509.21224)
- [LLM Agents for Always-On OS Tuning (NeurIPS ML4Sys 2025)](https://liargkovas.com/assets/pdf/Liargkovas_ML4Sys_NeurIPS25.pdf)
- [Thought Management System for long-horizon goal-driven LLM agents (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S1877750325002170)
- [LLM Powered Autonomous Agents (Lilian Weng)](https://lilianweng.github.io/posts/2023-06-23-agent/)

### Adaptive computation
- [PonderNet: Learning to Ponder (OpenReview PDF)](https://openreview.net/pdf?id=1EuxRTe0WN)
- [Learning to Ponder: Adaptive Reasoning in Latent Space (arXiv 2025)](https://arxiv.org/html/2509.24238)
- [Adaptive Test-Time Compute Allocation via Constrained Policy Opt.](https://arxiv.org/html/2604.14853v1)
- [Survey of Adaptive and Controllable Test-Time Compute](https://arxiv.org/pdf/2507.02076)
- [Input-Adaptive Allocation of LM Computation (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/ff414825df833edb8b1839e3d5d495e9-Paper-Conference.pdf)
- [Awesome Adaptive Computation (GitHub)](https://github.com/koayon/awesome-adaptive-computation)
- [Noteworthy LLM Research Papers of 2024 (Sebastian Raschka)](https://sebastianraschka.com/blog/2025/llm-research-2024.html)
