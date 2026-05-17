# Extending dbemu-bench to multimodal perception and control

This note maps the database-emulation benchmark onto the harder
problem its continuous-agent thesis was already pointing at:
**perception streams at multiple timescales, with control outputs,
where lossy is fine and partial-correctness is the norm — up to a
threshold past which the system fails outright**.

Reading order: [PLAN_V3.md](PLAN_V3.md) →
[CONTINUOUS_AGENT.md](CONTINUOUS_AGENT.md) → this file.

---

## 1. Why this extension is the natural next step

The continuous-agent argument in
[CONTINUOUS_AGENT.md](CONTINUOUS_AGENT.md) already assumes that real
agents have an internal clock decoupled from input arrival. That
assumption is mild in the DB setting (the op stream is still
discrete, typed, exact) and load-bearing in any perception/control
setting (sensors arrive continuously, at different rates, and the
control output must be emitted on time whether the agent has "thought
enough" or not).

The DB grammar was chosen because it has a *mechanical ground truth*
(SQLite). To extend to perception/control we need an analogous ground
truth that does **not** demand exact match — because exact match is
not the right semantics. A pixel off by one in luminance, a control
torque off by 0.1 %, an audio sample with quantization noise: none
of these are wrong in the DB sense. They are *good enough* up to a
task-dependent **tolerance band**, and they are *catastrophically
wrong* below it. That asymmetry is the central object this note
proposes to benchmark.

---

## 2. Five axes of mismatch with the current bench

| Axis           | Current (`dbemu-bench`)         | Multimodal extension                                       |
| -------------- | ------------------------------- | ---------------------------------------------------------- |
| Op type        | discrete, typed                 | continuous-valued tensor frames (image, audio, proprio)    |
| Time           | one ordered op stream           | several streams at different rates, partially async        |
| Ground truth   | SQLite-deterministic            | stochastic simulator + perceptual reference                |
| Correctness    | per-cell exact (with cell-F1)   | tolerance-banded equivalence; cliff outside band           |
| Information    | lossless aspiration             | lossy by design — compression is the goal, not the failure |

Anything below addresses one of these five axes.

---

## 3. Tolerance-banded scoring (the load-bearing primitive)

A perceptual or control task has a **piecewise** quality function:

- Within tolerance band $\varepsilon$, all answers are equivalent —
  the model gets full credit.
- Within an extended "graceful degradation" band
  $[\varepsilon, \varepsilon_{\max}]$, credit decays smoothly.
- Beyond $\varepsilon_{\max}$, credit is **zero** — the system has
  failed (image is unrecognizable, audio is unintelligible, controller
  is unstable, robot crashes).

Formally, define a task-specific equivalence relation
$\sim_\varepsilon$ on outputs (e.g., $\|p - t\|_2 < \varepsilon$ in
some embedding, or PSNR $> 30$ dB, or no-fall in a control rollout).
Then generalize the bits-recovered scalar to
$$
  I_{\sim}(T; P) \;=\; H([T]_\sim) \;-\; H([T]_\sim \mid P),
$$
where $[T]_\sim$ is the equivalence class of the ground truth.
Inside the band, $P$ and $T$ are in the same class — full mutual
information. Outside $\varepsilon_{\max}$, the prediction is mapped
to a sink class — zero mutual information.

This is the same scalar the DB bench uses, *modulo a task-specific
equivalence relation*. That is the only change.

### Threshold metrics that already exist (per modality)

| Modality          | In-band metric                      | Cliff condition                  |
| ----------------- | ----------------------------------- | -------------------------------- |
| Image             | SSIM, LPIPS, PSNR ≥ 30 dB           | unrecognizable; LPIPS > 0.4      |
| Audio (speech)    | PESQ, MOS ≥ 3.5                     | unintelligible; WER > 50 %        |
| Audio (music)     | ViSQOL, perceptual hash distance    | format-broken                    |
| Video             | VMAF, temporal coherence            | dropped frames, sync break       |
| Continuous ctrl   | episode reward, stability margin    | termination (fall, crash)        |
| Trajectory        | Hausdorff to safe set               | exit safe set                    |
| Language (mm out) | BLEURT / chrF / semantic equivalence| factually wrong                  |

The bench would not invent new perceptual metrics; it would *adopt*
each modality's standard one, plus a cliff condition.

---

## 4. Multi-timescale memory and action

The continuous-agent axis $\rho$ becomes **per stream**:
$\rho_{\text{vision}}, \rho_{\text{audio}}, \rho_{\text{proprio}},
\rho_{\text{action}}$. A vision sensor at 30 Hz, audio at 16 kHz,
proprio at 1 kHz, and a control loop at 100 Hz force the model to:

1. Maintain a fast inner loop (control / reflex).
2. Maintain a slow outer loop (planning, memory consolidation).
3. Bridge them with a hierarchy of summaries — exactly the
   hierarchical-summary strategy from L5 of the spectrum, now
   *required* by the rate mismatch rather than optional.

The DB benchmark surfaces this as a *capability*. The multimodal
extension surfaces it as a *necessity*.

### Cross-rate fusion as a task

Specific test: control a target with proprio at 1 kHz and vision at
3 Hz where the target is occasionally lost. The model must *predict
through* the slow stream during occlusion. This is the perception
analogue of `T10-mvcc` ("what did the world look like a moment ago?")
and `T5-long` ("recall a fact from far back") combined.

---

## 5. The strategy spectrum re-interpreted

The L0–L11 axis transfers almost verbatim; only the operational
content of each level changes.

| L  | DB version                | Multimodal version                                    |
| -- | ------------------------- | ----------------------------------------------------- |
| 0  | parametric                | behavior cloning to a fixed policy                    |
| 1  | full in-context           | feed all sensor frames into the context window        |
| 2  | recurrent / SSM           | recurrent world model + recurrent controller          |
| 3  | KV eviction               | observation downsampling / latent codec               |
| 4  | self-summary              | learned scene digest, captioning into language buffer |
| 5  | hierarchical compression  | multi-rate temporal abstraction (options, skills)     |
| 6  | text scratchpad           | language-mediated perception ("I see a red box")      |
| 7  | KV tool                   | external replay buffer / episodic memory store        |
| 8  | typed tool                | scene graph / semantic memory                         |
| 9  | SQL tool                  | call a planner / MPC controller / solver              |
| 10 | planner / executor split  | hierarchical RL with sub-controllers                  |
| 11 | pure DB oracle            | oracle controller from privileged state               |

The Pareto frontier across these has direct analogues in current
robotics and embodied-AI debates:

- L2 vs. L3 — recurrent world model vs. downsampled buffer.
- L6 vs. L7 — language-mediated perception vs. raw episodic replay.
- L9 vs. L10 — call an external planner vs. internalize the controller.

---

## 6. Proposed new task families

### Perception (`Tp_*`)

| ID       | Probes                                                       |
| -------- | ------------------------------------------------------------ |
| `Tp-rec` | reconstruct a held-out frame from compressed memory          |
| `Tp-vqa` | answer a question about a clip seen long ago (vision T5)     |
| `Tp-count` | count distinct objects across a long video (perception T7)|
| `Tp-perm` | object permanence under occlusion (perception T10)           |
| `Tp-evt` | detect and timestamp events in a long audio stream           |
| `Tp-mod` | cross-modal binding (which speaker said what)                |

### Control (`Tc_*`)

| ID         | Probes                                                  |
| ---------- | ------------------------------------------------------- |
| `Tc-balance` | continuous control with tight stability margin        |
| `Tc-track`   | tracking a moving target within tolerance ε           |
| `Tc-long`    | long-horizon manipulation requiring remembered scenes |
| `Tc-async`   | multi-rate fusion (slow vision, fast proprio)         |
| `Tc-safe`    | tolerance-banded reward + cliff on safety violation   |
| `Tc-mpc`     | controller called as a tool from a slow planner       |

Each task ships with: an episode generator (sim seed + hparams),
a tolerance band $\varepsilon$, a cliff condition, and a per-task
adopted perceptual / reward metric.

---

## 7. Overlaps — what carries over

| DB-bench piece               | Carries to multimodal?                                  |
| ---------------------------- | ------------------------------------------------------- |
| Op-stream grammar            | yes; becomes an action / sensor grammar                 |
| Oracle (SQLite)              | yes; becomes a simulator (MuJoCo / IsaacSim / codec ref)|
| JSON wire format             | partially; needs tensor extension (NPZ / safetensors)   |
| Normalized bits recovered    | yes, modulo $\sim_\varepsilon$                          |
| Strategy spectrum L0–L11     | yes, level-by-level reinterpretation                    |
| Idle ticks $\rho$            | yes, generalized to per-stream rates                    |
| State probe                  | yes; linear-probe hidden state → scene attributes       |
| Capacity curve               | yes; bits-recovered vs. recurrent state size            |
| Cell-F1                      | no; cells aren't the right unit                         |
| Tree edit distance           | no; replaced by perceptual / scene-graph distance       |
| Schema induction (T3)        | partial; becomes "discover the scene grammar"           |

The unchanged-in-spirit metrics — bits recovered, capacity curve,
spectrum slicing — are the ones that motivate doing the extension.

---

## 8. Gaps — what needs new work

1. **Tolerance / cliff design.** Each task family needs an
   empirically validated $(\varepsilon, \varepsilon_{\max})$ pair.
   Picking these wrong creates either an unwinnable benchmark or a
   trivially-saturated one. Recommended approach: anchor $\varepsilon$
   to a published just-noticeable-difference (JND) and
   $\varepsilon_{\max}$ to a published failure threshold (e.g., MOS
   < 2.5, control termination).
2. **Simulator standardization.** The "SQLite of multimodal" doesn't
   exist. MuJoCo-MJX, IsaacSim, Habitat, and codec reference
   implementations all have different licensing, determinism, and
   speed properties. The bench needs to pin specific versions and
   ship a Docker image; otherwise reproducibility is gone.
3. **Tensor-aware wire format.** JSON is fine for ops; perception
   payloads (frames, audio chunks) need a binary wire. Candidates:
   safetensors per frame, or a streaming protobuf. The grammar layer
   stays JSON; only the payload changes.
4. **Stochasticity in scoring.** Sim is non-deterministic;
   ground-truth answers must be defined as a *distribution*, and
   bits-recovered estimation needs Monte-Carlo over sim seeds. This
   requires care: many seeds, narrow confidence intervals.
5. **Action grammar standardization.** Unlike SQL, continuous control
   has no shared schema. Need an explicit action space description
   per task (joint torques, end-effector deltas, discrete macros).
6. **Multi-rate harness.** No current benchmark harness understands
   "stream A at 30 Hz, stream B at 1 kHz, control out at 100 Hz."
   The training framework (`dbemu-train`'s gym API) would need a
   *multi-clock* extension; existing gym/gymnasium assumes one tick.
7. **Lossy-by-design strategies.** L3–L5 of the spectrum become much
   more interesting: codec-style learned compression is a research
   area of its own (Lyra, Encodec, SoundStream, neural video codecs).
   The bench must score them by *both* downstream task accuracy and
   reconstruction quality, jointly.

---

## 9. Concrete extension: `dbemu-perc`

A fifth benchmark layered on the existing four. Same shape as
`dbemu-static`: a frozen taxonomy plus a procedural env.

- **Subtasks.** Five: one each from `Tp-vqa`, `Tp-perm`,
  `Tp-count` (perception side); `Tc-balance`, `Tc-long`
  (control side).
- **Splits.** 1 000 train / 100 dev / 100 test per subtask, fixed
  seeds.
- **Modalities.** Vision @ 30 Hz, audio @ 16 kHz, proprio @ 100 Hz,
  action @ 100 Hz. Subset per task.
- **Metric.** Tolerance-banded normalized bits recovered, plus a
  per-task perceptual / reward metric reported alongside.
- **Two leaderboards** (mirroring `dbemu-static`):
  - *Limited context.* Must compress sensor streams to fit.
  - *Unlimited / recurrent.* Reported per recurrent state bit.
- **Strategy spectrum disclosure.** Same as DB version. Codec-based
  strategies (L3–L5) become first-class submissions.
- **Continuous-agent regime.** Per-stream $\rho_*$ axes.
- **Procedural env (`dbemu-perc-env`).** Hparam grid: sim difficulty,
  occlusion frequency, sensor noise, episode length, idle-tick rates
  per stream. Same fractional-factorial discipline as `dbemu-env`.

### What `dbemu-perc` is *not*

- Not a SOTA-driving robotics benchmark (LIBERO, Calvin, ManiSkill
  already cover that with much better simulator support).
- Not a perception benchmark (ImageNet, COCO, LAION already cover
  that).
- It is a *capability* benchmark for **continuous, lossy, partially
  correct state tracking under threshold-of-failure semantics**,
  using perception and control as the substrate. The
  contribution is the *scoring framework and spectrum mapping*, not
  another dataset.

---

## 10. Open research questions the extension exposes

1. **Where is the L2 / L3 frontier for video?** Is a recurrent world
   model ever competitive with a learned codec + transformer over
   compressed tokens per bit of state?
2. **Does $\Delta_\rho$ (idle-compute gain) survive moving to
   perception?** Recurrent world models claim to consolidate during
   sleep; this is testable with the same axis.
3. **Cliff topology.** What does the tolerance-band → cliff
   transition look like as $\varepsilon_{\max}$ shrinks? Smooth,
   sharp, multi-modal? Probably modality-dependent and informative
   in itself.
4. **Cross-modal binding under compression.** When the vision codec
   and audio codec are independent, does the system still bind
   speaker-to-mouth? L4 (self-summary) and L8 (typed tool / scene
   graph) should differ sharply here.
5. **Planner-vs-policy at L9 / L10.** When does calling an external
   solver win over an internalized controller, in tolerance-banded
   regimes? The DB version of this question (when does SQL stop
   helping?) has a known answer; the control version does not.

---

## 11. Summary

`dbemu-bench` is a benchmark of long-horizon, structured state
tracking under exact semantics. Multimodal perception and control
share **all** of the structural properties — long horizon, structured
state, strategy spectrum from internalized to delegated — but replace
exact semantics with **tolerance bands and failure cliffs**. The only
substantive new primitive is the equivalence-relation-modulated
bits-recovered scalar. Everything else — the spectrum, the
continuous-agent axis, the capacity curve, the state probe — carries
over.

The right next deliverable is `dbemu-perc`: a small, focused fifth
benchmark that establishes the tolerance-banded scoring framework on
two or three modalities, leaving SOTA chasing to existing per-domain
benchmarks. If the scoring framework holds up, the larger ambition is
clear: one Pareto plot, one spectrum, one scalar — across DB
emulation, perception, and control alike.
