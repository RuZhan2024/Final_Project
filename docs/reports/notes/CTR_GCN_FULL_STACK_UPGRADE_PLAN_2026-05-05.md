# CTR-GCN Full-Stack Upgrade Plan

Date: 2026-05-05

Purpose: define the next CTR-GCN work needed to challenge or beat the defended
TCN line across offline metrics, operating-point fitting, and bounded custom
replay evidence.

## Current Position

Five-seed offline OP2 comparison:

| Model | AP mean | AUC mean | OP2 F1 mean | OP2 recall mean | OP2 FA/24h |
| --- | ---: | ---: | ---: | ---: | ---: |
| TCN | `0.9819` | `0.9897` | `0.8611` | `0.7600` | `0.0000` |
| CTR-GCN | `0.9707` | `0.9800` | `0.7421` | `0.6000` | `0.0000` |

Bounded 24-clip custom replay:

| Line | TP | TN | FP | FN | Correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| Strengthened TCN Candidate A/D | `6` | `10` | `2` | `6` | `16/24` |
| CTR-GCN seed `33724876`, own OP2 | `7` | `10` | `2` | `5` | `17/24` |

Interpretation:

- CTR-GCN has a promising replay seed, but not yet a stable offline win.
- Directly reusing TCN Candidate A/D thresholds on CTR-GCN fails because model
  probability scales differ.
- To claim a real upgrade, CTR-GCN must improve five-seed offline AP/AUC and
  OP2 event metrics, then keep or improve the `17/24` replay result without
  raising ADL false positives.

## Guardrails

- Do not train on test windows.
- Do not use custom replay clips as training data for a defended test claim.
- Replay clips can be used only as a diagnostic or as a clearly labelled
  deployment-profile calibration surface.
- Preserve the TCN five-seed seed set for comparison:
  `1337`, `17`, `2025`, `33724876`, `42`.
- Any split/window/label changes must regenerate TCN and CTR-GCN under the same
  protocol before claiming model-family superiority.

## Upgrade Layers

### 1. Extraction and Pose Contract

Audit before changing:

- pose preprocessing version and MediaPipe/keypoint schema
- confidence masking and missing-joint interpolation behavior
- fps handling and `motion_scale_by_fps`
- whether replay windows and train/eval windows share the same feature contract

Potential CTR-GCN improvements:

- preserve confidence as a stronger input signal
- add controlled coordinate noise and quantization augmentation
- test whether `center=pelvis` remains best for graph models

### 2. Splits and Leakage

Audit before changing:

- train/val/test video IDs are disjoint
- hard-negative lists are train-only
- fall/ADL distributions per split
- subject/scene/view balance if metadata is available

Potential improvement:

- if split imbalance is large, define a new balanced split protocol and rerun
  TCN plus CTR-GCN together. This is a protocol reset, not a CTR-only tweak.

Current audit result:

- `windows_eval_W48_S12` has no video overlap between train, val, and test.
- split sizes are:
  - train: `1024` windows, `80` videos, labels `614` non-fall / `410` fall
  - val: `144` windows, `10` videos, labels `85` non-fall / `59` fall
  - test: `146` windows, `10` videos, labels `82` non-fall / `64` fall

### 3. Labels and Windows

Audit before changing:

- span overlap thresholds for positive windows
- negative sampling ratio
- near-boundary hard negatives
- window length/stride contract: current `W48/S12`

Potential CTR-GCN candidates:

- `W64/S16` or `W64/S12` for longer temporal context
- keep `W48/S12` as baseline so gains are attributable
- increase hard negatives near fall boundaries and ADL low-posture clips
- test softer positive-window inclusion only if both TCN and CTR-GCN are rerun

Current quality finding:

- train contains `40` windows with `valid_frac < 0.25`, including `17` fall
  windows and `20` completely invalid windows.
- test contains `19` windows with `valid_frac < 0.25`, including `12` fall
  windows and `16` completely invalid windows.
- These windows are especially important for CTR-GCN because graph topology
  reasoning is fragile when the skeleton is absent or heavily masked. Improving
  extraction, interpolation, or low-quality-window handling may matter more than
  another small learning-rate sweep.

### 4. Trainer Parity

CTR-GCN currently lacks several mature trainer features already available in
the TCN/GCN trainers:

- EMA evaluation/checkpointing
- LR schedulers: plateau/cosine/onecycle
- val loss tracking
- resume/hard-negative continuation support
- hard-negative leakage guard
- imbalance-strategy guard

This is the safest first engineering task because it gives CTR-GCN the same
optimization tools as the defended TCN line without changing the data protocol.

Status:

- implemented in `ml/src/fall_detection/training/train_ctr_gcn.py`
- smoke-tested with `--scheduler cosine --use_ema 1`
- existing CTR-GCN/model guard tests still pass

### 5. Architecture

Current CTR-GCN is project-adapted single-stream `[B,T,V,F]`.

Candidate upgrades:

- deeper/wider channels: e.g. `64,64,128,128,256`
- temporal kernels: `5`, `9`, `13`
- lower and higher CTR rank: `4`, `8`, `16`
- multi-stream CTR-GCN:
  - joint stream
  - motion stream
  - optional bone/bone-length stream
  - late fusion head

Priority should be trainer parity first, then multi-stream. Multi-stream changes
are more likely to move offline metrics meaningfully but also increase variance.

### 6. Loss and Sampling

Candidates:

- BCE with `pos_weight=auto`
- BCE with balanced sampler and `pos_weight=none`
- focal loss without balanced sampler
- hard-negative continuation from the best seed/checkpoint

Avoid double correction unless explicitly testing it as an ablation.

### 7. Fit and Operating Points

CTR-GCN needs its own fit policy:

- sweep `tau_high/tau_low/k/n/ema_alpha`
- keep `FA24h=0` where possible
- target OP2 recall first, then replay false-positive control
- explicitly record confirm fallback rather than hiding it

Replay-aware policy tuning should be labelled as deployment-profile calibration,
not as offline model improvement.

## Proposed Execution Order

1. Add CTR-GCN trainer parity features.
2. Run a small single-seed trainer-parity smoke and one serious seed.
3. If promising, run a bounded trainer-parity sweep on seed `33724876`.
4. Freeze the best non-replay-selected config.
5. Run five-seed stability.
6. Fit ops for each seed.
7. Run custom replay on:
   - CTR-GCN own OP2
   - replay diagnostic sweep under `FP<=2`
8. Compare against TCN:
   - five-seed offline AP/AUC/F1/recall/FA24h/delay
   - strengthened TCN `16/24` custom replay line

## Success Criteria

Minimum interesting result:

- CTR-GCN beats TCN on bounded custom replay without increasing false positives.

Stronger result:

- CTR-GCN beats TCN on five-seed OP2 F1 or recall while keeping `FA24h=0`.

Full upgrade claim:

- CTR-GCN beats or ties TCN on AP/AUC,
- beats TCN on OP2 F1/recall,
- keeps `FA24h=0`,
- matches or beats `16/24` replay across more than one seed,
- and all evidence is produced under leakage-safe train/val/test separation.
