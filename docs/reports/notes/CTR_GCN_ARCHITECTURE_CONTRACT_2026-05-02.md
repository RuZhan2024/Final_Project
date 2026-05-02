# CTR-GCN Architecture Contract

Date: 2026-05-02  
Branch: `feat/ctr-gcn-upgrade`

## 1. Public Description

This project will use the name:

**`project-adapted single-stream CTR-GCN`**

This label is intentionally narrower than “full CTR-GCN reproduction”.

The model is based on the CTR-GCN idea of:

- a shared topology prior
- topology refinement
- channel-aware graph reasoning
- temporal modelling after graph propagation

However, the first-pass implementation is adapted to the current fall-detection project contract rather than imported from the original multi-modal action-recognition training stack.

## 2. Input / Output Contract

### External contract

- input tensor shape: `[B, T, V, F]`
- `B`: batch
- `T`: window length
- `V`: number of joints
- `F`: per-joint feature channels derived from `feat_cfg`

### Internal contract

- internal model layout: `[B, F, T, V]`
- no multi-person `M` dimension in the first pass
- no separate bone/motion stream in the first pass

### Output contract

- output: one fall logit per sample
- shape: `[B]` or `[B, 1]`, normalised by the shared `logits_1d()` helper

## 3. Graph / Topology Contract

### Base graph prior

The model starts from the existing MediaPipe 33-joint physical skeleton graph already used by the project.

The base adjacency is:

- undirected
- self-loop augmented
- symmetrically normalised

This base graph acts as the shared topology prior.

### Topology refinement design

The project version uses a **hybrid refinement mechanism**:

1. **shared dynamic relation term**
   - input-conditioned
   - derived from `theta/phi` projections
   - produces one batch-specific relation matrix `[B, V, V]`

2. **channel-wise low-rank refinement term**
   - learned
   - separate for each output channel
   - represented through low-rank factors
   - produces one channel-specific topology matrix `[C, V, V]`

The graph output is therefore:

- fixed graph prior contribution
- plus shared dynamic relation contribution
- plus channel-wise refinement contribution

This is the project’s concrete interpretation of “channel-wise topology refinement”.

## 4. Block Contract

Each block contains:

1. graph propagation with:
   - base adjacency prior
   - shared dynamic refinement
   - channel-wise refinement
2. temporal convolution
3. residual connection
4. batch normalisation
5. ReLU activation
6. dropout

### Residual rule

- identity when `in_ch == out_ch`
- `1x1` projection + normalisation when channel dimensions change

## 5. First-Pass Architecture Values

The first controlled baseline will use:

- `channels = (64, 64, 64, 128)`
- `rel_channels = 8`
- `ctr_rank = 8`
- `temporal_kernel = 9`
- `dropout = 0.30`

These are baseline values, not final tuned values.

## 6. Classifier Head

After the final block:

- global mean pooling over time and joints
- linear classifier head
- output dimension `1`

## 7. Checkpoint Contract

The saved bundle must include:

- `arch = "ctr_gcn"`
- `model_cfg`
- `feat_cfg`
- `data_cfg`
- `state_dict`

`model_cfg` must explicitly contain enough information to rebuild the model:

- `num_joints`
- `in_feats`
- `channels`
- `rel_channels`
- `ctr_rank`
- `temporal_kernel`
- `dropout`

No rebuild should depend on hidden training-script defaults.

## 8. Distinction from Existing Custom GCN

This line must remain clearly separate from the current `custom GCN`.

The key distinction is:

- current `custom GCN`: existing project graph baseline
- `CTR-GCN`: project-adapted graph model with explicit topology-refinement mechanism

This separation must remain visible in:

- code path
- checkpoint `arch`
- experiment naming
- report/paper comparison tables

## 9. Out of Scope for Task A

Task A does **not** claim:

- faithful reproduction of the full official CTR-GCN training stack
- multi-stream joint/bone/motion implementation
- multi-person modelling
- final tuned hyperparameters
- superior performance

Task A only fixes the architecture contract.

## 10. Task-A Completion Condition

Task A is complete when:

1. the code and documentation agree on the same `CTR-GCN` definition
2. the topology-refinement mechanism is explicit
3. the tensor layout is fixed
4. the checkpoint contract is fixed
5. the model is clearly distinct from the current custom GCN
