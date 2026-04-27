Date: 2026-04-27  
Owner: report/paper finalization branch  
Purpose: define the highest-yield retraining plan for genuinely strengthening the `CAUCAFall + TCN` line rather than only polishing wording.

# CAUCAFall TCN Offline Strengthening Plan

## 1. Objective

The goal is not to produce another generic TCN run.

The goal is to improve the strongest defended line of the project by reducing the **missed-fall tendency** that now appears on the bounded runtime surfaces, while preserving enough selectivity that the refitted `OP-2` line remains defendable.

This means the success target is:

1. stronger offline recall-side robustness on `CAUCAFall`
2. a better post-fit `OP-2` trade-off
3. fewer false negatives on the bounded custom replay surface
4. no catastrophic explosion of ADL false alarms

## 2. Why This Is the Right Place to Invest

The project already has many artifacts, but the highest-value publication-facing gap is concentrated in one place:

1. the `CAUCAFall + TCN` line is the strongest overall project line
2. the runtime weakness is not random; it is concentrated in missed-fall behaviour under the preferred bounded runtime preset
3. replacing the model family would add novelty risk and delay
4. broad experiment sprawl would not strengthen the defended narrative as efficiently as making this line stronger

Therefore the correct investment is:

`keep the architecture family, keep the primary dataset, and retrain toward recall-side robustness with deployment-aware screening.`

## 3. Baseline to Beat

Baseline training family:

1. `outputs/caucafall_tcn_W48S12_r2_train_hneg`
2. active runtime profile derived from `ops/configs/ops/tcn_caucafall.yaml`

Baseline evidence to compare against:

1. offline frozen metrics:
   - `outputs/metrics/tcn_caucafall_stb_s*.json`
2. active runtime profile:
   - `ops/configs/ops/tcn_caucafall.yaml`
3. bounded replay surfaces:
   - `artifacts/ops_delivery_verify_20260315/online_replay_summary.json`
   - `artifacts/fall_test_eval_20260315/summary_tcn_caucafall_locked_op2.csv`
   - `docs/reports/runbooks/FOUR_VIDEO_DELIVERY_PROFILE.md`

## 4. What Counts as Success

A new candidate should only be considered genuinely stronger if it satisfies all of the following:

1. offline `Recall` or `F1` on the defended `CAUCAFall` line improves or stays materially competitive
2. the post-fit `OP-2` line does not lose its bounded selectivity completely
3. bounded custom replay false negatives go down meaningfully
4. ADL false alarms do not explode to the point that the runtime line becomes less defendable than the current one

Concrete decision rule:

1. **promote** a candidate only if it improves offline recall-side evidence and produces a more defensible `OP-2` replay trade-off
2. **reject** a candidate if it improves recall only by making ADL control clearly worse than the current bounded line

## 5. Highest-Yield Experiment Slate

This slate is intentionally small. It is not a broad sweep. It is a set of four high-value candidates that target the exact weakness currently limiting the project.

### Candidate A. Hard-Neg Union Recall Push

Purpose:

1. keep the best existing training family as the starting point
2. strengthen boundary handling using the targeted hard-negative union file
3. preserve the current deployment-oriented feature contract

Rationale:

1. this is the lowest-risk true retraining candidate
2. it builds directly on the current strongest family
3. it is the most likely to improve the existing line without destabilising calibration completely

Command:

```bash
make train-tcn-caucafall \
  TCN_RESUME="outputs/caucafall_tcn_W48S12_r2_train_hneg/best.pt" \
  TCN_HARD_NEG_LIST="outputs/hardneg/tcn_caucafall_targeted_train_union.txt" \
  TCN_HARD_NEG_MULT="2" \
  TCN_LOSS="bce" \
  TCN_POS_WEIGHT="auto" \
  TCN_DROPOUT="0.40" \
  TCN_MASK_JOINT_P="0.12" \
  TCN_MASK_FRAME_P="0.08" \
  TCN_MONITOR="ap" \
  OUT_TAG="_rtA_hneg_union_recall"
```

Expectation:

1. modest offline improvement
2. best chance of improving bounded runtime without destroying selectivity

Execution update on 2026-04-27:

1. training completed successfully at:
   - `outputs/caucafall_tcn_W48S12_rtA_hneg_union_recall`
2. post-fit ops written to:
   - `configs/ops/tcn_caucafall_rtA_hneg_union_recall.yaml`
3. offline metrics written to:
   - `outputs/metrics/tcn_caucafall_rtA_hneg_union_recall.json`
4. bounded canonical four-folder replay improved from:
   - baseline `13/24` (`TP=3`, `TN=10`, `FP=2`, `FN=9`)
   - to Candidate A `16/24` (`TP=6`, `TN=10`, `FP=2`, `FN=6`)
5. important caution:
   - `fit_ops` required a confirm-disabled fallback after a degenerate sweep under confirmation
   - this means Candidate A is promising, but not yet an unqualified promotion

### Candidate B. Stronger Recall Bias with Balanced Sampling

Purpose:

1. attack the missed-fall tendency more aggressively
2. increase exposure to positive windows during training
3. test whether the runtime miss problem is partly caused by insufficient recall bias upstream

Rationale:

1. if the current line is too conservative, balanced sampling may recover more fall-supporting separation
2. this is a more aggressive candidate than A and may hurt calibration, so it must be screened carefully

Command:

```bash
make train-tcn-caucafall \
  TCN_RESUME="outputs/caucafall_tcn_W48S12_r2_train_hneg/best.pt" \
  TCN_HARD_NEG_LIST="outputs/hardneg/tcn_caucafall_targeted_train_union.txt" \
  TCN_HARD_NEG_MULT="1" \
  TCN_LOSS="bce" \
  TCN_POS_WEIGHT="none" \
  TCN_BALANCED_SAMPLER="1" \
  TCN_DROPOUT="0.38" \
  TCN_MASK_JOINT_P="0.10" \
  TCN_MASK_FRAME_P="0.06" \
  TCN_MONITOR="ap" \
  OUT_TAG="_rtB_balanced_sampler"
```

Expectation:

1. better recall-side pressure than Candidate A
2. higher risk of replay/runtime false alarms
3. worth running because it directly tests the main failure hypothesis

### Candidate C. Focal Recall Variant

Purpose:

1. bias optimisation toward harder windows without turning on balanced sampling
2. test whether the miss-prone clips are effectively hard positives rather than simply underweighted positives

Rationale:

1. focal loss can help if the difficult fall windows are being washed out by easier negatives
2. this candidate is useful precisely because it stresses a different mechanism from Candidates A and B

Command:

```bash
make train-tcn-caucafall \
  TCN_RESUME="outputs/caucafall_tcn_W48S12_r2_train_hneg/best.pt" \
  TCN_HARD_NEG_LIST="outputs/hardneg/tcn_caucafall_targeted_train_union.txt" \
  TCN_HARD_NEG_MULT="2" \
  TCN_LOSS="focal" \
  TCN_FOCAL_ALPHA="0.35" \
  TCN_FOCAL_GAMMA="1.5" \
  TCN_DROPOUT="0.38" \
  TCN_MASK_JOINT_P="0.10" \
  TCN_MASK_FRAME_P="0.06" \
  TCN_MONITOR="ap" \
  OUT_TAG="_rtC_focal_recall"
```

Expectation:

1. possible improvement on borderline fall windows
2. calibration risk higher than Candidate A
3. should only be promoted if post-fit runtime behaviour remains interpretable

### Candidate D. Extended Hard-Neg Plus Continuation

Purpose:

1. test whether the already stronger-looking extended family `r2_train_hneg_plus` can be pushed further
2. exploit the existing trajectory instead of only resuming the base `r2_train_hneg`

Rationale:

1. the repository already preserves `outputs/caucafall_tcn_W48S12_r2_train_hneg_plus`
2. this makes D a high-yield exploitation candidate rather than a fresh exploration candidate

Command:

```bash
make train-tcn-caucafall \
  TCN_RESUME="outputs/caucafall_tcn_W48S12_r2_train_hneg_plus/best.pt" \
  TCN_HARD_NEG_LIST="outputs/hardneg/tcn_caucafall_targeted_train_union.txt" \
  TCN_HARD_NEG_MULT="2" \
  TCN_LOSS="bce" \
  TCN_POS_WEIGHT="auto" \
  TCN_DROPOUT="0.40" \
  TCN_MASK_JOINT_P="0.12" \
  TCN_MASK_FRAME_P="0.08" \
  TCN_MONITOR="ap" \
  OUT_TAG="_rtD_hneg_plus_continue"
```

Expectation:

1. strongest chance of preserving existing calibration behaviour
2. useful if the best line is already close and only needs a better continuation pass

Execution update on 2026-04-27:

1. training completed successfully at:
   - `outputs/caucafall_tcn_W48S12_rtD_hneg_plus_continue`
2. post-fit ops written to:
   - `configs/ops/tcn_caucafall_rtD_hneg_plus_continue.yaml`
3. offline metrics written to:
   - `outputs/metrics/tcn_caucafall_rtD_hneg_plus_continue.json`
4. bounded runtime results:
   - canonical four-folder replay: `16/24`
   - locked 24-clip replay surface: `16/24`
5. important caution:
   - `fit_ops` again required confirm-disabled fallback after a degenerate sweep
6. comparison status:
   - Candidate D improves over baseline
   - Candidate D does not currently outperform Candidate A on the bounded runtime surfaces

## 6. Mandatory Post-Training Evaluation

Every candidate must go through the same three-stage screening process. No candidate should be discussed from training output alone.

### Stage 1. Offline Metrics Check

For each candidate:

1. collect the best checkpoint path
2. evaluate the candidate on the defended `CAUCAFall` metric surface
3. compare against the current defended TCN line on:
   - `AP`
   - `F1`
   - `Recall`
   - `FA24h`

Minimum rule:

1. discard candidates that clearly worsen offline recall-side quality

### Stage 2. Refit Operating Points

Each surviving candidate must get a new ops fit. Do not reuse the current active ops file.

Template command:

```bash
python3 ops/scripts/fit_ops.py \
  --arch tcn \
  --val_dir data/processed/caucafall/windows_eval_W48_S12/val \
  --ckpt <CANDIDATE_BEST_PT> \
  --out <CANDIDATE_OPS_YAML> \
  --fps_default 23 \
  --center pelvis \
  --use_motion 1 \
  --use_conf_channel 1 \
  --use_bone 1 \
  --use_bone_length 1 \
  --ema_alpha 0.20 \
  --k 2 \
  --n 3 \
  --cooldown_s 30 \
  --tau_low_ratio 0.78 \
  --confirm 0 \
  --confirm_s 2.0 \
  --confirm_min_lying 0.65 \
  --confirm_max_motion 0.08 \
  --confirm_require_low 1 \
  --thr_min 0.01 \
  --thr_max 0.95 \
  --thr_step 0.01 \
  --time_mode center \
  --merge_gap_s 1.0 \
  --overlap_slack_s 0.5 \
  --op1_recall 0.95 \
  --op3_fa24h 1.0 \
  --ops_picker conservative \
  --op_tie_break max_thr \
  --tie_eps 1e-3 \
  --save_sweep_json 1 \
  --allow_degenerate_sweep 0 \
  --emit_absolute_paths 0 \
  --min_tau_high 0.20
```

Minimum rule:

1. discard candidates whose fitted `OP-2` is obviously less usable than the current active bounded line

### Stage 3. Replay and Runtime Check

For each surviving candidate:

1. rerun the canonical replay matrix
2. rerun the bounded custom replay surface
3. compare especially:
   - false negatives on fall clips
   - false positives on ADL folders
   - whether the new `OP-2` remains interpretable as a balanced runtime preset

Priority comparison surface:

1. `corridor` and `kitchen` fall folders
2. `corridor_adl` and `kitchen_adl` non-fall folders

Minimum rule:

1. promote only candidates that reduce fall misses without making ADL false alarms clearly worse than the current bounded line

## 7. Promotion Logic

Do not promote by one metric alone.

Use the following ranking logic:

1. first priority: better bounded runtime trade-off under the new `OP-2`
2. second priority: stronger offline recall-side evidence
3. third priority: stable or acceptable false-alert burden
4. fourth priority: cleaner deployment narrative

This means:

1. a candidate that gains a little offline but destroys runtime selectivity should be rejected
2. a candidate that modestly improves offline and clearly improves runtime miss behaviour should be preferred

## 8. Expected Best Outcome

The realistic best-case outcome of this slate is not “solve deployment”.

The realistic best-case outcome is:

1. a stronger `CAUCAFall + TCN` offline line
2. a refitted `OP-2` that misses fewer fall clips
3. a better defended runtime story than the current `15/24` custom replay line
4. a stronger publication-facing argument that the project improved the system where its evidence was weakest

## 9. What Not to Do

Do not:

1. expand to new architectures before this slate finishes
2. run large blind sweeps over unrelated hyperparameters
3. promote a candidate only because it raises one replay number
4. let runtime tuning replace the need to evaluate offline and refit ops properly

## 10. Immediate Run Order

Run in this order:

1. Candidate A
2. Candidate D
3. Candidate B
4. Candidate C

Reason:

1. A and D have the best expected balance of gain and stability
2. B is the most direct recall-push stress test
3. C is the most calibration-risky and should be screened later

## 11. Final Decision Gate

At the end of the slate, answer one question only:

`Did any new CAUCAFall TCN candidate produce a meaningfully better offline-plus-runtime trade-off than the current defended line?`

If yes:

1. promote that candidate
2. regenerate the fitted ops profile
3. update the runtime evidence chain
4. revise report/paper language accordingly

If no:

1. stop retraining
2. keep the current line
3. present the existing runtime weakness honestly as a bounded limitation
