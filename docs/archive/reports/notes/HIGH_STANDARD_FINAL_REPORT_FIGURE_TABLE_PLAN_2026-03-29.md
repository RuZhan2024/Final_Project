# High-Standard Final Report Figure and Table Plan

Date: 2026-03-29  
Purpose: audit the current figure/table assets for the final report, identify which ones are directly usable, which ones must be redesigned, and define the final report-ready figure/table set.

## 1. Overall Judgment

The current figure assets are **not yet sufficient for a high-standard final report**.

The problem is not only quantity. It is mainly that the current figures were produced as internal analysis artifacts rather than as final report figures. As a result:

- several figures mix too many concepts into one panel
- axis labels are long, crowded, or ambiguous
- some figures do not serve a single clear claim
- visual hierarchy is weak
- some current plots are harder to read than the equivalent table
- deployment/runtime evidence is under-visualised
- there is no clean architecture figure yet
- there is no clean alert-policy figure yet

Conclusion:

- some raw assets are still useful as data sources
- most current figures should be treated as **draft analysis figures**, not final report figures
- the report needs a small, deliberate figure/table pack rather than a larger number of noisy plots

## 2. Audit of Current Figures

Current tracked figure files:

- [cross_dataset_transfer_bars.png](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/cross_dataset/cross_dataset_transfer_bars.png)
- [latency_profile_summary.png](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/latency/latency_profile_summary.png)
- [fc1_fc4_ap_comparison.png](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/pr_curves/fc1_fc4_ap_comparison.png)
- [fc_stability_boxplot.png](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/stability/fc_stability_boxplot.png)

### 2.1 Cross-Dataset Transfer Bars

File:
- [cross_dataset_transfer_bars.png](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/cross_dataset/cross_dataset_transfer_bars.png)

Current strengths:
- relevant to an important claim
- already tied to cross-dataset artifacts
- shows directional transfer behaviour

Current problems:
- labels are too long and visually heavy
- three panels are crammed into one figure without enough hierarchy
- the `FA24h` panel is hard to interpret because of the log-scaled y-axis and sparse signal
- the legend is generic rather than claim-oriented
- the figure does not immediately communicate the key message: **transfer is asymmetric and failure is especially severe in the `CAUCAFall -> LE2i` direction**

Decision:
- **Rework, do not use as-is**

Recommended redesign:
- simplify labels to directional short names
- focus on a smaller number of metrics
- probably show `F1` and `Recall` as the main figure
- move `FA24h` into a table or a companion panel only if needed

### 2.2 Latency Profile Summary

File:
- [latency_profile_summary.png](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/latency/latency_profile_summary.png)

Current strengths:
- latency is an important deployment topic
- the file is already available

Current problems:
- only one category is shown, so the bar chart is visually weak
- it does not compare local vs deployment, replay vs live, or before vs after optimisation
- the title is generic
- the figure is not yet aligned to the strongest runtime claim in the report

Decision:
- **Do not use in current form**

Recommended redesign:
- either replace with a compact runtime summary table
- or rebuild as a comparison figure with multiple clearly defined runtime scenarios

### 2.3 Candidate Quality Metrics

File:
- [fc1_fc4_ap_comparison.png](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/pr_curves/fc1_fc4_ap_comparison.png)

Current strengths:
- candidate-level summary exists
- useful as an internal orientation plot

Current problems:
- left panel mixes `AP`, `Precision`, `Recall`, and `F1` in one grouped bar chart, which is visually dense and not claim-focused
- right panel isolates `FA24h`, but the combination with the left panel makes the full figure harder to read
- labels such as `TCN-LE2I`, `GCN-LE2I`, `TCN-CAUCAFALL`, `GCN-CAUCAFALL` are readable, but the figure still feels like an internal dashboard rather than a publication-ready result
- it does not directly encode the main report logic:
  - `CAUCAFall` is the primary result-bearing dataset
  - `LE2i` is comparative evidence

Decision:
- **Use only as an internal source, not as a final report figure**

Recommended redesign:
- replace with a clean primary results table
- if a figure is still wanted, split the main result-bearing comparison from the comparative dataset

### 2.4 Stability Boxplot

File:
- [fc_stability_boxplot.png](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/stability/fc_stability_boxplot.png)

Current strengths:
- stability is a real part of the argument
- multi-seed evidence is important

Current problems:
- x-axis labels are effectively unreadable
- too many categories are plotted together
- the figure does not privilege the final frozen candidate comparison
- the current plot reads more like a debugging plot than a report figure

Decision:
- **Rework**

Recommended redesign:
- only show the final frozen candidates relevant to the report
- use short labels
- probably separate `CAUCAFall` and `LE2i`
- emphasise `F1` and `Recall`, not every available metric

## 3. Final Report Figure/Table Pack

The report should use a **small, coherent set** of visuals.

### 3.1 Required Figures

#### Figure 1. System Architecture

Status:
- **Now generated as SVG**

Purpose:
- explain frontend, backend, model, alert policy, persistence, and deployment path

Why needed:
- the report currently has no clean visual for the full system

Format:
- diagram, not a plot

#### Figure 2. Stability / Primary Offline Comparison Figure

Status:
- **Needs redesign from current stability artifacts**

Purpose:
- support the claim that TCN trends stronger than GCN under the frozen protocol while making variability visible

Why needed:
- this is the cleanest visual companion to the main offline result table

Likely content:
- `CAUCAFall` primary comparison
- possibly `F1` and `Recall` only

#### Figure 3. Cross-Dataset Transfer Figure

Status:
- **Needs redesign from current cross-dataset plot**

Purpose:
- support the claim that transfer is asymmetric and bounded

Why needed:
- this is one of the clearest limitation visuals in the project

Likely content:
- in-domain vs cross-domain deltas
- focus on `F1` and `Recall`

#### Figure 4. Alert Policy / Decision Flow

Status:
- **Now generated as SVG**

Purpose:
- explain how window scores become operational alerts

Why needed:
- calibration and operating points are central to the report narrative
- this is difficult to explain clearly with text alone

Format:
- diagram or flow figure, not a performance chart

### 3.2 Required Tables

#### Table 1. Frozen Candidate Protocol Summary

Purpose:
- list frozen candidates, datasets, and roles

Why:
- improves protocol clarity early in the methods/results boundary

#### Table 2. Main Offline Comparative Results

Purpose:
- present the key comparative metrics cleanly

Why:
- the current best quantitative evidence is tabular
- a table will be clearer than the current mixed candidate bar plot

Likely columns:
- dataset
- model
- AP mean
- F1 mean
- Recall mean
- Precision mean
- FA24h mean

#### Table 3. Significance / Stability Interpretation

Purpose:
- show the Wilcoxon-based caution explicitly

Why:
- avoids overstating TCN-vs-GCN claims

Likely columns:
- dataset
- metric
- mean difference
- Wilcoxon p
- interpretation

#### Table 4. Deployment and Runtime Summary

Purpose:
- consolidate replay lock, bounded custom replay evidence, limited field validation, and runtime caveats

Why:
- runtime evidence is important, but much of it is better summarised in a structured table than in a crowded chart

## 4. What Should Not Be Used As Final Figures

The following should **not** be used in their current form:

- [latency_profile_summary.png](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/latency/latency_profile_summary.png)
- [fc1_fc4_ap_comparison.png](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/pr_curves/fc1_fc4_ap_comparison.png)
- [fc_stability_boxplot.png](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/stability/fc_stability_boxplot.png) as currently rendered
- [cross_dataset_transfer_bars.png](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/cross_dataset/cross_dataset_transfer_bars.png) as currently rendered

Reason:
- they are analytically useful but not yet clear enough for the final report

## 5. Figure Design Rules for This Report

All new or redesigned report figures should follow these rules:

1. One figure should support one main claim.
2. Titles must say what the figure shows, not just the metric name.
3. Axis labels must use short, readable labels and explicit units where relevant.
4. Legends should be minimal and semantically meaningful.
5. Font sizes must remain readable when placed in a report PDF.
6. Avoid mixing too many metrics in one panel unless the comparison is genuinely simple.
7. Prefer a table over a figure when the message is mainly numerical rather than structural.
8. `CAUCAFall` should be visually privileged as the primary result-bearing dataset.
9. `LE2i` should be clearly marked as comparative/generalisation evidence.
10. Replay/deployment visuals must be labeled as bounded deployment evidence, not as formal unseen-test evidence.

## 6. Immediate Next Actions

The next figure/table work should proceed in this order:

1. Build a clean architecture diagram.
2. Build the main offline comparison table.
3. Redesign the stability figure around the frozen candidates only.
4. Redesign the cross-dataset figure around the asymmetry claim.
5. Build a deployment/runtime summary table.
6. Build an alert-policy flow figure if space permits.

## 7. Final Decision

Current visual assets are **not final-report-ready**.

They provide a usable data foundation, but the report should move forward with:

- **2 redesigned quantitative figures**
- **2 newly created explanatory figures**
- **3 to 4 clean tables**

rather than trying to reuse the current analysis figures unchanged.
