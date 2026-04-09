# Paper Headings and Subheadings

## Title
Pose-Based Fall Detection with Temporal and Graph Neural Models: Cross-Dataset Evaluation and Deployment Validation

## Abstract
### Background
### Aim
### Methods
### Results
### Conclusion

## Keywords

## 1. Introduction
### 1.1 Background and Research Context
### 1.2 Problem Statement and Motivation
### 1.3 Research Gap
### 1.4 Research Aim and Objectives
### 1.5 Research Questions
### 1.6 Contributions of This Study
### 1.7 Paper Structure

## 2. Related Work
### 2.1 Vision-Based Fall Detection
### 2.2 Pose-Based Human Action Understanding
### 2.3 Temporal and Graph Neural Approaches for Skeleton Sequences
### 2.4 Fall Detection in Real-World and Deployment Settings
### 2.5 Summary of the Research Gap

## 3. Methods
### 3.1 Research Design and System Overview
### 3.2 Fall Detection Task Formulation
### 3.3 Datasets
#### 3.3.1 CAUCAFall

#### 3.3.2 LE2i
#### 3.3.3 Replay and Field Validation Data
### 3.4 Labels, Event Definitions, and Data Splits
### 3.5 Pose Preprocessing Pipeline
#### 3.5.1 2D Pose Extraction 
#### 3.5.2 Skeleton Normalization and Resampling
#### 3.5.3 Feature Construction
#### 3.5.4 Temporal Window Generation
### 3.6 Fall Detection Models
#### 3.6.1 Temporal Convolutional Network
#### 3.6.2 Graph Convolutional Network
#### 3.6.3 Loss Function and Training Objective
#### 3.6.4 Regularization and Stability Measures
### 3.7 Operating-Point Fitting and Alert Policy
#### 3.7.1 Validation-Based Threshold Selection
#### 3.7.2 Recall, Balanced, and False-Alert-Control Profiles
#### 3.7.3 Smoothing, Confirmation, and Cooldown Logic
### 3.8 Experimental Setup
#### 3.8.1 Implementation Environment
#### 3.8.2 Training Configuration
#### 3.8.3 Evaluation Metrics 
#### 3.8.4 Objective-to-Metric Mapping
#### 3.8.5 Reproducibility and Artifact Tracking
### 3.9 Validity, Reliability, and Ethical Considerations

## 4. Results
### 4.1 In-Domain Benchmark Results on CAUCAFall
### 4.2 In-Domain Benchmark Results on LE2i
### 4.3 Comparison of TCN and GCN Performance
### 4.4 Precision, Recall, and False-Alert Trade-Offs
### 4.5 Cross-Dataset Generalization Results
#### 4.5.1 LE2i to CAUCAFall Transfer
#### 4.5.2 CAUCAFall to LE2i Transfer
#### 4.5.3 In-Domain Versus Cross-Domain Performance Drop
### 4.6 Multi-Seed Stability and Statistical Results
#### 4.6.1 Confidence Intervals for Key Metrics
#### 4.6.2 Significance Testing of Final Candidates
### 4.7 Deployment Validation Results
#### 4.7.1 Replay-Based Validation
#### 4.7.2 Field Validation Outcomes
#### 4.7.3 Detection Delay and False-Alert Behaviour

## 5. Discussion
### 5.1 Interpretation of the Main Findings
### 5.2 Comparison with Prior Literature
### 5.3 Cross-Dataset and Deployment Implications
### 5.4 Unexpected Findings and Error Patterns
### 5.5 Practical Trade-Offs for Real-World Alerting
### 5.6 Limitations and Threats to Validity

## 6. Conclusion
### 6.1 Summary of the Study
### 6.2 Contributions of the Proposed System
### 6.3 Recommendations for Future Work
### 6.4 Final Answers to the Research Questions

## References

## Appendix A. Reproducibility and Implementation Details
### A.1 Configuration Files and Runtime Profiles
### A.2 Training and Evaluation Commands
### A.3 Additional Implementation Notes
### A.4 Objective and Research Question Traceability

## Appendix B. Additional Tables and Figures
### B.1 Extended Metric Tables
### B.2 Stability and Transfer Plots
### B.3 Deployment Case Studies
