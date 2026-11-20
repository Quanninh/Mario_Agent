# Section 7: Experimental Results & Evaluation
> **Primary Owner**: Member 4 (Evaluation Lead)  
> **Source File**: `evaluation.tex`  
> **Associated Artifact**: `section_7_experimental_results_and_evaluation.md`

### Section Goals (Prompt Guidelines)
- **Simulations & Scenarios**: Document test runs and parameter configurations across the 3 curriculum tiers (Level 1, Level 2, Level 3; 1,200 total episodes).
- **Data & Comparisons**: Use tables (Table 1: All vs Converged, Table 2: Reward Ablation) and vector figures (Figures 1, 2, 3) to compare algorithms.
- **Analysis**: Provide a thorough evaluation supported by statistical significance tests (Welch's two-sample $t$-tests with $t$-statistics, degrees of freedom, and $p$-values).
- **Behavioral Profiling**: Analyze playstyle clustering and failure mode distributions.
- **Limitations**: Explicitly delineate perceptual aliasing and tabular representation boundaries.

### Assets & Figures
- Figure 1: `figures/graph_1_curriculum_learning_curve.pdf`
- Figure 2: `figures/graph_2_reward_profiling_benchmark.pdf`
- Figure 3: `figures/graph_3_playstyle_matrix.pdf`
- Raw Telemetry: `evaluation/mario_training_log.csv`
