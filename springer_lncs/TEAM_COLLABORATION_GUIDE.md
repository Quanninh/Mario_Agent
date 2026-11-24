# 👥 Team Collaboration & Modular Report Guide
> **Course**: Machine Learning — Project 5: Reinforcement Learning Game Agent (*Hoá's Midnight Escape*)  
> **Template**: Springer Nature Lecture Notes in Computer Science (LNCS)  
> **Target Document**: `main.tex` / `samplepaper.tex`

---

## 1. Folder Structure & Ownership Matrix

To prevent git merge conflicts and ensure everyone can edit in parallel without overwriting each other, the report has been divided into modular folders under `springer_lncs/sections/`. 

```text
springer_lncs/
├── main.tex                                    # Root LaTeX compiler file (uses \input{})
├── samplepaper.tex                             # Identical mirror of main.tex
├── references.bib                              # Shared BibTeX references
├── llncs.cls                                   # Springer LNCS document class
├── splncs04.bst                                # Springer bibliography style
├── figures/                                    # Academic evaluation plots (PDF & PNG)
│   ├── graph_1_curriculum_learning_curve.pdf
│   ├── graph_2_reward_profiling_benchmark.pdf
│   └── graph_3_playstyle_matrix.pdf
└── sections/                                   # 📂 8 MODULAR SECTION FOLDERS
    ├── 00_abstract/
    │   ├── abstract.tex                        # Single paragraph (<= 200 words)
    │   └── README.md
    ├── 01_introduction/
    │   ├── introduction.tex                    # Background, context, motivation, outline
    │   └── README.md
    ├── 02_related_work/
    │   ├── related_work.tex                    # RL in games, discretization, curriculum
    │   └── README.md
    ├── 03_objectives_problem_gaps/
    │   ├── objectives.tex                      # MDP formulation, research gaps, goals
    │   └── README.md
    ├── 04_teamwork_project_management/
    │   ├── teamwork.tex                        # Task distribution, workflow, ideation
    │   └── README.md
    ├── 05_proposed_approach/
    │   ├── approach.tex                        # 144-state math, Q-learning, reward shaping
    │   └── README.md
    ├── 06_implementation_details/
    │   ├── implementation.tex                  # Architecture, Pygame GUI, tech stack, code
    │   └── README.md
    ├── 07_experimental_results_evaluation/     # ⭐️ OWNED BY MEMBER 4 (USER)
    │   ├── evaluation.tex                      # Full evaluation, Tables 1 & 2, Figs 1-3, t-tests
    │   └── README.md
    └── 08_conclusion_future_work/
        ├── conclusion.tex                      # Reflections, takeaways, future research
        └── README.md
```

---

## 2. Team Member Assignments

| Section | Folder Path | Primary Owners | Description & Scope |
| :--- | :--- | :--- | :--- |
| **Abstract** | `sections/00_abstract/` | **All Members** (Lead: Member 4) | High-level synthesis under 200 words. |
| **1. Introduction** | `sections/01_introduction/` | **Member 1 & Member 4** | Game context (*Hoá's Midnight Escape*), RL motivation, and paper outline. |
| **2. Related Work** | `sections/02_related_work/` | **Member 2 & Member 3** | Literature review on Q-learning, deep RL vs tabular, curriculum, and reward shaping. |
| **3. Objectives & Gaps** | `sections/03_objectives_problem_gaps/` | **Member 2 & Member 3** | MDP formulation, curse of dimensionality, sparse vs dense limitations, goals. |
| **4. Teamwork & Mgmt** | `sections/04_teamwork_project_management/` | **All Members** | Roles breakdown, GitHub branching, SOLID design, concept ideation. |
| **5. Proposed Approach** | `sections/05_proposed_approach/` | **Member 2 & Member 3** | 144-state 5-tuple, Q-table update math, tie-breaking, asymmetric reward shaping. |
| **6. Implementation** | `sections/06_implementation_details/` | **Member 1, 2, & 3** | Pygame architecture, headless vs GUI modes, tech stack, key code snippets. |
| **7. Evaluation** | `sections/07_experimental_results_evaluation/` | **Member 4 (You)** | Full curriculum benchmarks, Table 1 & 2, Figures 1-3, Welch's t-tests ($p < 0.001$). |
| **8. Conclusion** | `sections/08_conclusion_future_work/` | **All Members** | Summary of contributions, team takeaways, and future work (deep RL, lookahead). |

---

## 3. Rules to Avoid Git Conflicts

1. **Only edit your own `.tex` file**:  
   - Member 1 edits `sections/01_introduction/`, `sections/04_teamwork_project_management/`, `sections/06_implementation_details/`.
   - Member 2 edits `sections/02_related_work/`, `sections/03_objectives_problem_gaps/`, `sections/05_proposed_approach/`.
   - Member 3 edits `sections/05_proposed_approach/`, `sections/06_implementation_details/`.
   - Member 4 edits `sections/07_experimental_results_evaluation/` (and coordinates `00_abstract/` and `08_conclusion/`).
2. **Do not modify `main.tex` directly**:  
   - The master document simply contains `\input{...}` calls. Modifying only your section file means git pulls and merges will resolve with zero conflicts.
3. **Adding References**:  
   - Add new `@article` or `@inproceedings` entries to the end of `references.bib`. Do not re-order existing bib entries.

---

## 4. How to Compile / Upload to Overleaf


- **On Local LaTeX (TeX Live / MacTeX / MiKTeX)**:
  ```bash
  cd springer_lncs
  pdflatex main.tex
  bibtex main
  pdflatex main.tex
  pdflatex main.tex
  ```
