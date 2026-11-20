# Section 5: Proposed Approach
> **Primary Owners**: Member 2 & Member 3  
> **Source File**: `approach.tex`

### Section Goals (Prompt Guidelines)
- **Input/Output**: Detail the continuous-to-discrete 5-tuple state transformation and the 5-element action space.
- **Methodology**: Present the core tabular Q-learning algorithm, Bellman TD update, tie-breaking logic, and $\epsilon$-greedy schedule.
- **Reward Shaping**: Formulate the modular reward functions, emphasizing the asymmetric progress shaping that prevents dithering.
- **System Flow**: Document the end-to-end RL training lifecycle.

### Writing Rules
- Display mathematical formulations cleanly in numbered equation environments.
- Maintain consistent variable notation ($\alpha = 0.1, \gamma = 0.99, |\mathcal{S}| = 144$).
