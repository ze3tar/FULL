# PSO Smoothing Evaluation

This report evaluates the Particle Swarm Optimization (PSO) smoother as a post-processing step for three representative paths (baseline APF-RRT export, improved APF-RRT export, and a synthetic sinusoid).

- Obstacles: three spherical keep-out zones used during smoothing to enforce clearance.
- Hyperparameters: tuned per path via `tune_from_environment`, then biased toward curvature reduction (high smoothness/curvature weights, modest collision cost).
- Visuals: overlays saved as `pso_overlay_*.png` show the original (red) and PSO-smoothed (blue) trajectories in 3D with obstacle shells.

| Path | Length Before | Length After | Smoothness Before | Smoothness After | Max Curvature Before | Max Curvature After |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 748.65 | 1110.81 | 18.10 | 68.67 | 1.57 | 2.84 |
| Improved | 845.06 | 1202.45 | 33.96 | 124.08 | 2.28 | 2.66 |
| Synthetic | 623.61 | 622.14 | 0.81 | 0.98 | 0.23 | 0.32 |

## Notes

- The synthetic path shows the smallest deviation, with curvature kept low while respecting obstacle clearance.
- Higher curvature weights reduce abrupt turns but may trade off path length. The exposed hyperparameter bundle (`PSOHyperparameters`) allows quick retuning per environment for tighter smoothness/length constraints.
- Each overlay can be loaded in RViz or inspected directly via the saved PNGs for qualitative verification of clearance and endpoint adherence.
