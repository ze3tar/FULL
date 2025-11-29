#!/usr/bin/env python3
"""Benchmark baseline APF-RRT against the RL-enhanced variant.

This script evaluates the policy without altering training parameters and saves both
summary and per-trial results for deeper analysis.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

try:
    from baseline_enhanced import (
        create_random_spheres,
        path_length,
        prune_path,
        rrt_apf_guided,
    )
    from rl_enhanced_apf_rrt import (
        ObservationNormalizer,
        PlannerParameters,
        ScenarioConfig,
        load_agent,
        plan_with_rl_for_benchmark,
    )
except ImportError as e:  # pragma: no cover - import guard
    print(f"❌ Import error: {e}")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials",
        type=int,
        default=60,
        help="Number of trials per scenario (default: 60)",
    )
    parser.add_argument(
        "--model-path",
        default=str(base_dir / "models/final_model.zip"),
        help="Path to PPO policy zip (default: models/final_model.zip relative to script)",
    )
    parser.add_argument(
        "--normalizer-path",
        default=str(base_dir / "models/obs_normalizer.npz"),
        help="Path to observation normalizer (default: models/obs_normalizer.npz relative to script)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/user-data/outputs"),
        help="Directory to store CSV outputs (default: /mnt/user-data/outputs)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions from the policy (default: stochastic)",
    )
    return parser.parse_args()


def load_model(model_path: str, normalizer_path: str) -> Tuple[PPO, ObservationNormalizer]:
    print("\n1. Loading RL model...")
    try:
        model, loaded_normalizer = load_agent(Path(model_path))
        normalizer: ObservationNormalizer
        if Path(normalizer_path).exists():
            normalizer = ObservationNormalizer.from_file(Path(normalizer_path))
        elif loaded_normalizer is not None:
            normalizer = loaded_normalizer
        else:
            raise FileNotFoundError(
                "Observation normalizer not found; ensure obs_normalizer.npz is available."
            )
        print(f" Model loaded: {model_path}")
        print(f" Normalizer loaded: {normalizer_path if Path(normalizer_path).exists() else loaded_normalizer}")
        return model, normalizer
    except Exception as exc:  # pragma: no cover - runtime guard
        print(f" Failed to load model: {exc}")
        sys.exit(1)


def calc_stats(stats_dict: Dict[str, List[float]]) -> Tuple[float, float, float, float]:
    success_flags = stats_dict["success"]
    success_rate = (np.mean(success_flags) * 100) if success_flags else 0.0
    if any(success_flags):
        avg_time = float(np.mean(stats_dict["time"])) if stats_dict["time"] else np.nan
        avg_nodes = float(np.mean(stats_dict["nodes"])) if stats_dict["nodes"] else np.nan
        avg_length = float(np.mean(stats_dict["length"])) if stats_dict["length"] else np.nan
    else:
        avg_time = avg_nodes = avg_length = np.nan
    return success_rate, avg_time, avg_nodes, avg_length


def fmt(value: float, precision: str = "{:.2f}") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return precision.format(value)


def record_trial(
    stats: Dict[str, List[float]],
    path: Iterable[Tuple[float, float, float]],
    nodes: List[Tuple[float, float, float]],
    elapsed: float,
    obstacles: np.ndarray,
    *,
    path_length_override: float | None = None,
    node_count: int | None = None,
) -> None:
    pruned = prune_path(path, obstacles)
    stats["time"].append(elapsed)
    stats["nodes"].append(node_count if node_count is not None else len(nodes))
    stats["length"].append(path_length_override if path_length_override is not None else path_length(pruned))
    stats["success"].append(1)


def run_planner(
    start: Tuple[float, float, float],
    goal: Tuple[float, float, float],
    obstacles: np.ndarray,
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    params: PlannerParameters,
) -> Tuple[np.ndarray, List[Tuple[float, float, float]], List[int], float]:
    return rrt_apf_guided(
        start,
        goal,
        obstacles,
        bounds,
        max_iters=1000,
        r_step=params.step_size,
        goal_radius=10.0,
        K_att=params.K_att,
        K_rep=params.K_rep,
        d0=params.d0,
    )


def benchmark_scenario(
    scenario: Dict[str, object],
    trials: int,
    model: PPO,
    normalizer: Dict[str, np.ndarray],
    deterministic: bool,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    print(f"\n{'='*80}")
    print(f"Testing Scenario: {scenario['name']} ({scenario['num_obs']} obstacles)")
    print(f"{'='*80}")

    baseline_stats = {"time": [], "nodes": [], "length": [], "success": []}
    rl_stats = {"time": [], "nodes": [], "length": [], "success": []}
    trial_rows: List[Dict[str, object]] = []

    # Pre-generate seeds so baseline/RL share identical initial conditions
    seeds = np.random.SeedSequence(42).spawn(trials)

    for idx, seed in enumerate(seeds):
        # Seed numpy/random with a Python int to keep random generation compatible
        base_seed = int(seed.generate_state(1, dtype=np.uint32)[0])
        np.random.seed(base_seed)
        start = tuple(np.random.uniform(-40, -20, 3))
        goal = tuple(np.random.uniform(20, 40, 3))
        obstacle_seed = int(seed.generate_state(1, dtype=np.uint32)[0])
        obstacles = create_random_spheres(
            num=scenario["num_obs"],
            bounds=scenario["bounds"],
            rmin=5,
            rmax=10,
            seed=obstacle_seed,
        )

        # Baseline
        try:
            baseline_params = PlannerParameters()
            baseline_params.step_size = 5.0
            baseline_params.K_att = 1.0
            baseline_params.K_rep = 0.2
            baseline_params.d0 = 15.0

            start_time = time.perf_counter()
            path_base, nodes_base, parents_base, time_base = run_planner(
                start,
                goal,
                obstacles,
                scenario["bounds"],
                baseline_params,
            )
            elapsed_base = time.perf_counter() - start_time if time_base is None else time_base

            if path_base is not None:
                record_trial(
                    baseline_stats,
                    path_base,
                    nodes_base,
                    elapsed_base,
                    obstacles,
                    node_count=len(nodes_base),
                )
            else:
                baseline_stats["success"].append(0)
        except Exception as exc:
            print(f"   Baseline trial {idx} failed: {exc}")
            baseline_stats["success"].append(0)

        # RL-enhanced using unified planner
        try:
            scenario_cfg = ScenarioConfig(
                difficulty="medium",
                n_joints=3,
                dynamic_probability=0.0,
            )
            rl_result = plan_with_rl_for_benchmark(
                start,
                goal,
                obstacles,
                scenario_cfg,
                model,
                normalizer,
                max_attempts=3,
            )

            if rl_result.success:
                record_trial(
                    rl_stats,
                    rl_result.path,
                    rl_result.path,
                    rl_result.planning_time,
                    obstacles,
                    path_length_override=rl_result.path_length,
                    node_count=rl_result.num_nodes,
                )
            else:
                rl_stats["success"].append(0)
        except Exception as exc:
            print(f"   RL trial {idx} failed: {exc}")
            rl_stats["success"].append(0)

        if (idx + 1) % 10 == 0:
            print(f"  Progress: {idx + 1}/{trials} trials completed")

        trial_rows.append(
            {
                "scenario": scenario["name"],
                "trial": idx,
                "start": start,
                "goal": goal,
                "baseline_success": baseline_stats["success"][-1] if baseline_stats["success"] else 0,
                "rl_success": rl_stats["success"][-1] if rl_stats["success"] else 0,
                "baseline_time": baseline_stats["time"][-1] if baseline_stats["time"] else np.nan,
                "rl_time": rl_stats["time"][-1] if rl_stats["time"] else np.nan,
                "baseline_nodes": baseline_stats["nodes"][-1] if baseline_stats["nodes"] else np.nan,
                "rl_nodes": rl_stats["nodes"][-1] if rl_stats["nodes"] else np.nan,
                "baseline_length": baseline_stats["length"][-1] if baseline_stats["length"] else np.nan,
                "rl_length": rl_stats["length"][-1] if rl_stats["length"] else np.nan,
            }
        )

    base_sr, base_time, base_nodes, base_length = calc_stats(baseline_stats)
    rl_sr, rl_time, rl_nodes, rl_length = calc_stats(rl_stats)

    def improvement(base: float, rl: float) -> float:
        if (
            base is None
            or rl is None
            or (isinstance(base, float) and np.isnan(base))
            or (isinstance(rl, float) and np.isnan(rl))
            or base == 0
        ):
            return np.nan
        return (base - rl) / base * 100

    time_improv = improvement(base_time, rl_time)
    nodes_improv = improvement(base_nodes, rl_nodes)
    length_improv = improvement(base_length, rl_length)

    scenario_row = {
        "Scenario": scenario["name"],
        "Baseline Success %": base_sr,
        "RL Success %": rl_sr,
        "Baseline Time (s)": base_time,
        "RL Time (s)": rl_time,
        "Time Improvement %": time_improv,
        "Baseline Nodes": base_nodes,
        "RL Nodes": rl_nodes,
        "Nodes Improvement %": nodes_improv,
        "Baseline Length (mm)": base_length,
        "RL Length (mm)": rl_length,
        "Length Improvement %": length_improv,
    }

    print(f"\n  Results for {scenario['name']}:")
    print(
        f"    Baseline: {fmt(base_sr, '{:.1f}')}% success, "
        f"{fmt(base_time)}s, {fmt(base_nodes, '{:.0f}')} nodes, {fmt(base_length, '{:.1f}')}mm"
    )
    print(
        f"    RL:       {fmt(rl_sr, '{:.1f}')}% success, "
        f"{fmt(rl_time)}s, {fmt(rl_nodes, '{:.0f}')} nodes, {fmt(rl_length, '{:.1f}')}mm"
    )
    print(
        "    Improvement: "
        f"{fmt(time_improv, '{:.1f}')}% time, "
        f"{fmt(nodes_improv, '{:.1f}')}% nodes, {fmt(length_improv, '{:.1f}')}% length"
    )

    return scenario_row, trial_rows


def main() -> None:
    args = parse_args()

    print("=" * 80)
    print("STEP 2 FINAL BENCHMARK: BASELINE vs RL-ENHANCED")
    print("=" * 80)

    model, normalizer = load_model(args.model_path, args.normalizer_path)

    scenarios = [
        {"name": "Simple", "num_obs": 2, "bounds": ((-50, 50), (-50, 50), (-50, 50))},
        {"name": "Medium", "num_obs": 5, "bounds": ((-50, 50), (-50, 50), (-50, 50))},
        {"name": "Complex", "num_obs": 8, "bounds": ((-50, 50), (-50, 50), (-50, 50))},
    ]

    results: List[Dict[str, object]] = []
    per_trial_rows: List[Dict[str, object]] = []

    for scenario in scenarios:
        scenario_row, trial_rows = benchmark_scenario(
            scenario, args.trials, model, normalizer, args.deterministic
        )
        results.append(scenario_row)
        per_trial_rows.extend(trial_rows)

    summary_df = pd.DataFrame(results)
    display_df = summary_df.copy()
    for column, precision in [
        ("Baseline Success %", "{:.1f}"),
        ("RL Success %", "{:.1f}"),
        ("Baseline Time (s)", "{:.2f}"),
        ("RL Time (s)", "{:.2f}"),
        ("Time Improvement %", "{:.1f}"),
        ("Baseline Nodes", "{:.0f}"),
        ("RL Nodes", "{:.0f}"),
        ("Nodes Improvement %", "{:.1f}"),
        ("Baseline Length (mm)", "{:.1f}"),
        ("RL Length (mm)", "{:.1f}"),
        ("Length Improvement %", "{:.1f}"),
    ]:
        display_df[column] = display_df[column].apply(lambda v: fmt(v, precision))

    print("\n" + "=" * 80)
    print("FINAL BENCHMARK RESULTS")
    print("=" * 80)
    print(display_df.to_string(index=False))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "step2_benchmark_results.csv"
    trials_path = output_dir / "step2_benchmark_trials.csv"
    summary_df.to_csv(summary_path, index=False)
    pd.DataFrame(per_trial_rows).to_csv(trials_path, index=False)
    print(f"\n Summary saved to: {summary_path}")
    print(f" Per-trial details saved to: {trials_path}")

    print("\n" + "=" * 80)
    print("COMPARISON WITH PAPER TARGETS")
    print("=" * 80)
    print("Target improvements from paper:")
    print("  - Planning time: ↓44% (1.79s → 1.0s)")
    print("  - Nodes explored: ↓31% (726 → 500)")
    print("  - Path length: ↓9% (1016mm → 920mm)")

    print("\nYour results:")
    avg_time_improv = pd.to_numeric(summary_df["Time Improvement %"], errors="coerce").mean()
    avg_nodes_improv = pd.to_numeric(summary_df["Nodes Improvement %"], errors="coerce").mean()
    avg_length_improv = pd.to_numeric(summary_df["Length Improvement %"], errors="coerce").mean()
    print(f"  - Planning time: ↓{avg_time_improv:.1f}%")
    print(f"  - Nodes explored: ↓{avg_nodes_improv:.1f}%")
    print(f"  - Path length: ↓{avg_length_improv:.1f}%")

    if avg_time_improv >= 40 and avg_nodes_improv >= 25 and avg_length_improv >= 5:
        print("\n RL model meets or exceeds paper targets!")
    else:
        print("\n⚠️  Results below paper targets. Consider:")
        print("   - Training longer (current: 5M steps, try 10M)")
        print("   - Tuning reward function")
        print("   - Adjusting PPO hyperparameters")

    print("=" * 80)


if __name__ == "__main__":
    main()
