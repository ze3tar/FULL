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
    from rl_enhanced_apf_rrt import APFRRTEnv, PlannerParameters, ScenarioConfig
except ImportError as e:  # pragma: no cover - import guard
    print(f"❌ Import error: {e}")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials",
        type=int,
        default=60,
        help="Number of trials per scenario (default: 60)",
    )
    parser.add_argument(
        "--model-path",
        default="/mnt/user-data/uploads/best_model_zip.zip",
        help="Path to PPO policy zip (default: /mnt/user-data/uploads/best_model_zip.zip)",
    )
    parser.add_argument(
        "--normalizer-path",
        default="/mnt/user-data/uploads/obs_normalizer.npz",
        help="Path to observation normalizer (default: /mnt/user-data/uploads/obs_normalizer.npz)",
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


def load_model(model_path: str, normalizer_path: str) -> Tuple[PPO, Dict[str, np.ndarray]]:
    print("\n1. Loading RL model...")
    try:
        model = PPO.load(model_path)
        normalizer = np.load(normalizer_path)
        print(f" Model loaded: {model_path}")
        print(f" Normalizer loaded: {normalizer_path}")
        return model, normalizer
    except Exception as exc:  # pragma: no cover - runtime guard
        print(f" Failed to load model: {exc}")
        sys.exit(1)


def normalize_observation(obs: np.ndarray, normalizer: Dict[str, np.ndarray]) -> np.ndarray:
    if "mean" in normalizer and "var" in normalizer:
        return (obs - normalizer["mean"]) / np.sqrt(normalizer["var"] + 1e-8)
    return obs


def calc_stats(stats_dict: Dict[str, List[float]]) -> Tuple[float, float, float, float]:
    success_rate = np.mean(stats_dict["success"]) * 100
    if sum(stats_dict["success"]) > 0:
        avg_time = np.mean([t for t, s in zip(stats_dict["time"], stats_dict["success"]) if s])
        avg_nodes = np.mean([n for n, s in zip(stats_dict["nodes"], stats_dict["success"]) if s])
        avg_length = np.mean([l for l, s in zip(stats_dict["length"], stats_dict["success"]) if s])
    else:
        avg_time = avg_nodes = avg_length = np.nan
    return success_rate, avg_time, avg_nodes, avg_length


def record_trial(
    stats: Dict[str, List[float]],
    path: Iterable[Tuple[float, float, float]],
    nodes: List[Tuple[float, float, float]],
    elapsed: float,
    obstacles: np.ndarray,
) -> None:
    pruned = prune_path(path, obstacles)
    stats["time"].append(elapsed)
    stats["nodes"].append(len(nodes))
    stats["length"].append(path_length(pruned))
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
        np.random.seed(seed.generate_state(1)[0])
        start = tuple(np.random.uniform(-40, -20, 3))
        goal = tuple(np.random.uniform(20, 40, 3))
        obstacles = create_random_spheres(
            num=scenario["num_obs"],
            bounds=scenario["bounds"],
            rmin=5,
            rmax=10,
            seed=seed.generate_state(1)[0],
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
                record_trial(baseline_stats, path_base, nodes_base, elapsed_base, obstacles)
            else:
                baseline_stats["success"].append(0)
        except Exception as exc:
            print(f"   Baseline trial {idx} failed: {exc}")
            baseline_stats["success"].append(0)

        # RL-enhanced
        try:
            env = APFRRTEnv(ScenarioConfig(difficulty="medium"), seed=seed.generate_state(1)[0])
            obs, _ = env.reset()
            obs_norm = normalize_observation(obs, normalizer)
            action, _ = model.predict(obs_norm, deterministic=deterministic)

            rl_params = PlannerParameters()
            rl_params.apply_delta(action)

            start_time = time.perf_counter()
            path_rl, nodes_rl, parents_rl, time_rl = run_planner(
                start,
                goal,
                obstacles,
                scenario["bounds"],
                rl_params,
            )
            elapsed_rl = time.perf_counter() - start_time if time_rl is None else time_rl

            if path_rl is not None:
                record_trial(rl_stats, path_rl, nodes_rl, elapsed_rl, obstacles)
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

    time_improv = ((base_time - rl_time) / base_time * 100) if not np.isnan(base_time) else 0
    nodes_improv = ((base_nodes - rl_nodes) / base_nodes * 100) if not np.isnan(base_nodes) else 0
    length_improv = ((base_length - rl_length) / base_length * 100) if not np.isnan(base_length) else 0

    scenario_row = {
        "Scenario": scenario["name"],
        "Baseline Success %": f"{base_sr:.1f}",
        "RL Success %": f"{rl_sr:.1f}",
        "Baseline Time (s)": f"{base_time:.2f}" if not np.isnan(base_time) else "N/A",
        "RL Time (s)": f"{rl_time:.2f}" if not np.isnan(rl_time) else "N/A",
        "Time Improvement %": f"{time_improv:.1f}",
        "Baseline Nodes": f"{base_nodes:.0f}" if not np.isnan(base_nodes) else "N/A",
        "RL Nodes": f"{rl_nodes:.0f}" if not np.isnan(rl_nodes) else "N/A",
        "Nodes Improvement %": f"{nodes_improv:.1f}",
        "Baseline Length (mm)": f"{base_length:.1f}" if not np.isnan(base_length) else "N/A",
        "RL Length (mm)": f"{rl_length:.1f}" if not np.isnan(rl_length) else "N/A",
        "Length Improvement %": f"{length_improv:.1f}",
    }

    print(f"\n  Results for {scenario['name']}:")
    print(
        f"    Baseline: {base_sr:.1f}% success, {base_time:.2f}s, {base_nodes:.0f} nodes, {base_length:.1f}mm"
    )
    print(f"    RL:       {rl_sr:.1f}% success, {rl_time:.2f}s, {rl_nodes:.0f} nodes, {rl_length:.1f}mm")
    print(
        f"    Improvement: {time_improv:.1f}% time, {nodes_improv:.1f}% nodes, {length_improv:.1f}% length"
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
    print("\n" + "=" * 80)
    print("FINAL BENCHMARK RESULTS")
    print("=" * 80)
    print(summary_df.to_string(index=False))

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
    avg_time_improv = summary_df["Time Improvement %"].str.rstrip("%").astype(float).mean()
    avg_nodes_improv = summary_df["Nodes Improvement %"].str.rstrip("%").astype(float).mean()
    avg_length_improv = summary_df["Length Improvement %"].str.rstrip("%").astype(float).mean()
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
