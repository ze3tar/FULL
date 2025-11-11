#!/usr/bin/env python3
"""
Comprehensive Comparison Framework for Path Planning Algorithms
Compares: Basic RRT, RRT*, RRT-Connect, APF-RRT, RL-APF-RRT, +PSO, +Prediction
Supports both static and dynamic environments with ROS visualization
"""

from pathlib import Path
import time
import json

try:
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
    raise ModuleNotFoundError(
        "NumPy is required for benchmarking. Install it with `pip install numpy` "
        "or run the notebook setup cell in Colab."
    ) from exc

try:
    import pandas as pd
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
    raise ModuleNotFoundError(
        "Pandas is required for benchmarking. Install it with `pip install pandas` "
        "or run the notebook setup cell in Colab."
    ) from exc

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
    raise ModuleNotFoundError(
        "Matplotlib is required for plotting benchmark results. Install it with "
        "`pip install matplotlib`."
    ) from exc

# Import all planners
from baseline_enhanced import (
    rrt_basic, rrt_apf_guided, prune_path, 
    create_random_spheres, path_length
)

try:
    from rl_enhanced_apf_rrt import RLEnhancedAPF_RRT
    RL_AVAILABLE = True
except:
    RL_AVAILABLE = False
    print("Warning: RL planner not available")

try:
    from obstacle_predictor import ObstaclePredictor, DynamicObstacleManager
    PREDICTOR_AVAILABLE = True
except:
    PREDICTOR_AVAILABLE = False
    print("Warning: Obstacle predictor not available")

try:
    from pso_path_smoother import PSOPathSmoother
    PSO_AVAILABLE = True
except:
    PSO_AVAILABLE = False
    print("Warning: PSO smoother not available")


class PlanningBenchmark:
    """
    Comprehensive benchmarking framework for path planning algorithms
    """
    
    def __init__(self, bounds=((0,500),(0,500),(0,300))):
        self.bounds = bounds
        self.results = []
        
    def run_comparison(self, scenarios, n_trials=10, use_pso=True, verbose=True):
        """
        Run comprehensive comparison across multiple scenarios
        
        Args:
            scenarios: List of scenario dicts with 'start', 'goal', 'obstacles', 'name'
            n_trials: Number of trials per algorithm per scenario
            use_pso: Whether to apply PSO smoothing
            verbose: Print progress
        
        Returns:
            results_df: DataFrame with all results
        """
        if verbose:
            print("\n" + "="*70)
            print(" "*20 + "PATH PLANNING BENCHMARK")
            print("="*70)
            print(f"Scenarios: {len(scenarios)}")
            print(f"Trials per algorithm: {n_trials}")
            print(f"PSO smoothing: {use_pso}")
            print("="*70 + "\n")
        
        pso_smoother = PSOPathSmoother(verbose=False) if use_pso and PSO_AVAILABLE else None
        
        for scenario_idx, scenario in enumerate(scenarios):
            if verbose:
                print(f"\n{'='*70}")
                print(f"Scenario {scenario_idx + 1}/{len(scenarios)}: {scenario['name']}")
                print(f"{'='*70}")
            
            start = scenario['start']
            goal = scenario['goal']
            obstacles = scenario['obstacles']
            
            # Test each algorithm
            algorithms = [
                ('Basic RRT', self._run_basic_rrt),
                ('APF-RRT (Baseline)', self._run_apf_rrt),
            ]
            
            if RL_AVAILABLE:
                algorithms.append(('RL-APF-RRT', self._run_rl_apf_rrt))
            
            for algo_name, algo_func in algorithms:
                if verbose:
                    print(f"\n{algo_name}:")
                    print("-" * 50)
                
                for trial in range(n_trials):
                    result = algo_func(start, goal, obstacles, self.bounds)
                    result['scenario'] = scenario['name']
                    result['algorithm'] = algo_name
                    result['trial'] = trial
                    result['pso_applied'] = False
                    
                    # Apply PSO if path found and enabled
                    if result['success'] and pso_smoother and use_pso:
                        result_pso = self._apply_pso_smoothing(
                            result, obstacles, pso_smoother
                        )
                        result_pso['algorithm'] = f"{algo_name} + PSO"
                        result_pso['pso_applied'] = True
                        self.results.append(result_pso)
                    
                    self.results.append(result)
                    
                    if verbose and (trial + 1) % max(1, n_trials//5) == 0:
                        print(f"  Trial {trial + 1}/{n_trials} complete")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(self.results)
        
        if verbose:
            print("\n" + "="*70)
            print("BENCHMARK COMPLETE")
            print("="*70)
            self._print_summary(results_df)
        
        return results_df
    
    def _run_basic_rrt(self, start, goal, obstacles, bounds):
        """Run basic RRT"""
        start_time = time.time()
        path, nodes, parents, plan_time = rrt_basic(
            start, goal, obstacles, bounds,
            max_iters=8000, step_size=30.0, 
            goal_radius=20.0, goal_bias=0.05
        )
        
        success = path is not None
        
        return {
            'success': success,
            'planning_time': plan_time,
            'nodes_explored': len(nodes),
            'path_length': path_length(path) if success else np.nan,
            'path': path,
            'nodes': nodes
        }
    
    def _run_apf_rrt(self, start, goal, obstacles, bounds):
        """Run APF-guided RRT"""
        start_time = time.time()
        path, nodes, parents, plan_time = rrt_apf_guided(
            start, goal, obstacles, bounds,
            max_iters=8000, r_step=35.0,
            goal_radius=20.0, goal_bias=0.07,
            K_att=1.0, K_rep=0.3, d0=100.0
        )
        
        success = path is not None
        
        return {
            'success': success,
            'planning_time': plan_time,
            'nodes_explored': len(nodes),
            'path_length': path_length(path) if success else np.nan,
            'path': path,
            'nodes': nodes
        }
    
    def _run_rl_apf_rrt(self, start, goal, obstacles, bounds):
        """Run RL-enhanced APF-RRT"""
        if not RL_AVAILABLE:
            return {'success': False}

        # Convert to numpy arrays and normalize
        start_arr = np.asarray(start, dtype=float) / 100.0
        goal_arr  = np.asarray(goal, dtype=float)  / 100.0

        # Desired planner dimension
        DESIRED_DIM = 6

        # Helper to pad or trim vectors
        def pad_vec(v, dim=DESIRED_DIM):
            v = np.asarray(v, dtype=float)
            if v.size < dim:
                return np.concatenate([v, np.zeros(dim - v.size, dtype=float)])
            else:
                return v[:dim]

        q_start = pad_vec(start_arr, DESIRED_DIM)
        q_goal  = pad_vec(goal_arr,  DESIRED_DIM)

        # Normalize and pad obstacles
        q_obstacles = []
        for item in obstacles:
            center = np.asarray(item[0], dtype=float) / 100.0
            radius = float(item[1]) / 100.0
            q_obstacles.append((pad_vec(center, DESIRED_DIM), radius))

        # Instantiate planner without requiring a trained agent.  When the agent
        # is missing the planner raises a clear ValueError which we interpret as
        # a failed trial (the benchmark should keep running for other planners).
        planner = RLEnhancedAPF_RRT(agent=None)
        try:
            path, nodes, plan_time, metrics = planner.plan(
                q_start, q_goal, q_obstacles, max_iters=8000
            )
            success = path is not None
        except ValueError as exc:
            success = False
            plan_time = float('nan')
            metrics = {}
            nodes = []
            path = None
            _ = exc  # Planner requires an agent; absence is handled as failure.

        # Convert back to original scale
        if success:
            path = [np.array(p) * 100.0 for p in path]
            nodes = [np.array(n) * 100.0 for n in nodes]

        return {
            'success': success,
            'planning_time': plan_time,
            'nodes_explored': len(nodes),
            'path_length': path_length(path) if success else np.nan,
            'path': path,
            'nodes': nodes,
            'K_att_final': metrics.get('K_att_final', np.nan),
            'K_rep_final': metrics.get('K_rep_final', np.nan)
        }



    
    def _apply_pso_smoothing(self, result, obstacles, pso_smoother):
        """Apply PSO smoothing to path"""
        if not result['success']:
            return result
        
        smoothed_path, cost, metrics = pso_smoother.smooth(
            result['path'], 
            obstacles=[(np.array(c), r) for c, r in obstacles],
            fixed_endpoints=True,
            verbose=False
        )
        
        result_copy = result.copy()
        result_copy['path'] = smoothed_path
        result_copy['path_length'] = metrics['final_length']
        result_copy['smoothness'] = metrics['final_smoothness']
        result_copy['pso_improvement'] = metrics['improvement_percent']
        
        return result_copy
    
    def _print_summary(self, results_df):
        """Print summary statistics"""
        print("\nSummary Statistics:")
        print("-" * 70)
        
        # Group by algorithm
        grouped = results_df.groupby('algorithm')
        
        for algo_name, group in grouped:
            success_rate = (group['success'].sum() / len(group)) * 100
            
            if success_rate > 0:
                avg_time = group[group['success']]['planning_time'].mean()
                avg_nodes = group[group['success']]['nodes_explored'].mean()
                avg_length = group[group['success']]['path_length'].mean()
                
                print(f"\n{algo_name}:")
                print(f"  Success Rate: {success_rate:.1f}%")
                print(f"  Avg Planning Time: {avg_time:.3f}s")
                print(f"  Avg Nodes Explored: {avg_nodes:.0f}")
                print(f"  Avg Path Length: {avg_length:.2f}")
            else:
                print(f"\n{algo_name}:")
                print(f"  Success Rate: {success_rate:.1f}%")
    
    def save_results(self, filename='benchmark_results.json'):
        """Save results to JSON"""
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        serializable_results = []
        for result in self.results:
            r = result.copy()
            # Convert arrays to lists
            if 'path' in r and r['path'] is not None:
                r['path'] = [p.tolist() if isinstance(p, np.ndarray) else p
                           for p in r['path']]
            if 'nodes' in r:
                r['nodes'] = [n.tolist() if isinstance(n, np.ndarray) else n
                            for n in r['nodes']]
            serializable_results.append(r)

        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"\nResults saved to {filename}")
    
    def plot_comparison(self, results_df, save_path='comparison_plots.png'):
        """Create comprehensive comparison plots"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=(18, 10))
        
        # Filter successful trials
        success_df = results_df[results_df['success']]
        
        # 1. Planning Time Comparison
        ax1 = fig.add_subplot(2, 3, 1)
        success_df.boxplot(column='planning_time', by='algorithm', ax=ax1)
        ax1.set_title('Planning Time Comparison')
        ax1.set_xlabel('Algorithm')
        ax1.set_ylabel('Time (s)')
        plt.sca(ax1)
        plt.xticks(rotation=45, ha='right')
        
        # 2. Nodes Explored
        ax2 = fig.add_subplot(2, 3, 2)
        success_df.boxplot(column='nodes_explored', by='algorithm', ax=ax2)
        ax2.set_title('Nodes Explored')
        ax2.set_xlabel('Algorithm')
        ax2.set_ylabel('Node Count')
        plt.sca(ax2)
        plt.xticks(rotation=45, ha='right')
        
        # 3. Path Length
        ax3 = fig.add_subplot(2, 3, 3)
        success_df.boxplot(column='path_length', by='algorithm', ax=ax3)
        ax3.set_title('Path Length')
        ax3.set_xlabel('Algorithm')
        ax3.set_ylabel('Length (mm)')
        plt.sca(ax3)
        plt.xticks(rotation=45, ha='right')
        
        # 4. Success Rate by Scenario
        ax4 = fig.add_subplot(2, 3, 4)
        success_rate = results_df.groupby(['algorithm', 'scenario'])['success'].mean() * 100
        success_rate.unstack().plot(kind='bar', ax=ax4)
        ax4.set_title('Success Rate by Scenario')
        ax4.set_xlabel('Algorithm')
        ax4.set_ylabel('Success Rate (%)')
        ax4.legend(title='Scenario')
        plt.xticks(rotation=45, ha='right')
        
        # 5. Efficiency Metrics
        ax5 = fig.add_subplot(2, 3, 5)
        efficiency = success_df.groupby('algorithm').agg({
            'planning_time': 'mean',
            'nodes_explored': 'mean',
            'path_length': 'mean'
        })
        efficiency_normalized = (efficiency - efficiency.min()) / (efficiency.max() - efficiency.min())
        efficiency_normalized.plot(kind='bar', ax=ax5)
        ax5.set_title('Normalized Efficiency Metrics')
        ax5.set_xlabel('Algorithm')
        ax5.set_ylabel('Normalized Score (0-1)')
        ax5.legend(['Time', 'Nodes', 'Length'])
        plt.xticks(rotation=45, ha='right')
        
        # 6. Overall Performance Score
        ax6 = fig.add_subplot(2, 3, 6)
        # Lower is better for all metrics, so invert
        performance_score = (
            1.0 / success_df.groupby('algorithm')['planning_time'].mean() +
            1.0 / success_df.groupby('algorithm')['nodes_explored'].mean() +
            1.0 / success_df.groupby('algorithm')['path_length'].mean()
        )
        performance_score = performance_score / performance_score.max() * 100
        performance_score.plot(kind='bar', ax=ax6, color='green', alpha=0.7)
        ax6.set_title('Overall Performance Score')
        ax6.set_xlabel('Algorithm')
        ax6.set_ylabel('Score (higher is better)')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nComparison plots saved to {save_path}")
        plt.show()


def create_test_scenarios():
    """Create test scenarios with varying difficulty"""
    scenarios = []
    
    # Scenario 1: Simple
    scenarios.append({
        'name': 'Simple',
        'start': (30.0, 30.0, 40.0),
        'goal': (450.0, 450.0, 240.0),
        'obstacles': create_random_spheres(num=3, rmin=30, rmax=50, seed=1)
    })
    
    # Scenario 2: Medium
    scenarios.append({
        'name': 'Medium',
        'start': (50.0, 50.0, 50.0),
        'goal': (400.0, 400.0, 200.0),
        'obstacles': create_random_spheres(num=5, rmin=25, rmax=60, seed=2)
    })
    
    # Scenario 3: Complex
    scenarios.append({
        'name': 'Complex',
        'start': (30.0, 30.0, 30.0),
        'goal': (470.0, 470.0, 270.0),
        'obstacles': create_random_spheres(num=7, rmin=30, rmax=70, seed=3)
    })
    
    # Scenario 4: Narrow passage
    scenarios.append({
        'name': 'Narrow',
        'start': (50.0, 50.0, 50.0),
        'goal': (450.0, 450.0, 250.0),
        'obstacles': [
            # Create a narrow passage
            ((200.0, 200.0, 150.0), 80.0),
            ((300.0, 300.0, 150.0), 80.0),
            ((250.0, 150.0, 150.0), 60.0),
            ((250.0, 350.0, 150.0), 60.0),
        ]
    })
    
    return scenarios


def main():
    """Main benchmarking routine"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Path Planning Benchmark')
    parser.add_argument('--trials', type=int, default=10,
                       help='Number of trials per algorithm')
    parser.add_argument('--pso', action='store_true',
                       help='Enable PSO smoothing')
    parser.add_argument('--save', type=str, default='benchmarks/final_benchmark',
                       help='Base name for saving results')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(" "*15 + "PATH PLANNING COMPREHENSIVE BENCHMARK")
    print("="*70)
    print(f"Trials per algorithm: {args.trials}")
    print(f"PSO smoothing: {args.pso}")
    print(f"RL available: {RL_AVAILABLE}")
    print(f"Predictor available: {PREDICTOR_AVAILABLE}")
    print(f"PSO available: {PSO_AVAILABLE}")
    print("="*70)
    
    # Create scenarios
    scenarios = create_test_scenarios()
    
    # Run benchmark
    benchmark = PlanningBenchmark()
    results_df = benchmark.run_comparison(
        scenarios, 
        n_trials=args.trials,
        use_pso=args.pso,
        verbose=True
    )
    
    # Save results
    save_base = Path(args.save)
    save_base.parent.mkdir(parents=True, exist_ok=True)

    benchmark.save_results(save_base.with_suffix('.json'))
    results_df.to_csv(save_base.with_suffix('.csv'), index=False)
    print(f"Results saved to {save_base.with_suffix('.csv')}")

    # Create plots
    benchmark.plot_comparison(results_df, save_base.with_name(save_base.name + '_plots.png'))
    
    print("\n✓ Benchmark complete!")
    print(f"\nGenerated files:")
    print(f"  - {args.save}.json")
    print(f"  - {args.save}.csv")
    print(f"  - {args.save}_plots.png")


if __name__ == "__main__":
    main()
