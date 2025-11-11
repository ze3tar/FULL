#!/usr/bin/env python3
"""
Quick Test Script - Verifies All Components Work
Run this to check your setup before starting the full implementation
"""

import sys
import importlib
import subprocess

def print_header(text):
    print("\n" + "="*70)
    print(f" {text}")
    print("="*70)

def test_component(name, test_func):
    """Test a component and report status"""
    print(f"\n{'Testing:':<20} {name}")
    print("-" * 70)
    try:
        result = test_func()
        if result:
            print(f"{'Status:':<20} PASS")
            return True
        else:
            print(f"{'Status:':<20} FAIL")
            return False
    except Exception as e:
        print(f"{'Status:':<20} ERROR")
        print(f"{'Error:':<20} {str(e)}")
        return False

def test_dependencies():
    """Test if required packages are installed"""
    required_packages = {
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'torch': 'PyTorch',
        'gymnasium': 'Gymnasium',
        'stable_baselines3': 'Stable Baselines3',
        'pandas': 'Pandas'
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            importlib.import_module(package)
            print(f"  ✓ {name:<20} installed")
        except ImportError:
            print(f"  ✗ {name:<20} MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n  Install missing packages:")
        print(f"  pip install {' '.join(missing)}")
        return False
    return True

def test_baseline():
    """Test baseline APF-RRT"""
    from baseline_enhanced import run_experiment
    
    print("  Running baseline APF-RRT (seed=1, 3 scenarios)...")
    result = run_experiment(seed=1, show_plot=False, export_ros=False)
    
    success = (result['improved']['path'] is not None and
              len(result['improved']['nodes']) < 1000)
    
    if success:
        print(f"  ✓ Nodes: {len(result['improved']['nodes'])}")
        print(f"  ✓ Runtime: {result['improved']['runtime']:.2f}s")
    
    return success

def test_pso():
    """Test PSO path smoother"""
    from pso_path_smoother import PSOPathSmoother
    import numpy as np
    
    print("  Testing PSO smoother with synthetic path...")
    
    # Create simple test path
    t = np.linspace(0, 1, 5)
    test_path = np.column_stack([t, t**2, np.sin(2*np.pi*t)])
    
    smoother = PSOPathSmoother(n_particles=10, max_iters=10)
    smoothed, cost, metrics = smoother.smooth(
        test_path, 
        obstacles=None, 
        fixed_endpoints=True,
        verbose=False
    )
    
    success = (smoothed is not None and 
              metrics['final_cost'] < metrics['original_cost'])
    
    if success:
        print(f"  ✓ Cost improved: {metrics['improvement_percent']:.1f}%")
    
    return success

def test_rl_setup():
    """Test RL environment setup"""
    try:
        from config_space_apf_rrt import ConfigSpaceAPF_RRT, ConfigSpaceSettings

        print("  Creating RL training environment...")
        env = ConfigSpaceAPF_RRT(ConfigSpaceSettings(difficulty='easy'))

        print("  Testing environment step...")
        state, info = env.reset()
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)

        print(f"  ✓ State shape: {state.shape}")
        print(f"  ✓ Action shape: {action.shape}")
        print(f"  ✓ Reward: {reward:.2f}")

        env.close()
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_lstm_setup():
    """Test LSTM predictor setup"""
    try:
        from obstacle_predictor import ObstaclePredictorLSTM
        import torch
        
        print("  Creating LSTM model...")
        model = ObstaclePredictorLSTM()
        
        print("  Testing forward pass...")
        batch_size = 2
        seq_len = 10
        input_size = 6
        test_input = torch.randn(batch_size, seq_len, input_size)
        
        output = model(test_input)
        
        print(f"  ✓ Input shape: {test_input.shape}")
        print(f"  ✓ Output shape: {output.shape}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_comparison_framework():
    """Test comparison framework"""
    try:
        from comprehensive_comparison import PlanningBenchmark, create_test_scenarios
        
        print("  Creating test scenarios...")
        scenarios = create_test_scenarios()[:1]  # Just one scenario for quick test
        
        print("  Running mini benchmark (1 trial)...")
        benchmark = PlanningBenchmark()
        results_df = benchmark.run_comparison(
            scenarios, 
            n_trials=1, 
            use_pso=False,
            verbose=False
        )
        
        success_count = results_df['success'].sum()
        print(f"  ✓ Completed {len(results_df)} planning attempts")
        print(f"  ✓ {success_count} successful")
        
        return success_count > 0
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def check_gpu():
    """Check if GPU is available for training"""
    import torch
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  ✓ GPU Available: {gpu_name}")
        print(f"  ✓ GPU Memory: {gpu_memory:.1f} GB")
        return True
    else:
        print(f"  ℹ No GPU detected (CPU training will be slower)")
        return True  # Not a failure, just slower

def generate_quick_demo():
    """Generate a quick demonstration"""
    print("\n  Generating quick demonstration...")
    
    from baseline_enhanced import run_experiment
    
    result = run_experiment(seed=42, show_plot=True, export_ros=True)
    
    print("\n  Generated files:")
    import os
    files = [
        'path_points_baseline.csv',
        'path_points_improved.csv',
        'path_improved_ros.yaml',
        'obstacles.csv',
        'apf_rrt_comparison.png'
    ]
    
    for f in files:
        if os.path.exists(f):
            print(f"    ✓ {f}")
        else:
            print(f"    ✗ {f} (not found)")

def main():
    print_header("ML-Enhanced APF-RRT - Quick Test Suite")

    print("\nThis script tests all components to verify your setup.")
    print("It should take 2-3 minutes to complete.")
    print("\nPress Enter to start...")
    auto_start = ('--auto' in sys.argv) or ('-y' in sys.argv) or ('--yes' in sys.argv)

    if auto_start:
        print("\nAuto-start flag detected. Beginning tests immediately...")
    else:
        stdin = getattr(sys, 'stdin', None)
        if stdin is not None and stdin.isatty():
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                print("\nInput interrupted. Continuing with tests...")
        else:
            print("\nNon-interactive environment detected. Beginning tests automatically...")
    
    # Track results
    results = {}
    
    # Test 1: Dependencies
    print_header("Test 1/7: Python Dependencies")
    results['dependencies'] = test_component(
        "Package Dependencies",
        test_dependencies
    )
    
    if not results['dependencies']:
        print("\n⚠️  Please install missing packages before continuing.")
        print("   pip install numpy matplotlib torch gymnasium stable-baselines3 pandas")
        return
    
    # Test 2: Baseline
    print_header("Test 2/7: Baseline APF-RRT")
    results['baseline'] = test_component(
        "Baseline APF-RRT Planner",
        test_baseline
    )
    
    # Test 3: PSO
    print_header("Test 3/7: PSO Path Smoother")
    results['pso'] = test_component(
        "PSO Path Smoother",
        test_pso
    )
    
    # Test 4: RL Setup
    print_header("Test 4/7: RL Environment")
    results['rl'] = test_component(
        "RL Training Environment",
        test_rl_setup
    )
    
    # Test 5: LSTM Setup
    print_header("Test 5/7: LSTM Predictor")
    results['lstm'] = test_component(
        "LSTM Obstacle Predictor",
        test_lstm_setup
    )
    
    # Test 6: Comparison Framework
    print_header("Test 6/7: Comparison Framework")
    results['comparison'] = test_component(
        "Benchmarking Framework",
        test_comparison_framework
    )
    
    # Test 7: GPU Check
    print_header("Test 7/7: Hardware Check")
    results['gpu'] = test_component(
        "GPU Availability",
        check_gpu
    )
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"\nResults: {passed}/{total} tests passed")
    print()
    
    for test_name, passed in results.items():
        status = " PASS" if passed else " FAIL"
        print(f"  {test_name:<20} {status}")
    
    if passed == total:
        print("\n" + "="*70)
        print("   ALL TESTS PASSED! ")
        print("="*70)
        print("\n  Your setup is ready!")
        print("\n  Next steps:")
        print("    1. Read SUMMARY.md for the roadmap overview")
        print("    2. Start with Week 1-2 tasks (ROS integration)")
        print("    3. Run: python3 rl_enhanced_apf_rrt.py train")
        print("\n  Optional: Generate a quick demo?")
        response = 'n'
        if auto_start:
            print("  Skipping demo prompt in auto-start mode.")
        else:
            stdin = getattr(sys, 'stdin', None)
            if stdin is not None and stdin.isatty():
                try:
                    response = input("  Generate demo now? (y/n): ")
                except (EOFError, KeyboardInterrupt):
                    print("  Input interrupted. Skipping demo generation.")
            else:
                print("  Non-interactive environment detected. Skipping demo generation.")

        if response.lower() == 'y':
            generate_quick_demo()
    else:
        print("\n" + "="*70)
        print("  ⚠️  SOME TESTS FAILED")
        print("="*70)
        print("\n  Please fix the failing components before proceeding.")
        print("  Check error messages above for details.")
    
    print("\n" + "="*70)
    print("  Test Complete")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
