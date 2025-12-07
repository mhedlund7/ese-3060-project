import os
import torch
import numpy as np
from scipy import stats
from airbench94_experiments import main, hyp

N_RUNS = 200

def run_experiment(mode, lr, runs):
    print(f"\n--- Running {runs} trials for {mode} (LR={lr}) ---")
    accuracies = []
    times = []
    
    class Args:
        def __init__(self, mode, lr):
            self.mode = mode
            self.lr = lr
            self.wd = None
            self.beta2 = None
            self.bias_scaler = None
            self.label_smoothing = None
            self.epochs = None

    args = Args(mode, lr)

    for i in range(runs):
        acc, time = main(f"bench_{mode}_{i}", args)
        accuracies.append(acc)
        times.append(time)
        print(f"Run {i+1}/{runs}: Acc={acc:.4f}, Time={time:.4f}s")

    return np.array(accuracies), np.array(times)

if __name__ == "__main__":
    print(f"Starting Rigorous Benchmark (N={N_RUNS})...")
    
    # 1. Run Baseline
    base_acc, base_time = run_experiment('baseline', 11.5, N_RUNS)
    
    # 2. Run HardSwish
    our_acc, our_time = run_experiment('hardswish', 12.0, N_RUNS)

    # 3. Statistical Analysis
    print("\n" + "="*40)
    print("FINAL STATISTICAL REPORT")
    print("="*40)

    # Accuracy Stats
    print(f"Accuracy (Baseline): {base_acc.mean():.4f} ± {base_acc.std():.4f}")
    print(f"Accuracy (Ours):     {our_acc.mean():.4f} ± {our_acc.std():.4f}")
    t_stat_acc, p_val_acc = stats.ttest_ind(base_acc, our_acc, equal_var=False)
    print(f"Accuracy P-Value:    {p_val_acc:.4e} ({'Significant' if p_val_acc < 0.05 else 'Not Significant'})")

    print("-" * 20)

    # Time Stats
    print(f"Time (Baseline):     {base_time.mean():.4f}s ± {base_time.std():.4f}s")
    print(f"Time (Ours):         {our_time.mean():.4f}s ± {our_time.std():.4f}s")
    t_stat_time, p_val_time = stats.ttest_ind(base_time, our_time, equal_var=False)
    print(f"Time P-Value:        {p_val_time:.4e} ({'Significant' if p_val_time < 0.05 else 'Not Significant'})")
    
    speedup = (base_time.mean() - our_time.mean()) / base_time.mean() * 100
    print(f"\nObserved Speedup: {speedup:.2f}%")