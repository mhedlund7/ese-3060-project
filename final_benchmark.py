import os
import sys
import uuid
import json
import torch
import numpy as np
from scipy import stats
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import airbench94_lion_hardswish_experiments as exp_module
from airbench94_lion_hardswish_experiments import hyp

N_RUNS = 200
BOOTSTRAP_ITERATIONS = 10000

def compute_confidence_interval(data, confidence=0.95):
    """Compute confidence interval using t-distribution."""
    n = len(data)
    if n < 2:
        return (float(data[0]), float(data[0]))
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return (mean - h, mean + h)

def compute_bootstrap_ci(data, confidence=0.95, n_iterations=10000):
    """Compute bootstrap confidence interval."""
    n = len(data)
    bootstrap_means = []
    
    for _ in range(n_iterations):
        # Resample with replacement
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    bootstrap_means = np.array(bootstrap_means)
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return (lower, upper)


def run_experiment(mode, lr, runs, wd=None, beta2=None, capture_curves=True):
    """Run an experiment and return accuracies, times, and training curves."""
    print(f"\n--- Running {runs} trials for {mode} (LR={lr}" + 
          (f", WD={wd}" if wd is not None else "") +
          (f", Beta2={beta2}" if beta2 is not None else "") + ") ---")
    accuracies = []
    times = []
    training_curves = []  # List of curves, each is a list of epoch dicts
    
    class Args:
        def __init__(self, mode, lr, wd, beta2, capture_curves=False):
            self.mode = mode
            self.lr = lr
            self.wd = wd
            self.beta2 = beta2
            self.bias_scaler = None
            self.label_smoothing = None
            self.epochs = None
            self.capture_curves = capture_curves

    args = Args(mode, lr, wd, beta2, capture_curves=capture_curves)
    
    # Warmup run (not counted)
    print("Performing warmup run...")
    warmup_args = Args(mode, lr, wd, beta2, capture_curves=False)
    exp_module.main('warmup', warmup_args)
    print("Warmup complete. Starting measured runs.\n")

    for i in range(runs):
        result = exp_module.main(f"bench_{mode}_{i}", args)
        if capture_curves and len(result) == 3:
            acc, time, curve = result
            training_curves.append(curve)
        else:
            acc, time = result
        accuracies.append(acc)
        times.append(time)
        if (i + 1) % 20 == 0 or i == 0:
            print(f"Run {i+1}/{runs}: Acc={acc:.4f}, Time={time:.4f}s")

    return np.array(accuracies), np.array(times), training_curves

def analyze_configuration(name, accs, times, baseline_accs=None, baseline_times=None):
    """Perform comprehensive statistical analysis for a configuration."""
    results = {
        'name': name,
        'n_runs': len(accs),
        'accuracy': {
            'mean': float(np.mean(accs)),
            'std': float(np.std(accs)),
            'median': float(np.median(accs)),
            'min': float(np.min(accs)),
            'max': float(np.max(accs)),
            'ci_95_t': compute_confidence_interval(accs),
            'ci_95_bootstrap': compute_bootstrap_ci(accs, n_iterations=BOOTSTRAP_ITERATIONS),
        },
        'time': {
            'mean': float(np.mean(times)),
            'std': float(np.std(times)),
            'median': float(np.median(times)),
            'min': float(np.min(times)),
            'max': float(np.max(times)),
            'ci_95_t': compute_confidence_interval(times),
            'ci_95_bootstrap': compute_bootstrap_ci(times, n_iterations=BOOTSTRAP_ITERATIONS),
        }
    }
    
    # Compare to baseline if provided
    if baseline_accs is not None and baseline_times is not None:
        # Accuracy comparison
        t_stat_acc, p_val_acc = stats.ttest_ind(baseline_accs, accs, equal_var=False)
        results['accuracy']['vs_baseline'] = {
            't_statistic': float(t_stat_acc),
            'p_value': float(p_val_acc),
            'significant': p_val_acc < 0.05,
            'difference': float(np.mean(accs) - np.mean(baseline_accs)),
        }
        
        # Time comparison
        t_stat_time, p_val_time = stats.ttest_ind(baseline_times, times, equal_var=False)
        speedup_pct = ((np.mean(baseline_times) - np.mean(times)) / np.mean(baseline_times)) * 100
        results['time']['vs_baseline'] = {
            't_statistic': float(t_stat_time),
            'p_value': float(p_val_time),
            'significant': p_val_time < 0.05,
            'speedup_percent': float(speedup_pct),
        }
    
    return results

def print_statistics(results):
    """Print formatted statistics for a configuration."""
    print(f"\n{'='*80}")
    print(f"CONFIGURATION: {results['name']}")
    print(f"{'='*80}")
    
    acc = results['accuracy']
    print(f"\n📊 ACCURACY")
    print(f"  Mean: {acc['mean']:.4f} ± {acc['std']:.4f}")
    print(f"  Median: {acc['median']:.4f}")
    print(f"  Range: [{acc['min']:.4f}, {acc['max']:.4f}]")
    print(f"  95% CI (t-dist): [{acc['ci_95_t'][0]:.4f}, {acc['ci_95_t'][1]:.4f}]")
    print(f"  95% CI (bootstrap): [{acc['ci_95_bootstrap'][0]:.4f}, {acc['ci_95_bootstrap'][1]:.4f}]")
    
    if 'vs_baseline' in acc:
        vs = acc['vs_baseline']
        print(f"  vs Baseline: Δ = {vs['difference']:+.4f}, t = {vs['t_statistic']:.2f}, "
              f"p = {vs['p_value']:.4e} {'✓' if vs['significant'] else '✗'}")
    
    time = results['time']
    print(f"\n⏱️  TIME")
    print(f"  Mean: {time['mean']:.4f}s ± {time['std']:.4f}s")
    print(f"  Median: {time['median']:.4f}s")
    print(f"  Range: [{time['min']:.4f}s, {time['max']:.4f}s]")
    print(f"  95% CI (t-dist): [{time['ci_95_t'][0]:.4f}s, {time['ci_95_t'][1]:.4f}s]")
    print(f"  95% CI (bootstrap): [{time['ci_95_bootstrap'][0]:.4f}s, {time['ci_95_bootstrap'][1]:.4f}s]")
    
    if 'vs_baseline' in time:
        vs = time['vs_baseline']
        print(f"  vs Baseline: Speedup = {vs['speedup_percent']:+.2f}%, t = {vs['t_statistic']:.2f}, "
              f"p = {vs['p_value']:.4e} {'✓' if vs['significant'] else '✗'}")

def aggregate_training_curves(curves):
    """
    Aggregate training curves across runs.
    Returns dict with mean ± std for each metric at each epoch.
    """
    if not curves or len(curves) == 0:
        return None
    
    # Find max epochs across all runs
    max_epochs = max(len(c) for c in curves)
    
    aggregated = {
        'epochs': list(range(1, max_epochs + 1)),
        'train_loss': {'mean': [], 'std': []},
        'train_acc': {'mean': [], 'std': []},
        'val_acc': {'mean': [], 'std': []},
    }
    
    for epoch_idx in range(max_epochs):
        # Collect values for this epoch across all runs
        epoch_losses = []
        epoch_train_accs = []
        epoch_val_accs = []
        
        for curve in curves:
            if epoch_idx < len(curve):
                epoch_losses.append(curve[epoch_idx]['train_loss'])
                epoch_train_accs.append(curve[epoch_idx]['train_acc'])
                epoch_val_accs.append(curve[epoch_idx]['val_acc'])
        
        if epoch_losses:
            aggregated['train_loss']['mean'].append(float(np.mean(epoch_losses)))
            aggregated['train_loss']['std'].append(float(np.std(epoch_losses)))
            aggregated['train_acc']['mean'].append(float(np.mean(epoch_train_accs)))
            aggregated['train_acc']['std'].append(float(np.std(epoch_train_accs)))
            aggregated['val_acc']['mean'].append(float(np.mean(epoch_val_accs)))
            aggregated['val_acc']['std'].append(float(np.std(epoch_val_accs)))
        else:
            # Pad with NaN if no data
            aggregated['train_loss']['mean'].append(float('nan'))
            aggregated['train_loss']['std'].append(float('nan'))
            aggregated['train_acc']['mean'].append(float('nan'))
            aggregated['train_acc']['std'].append(float('nan'))
            aggregated['val_acc']['mean'].append(float('nan'))
            aggregated['val_acc']['std'].append(float('nan'))
    
    return aggregated

def plot_training_curves(all_curves, output_path):
    """
    Create training curve plots with mean ± std for all configurations.
    
    Args:
        all_curves: dict mapping config_name -> aggregated_curve_dict
        output_path: where to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = {'baseline': '#1f77b4', 'hardswish': '#ff7f0e', 'lion': '#2ca02c'}
    labels = {'baseline': 'Baseline (GELU)', 'hardswish': 'HardSwish', 'lion': 'Lion'}
    
    # Plot 1: Training Loss
    ax = axes[0]
    for name, curve in all_curves.items():
        if curve is None:
            continue
        epochs = curve['epochs']
        mean = curve['train_loss']['mean']
        std = curve['train_loss']['std']
        # Filter out NaN
        valid = [not np.isnan(m) for m in mean]
        epochs_valid = [e for e, v in zip(epochs, valid) if v]
        mean_valid = [m for m, v in zip(mean, valid) if v]
        std_valid = [s for s, v in zip(std, valid) if v]
        
        ax.plot(epochs_valid, mean_valid, label=labels.get(name, name), 
                color=colors.get(name, 'gray'), linewidth=2)
        ax.fill_between(epochs_valid, 
                       np.array(mean_valid) - np.array(std_valid),
                       np.array(mean_valid) + np.array(std_valid),
                       alpha=0.2, color=colors.get(name, 'gray'))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Training Accuracy
    ax = axes[1]
    for name, curve in all_curves.items():
        if curve is None:
            continue
        epochs = curve['epochs']
        mean = curve['train_acc']['mean']
        std = curve['train_acc']['std']
        valid = [not np.isnan(m) for m in mean]
        epochs_valid = [e for e, v in zip(epochs, valid) if v]
        mean_valid = [m for m, v in zip(mean, valid) if v]
        std_valid = [s for s, v in zip(std, valid) if v]
        
        ax.plot(epochs_valid, mean_valid, label=labels.get(name, name),
                color=colors.get(name, 'gray'), linewidth=2)
        ax.fill_between(epochs_valid,
                       np.array(mean_valid) - np.array(std_valid),
                       np.array(mean_valid) + np.array(std_valid),
                       alpha=0.2, color=colors.get(name, 'gray'))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Accuracy')
    ax.set_title('Training Accuracy Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Validation Accuracy
    ax = axes[2]
    for name, curve in all_curves.items():
        if curve is None:
            continue
        epochs = curve['epochs']
        mean = curve['val_acc']['mean']
        std = curve['val_acc']['std']
        valid = [not np.isnan(m) for m in mean]
        epochs_valid = [e for e, v in zip(epochs, valid) if v]
        mean_valid = [m for m, v in zip(mean, valid) if v]
        std_valid = [s for s, v in zip(std, valid) if v]
        
        ax.plot(epochs_valid, mean_valid, label=labels.get(name, name),
                color=colors.get(name, 'gray'), linewidth=2)
        ax.fill_between(epochs_valid,
                       np.array(mean_valid) - np.array(std_valid),
                       np.array(mean_valid) + np.array(std_valid),
                       alpha=0.2, color=colors.get(name, 'gray'))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Validation Accuracy Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Training curves saved to: {os.path.abspath(output_path)}")
    plt.close()

def save_log(config_name, accs, times, hyperparams, results, training_curves=None):
    """Save comprehensive log file."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('logs', f"{config_name}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Aggregate training curves if provided
    aggregated_curves = None
    if training_curves and len(training_curves) > 0:
        aggregated_curves = aggregate_training_curves(training_curves)
    
    log = {
        'experiment_name': config_name,
        'timestamp': datetime.now().isoformat(),
        'n_runs': len(accs),
        'hyperparameters': hyperparams,
        'results': results,
        'raw_data': {
            'accuracies': accs.tolist(),
            'times': times.tolist(),
        },
        'statistics': results,
    }
    
    if aggregated_curves is not None:
        log['training_curves'] = aggregated_curves
        log['raw_training_curves'] = training_curves  # Keep raw data too
    
    # Save as both .pt and .json
    torch_path = os.path.join(log_dir, 'log.pt')
    json_path = os.path.join(log_dir, 'log.json')
    
    torch.save(log, torch_path)
    
    # JSON-safe version (convert numpy arrays)
    json_log = {k: v for k, v in log.items()}
    with open(json_path, 'w') as f:
        json.dump(json_log, f, indent=2, default=str)
    
    print(f"\n💾 Log saved to: {os.path.abspath(torch_path)}")
    print(f"   JSON: {os.path.abspath(json_path)}")
    
    return log_dir

if __name__ == "__main__":
    print(f"Starting Rigorous Benchmark (N={N_RUNS} runs per configuration)")
    print(f"Bootstrap iterations: {BOOTSTRAP_ITERATIONS}")
    print("="*80)
    
    all_results = {}
    
    all_curves = {}  # Store aggregated curves for plotting
    
    # 1. Run Baseline
    print("\n" + "="*80)
    print("CONFIGURATION 1: BASELINE (GELU + SGD, LR=11.5)")
    print("="*80)
    base_acc, base_time, base_curves = run_experiment('baseline', 11.5, N_RUNS, capture_curves=True)
    base_results = analyze_configuration('baseline', base_acc, base_time)
    print_statistics(base_results)
    base_log_dir = save_log('baseline', base_acc, base_time, 
                           {'mode': 'baseline', 'lr': 11.5}, base_results, base_curves)
    all_results['baseline'] = base_results
    all_curves['baseline'] = aggregate_training_curves(base_curves)
    
    # 2. Run HardSwish
    print("\n" + "="*80)
    print("CONFIGURATION 2: HARDSWISH (LR=12.0)")
    print("="*80)
    hs_acc, hs_time, hs_curves = run_experiment('hardswish', 12.0, N_RUNS, capture_curves=True)
    hs_results = analyze_configuration('hardswish', hs_acc, hs_time, base_acc, base_time)
    print_statistics(hs_results)
    hs_log_dir = save_log('hardswish', hs_acc, hs_time,
                         {'mode': 'hardswish', 'lr': 12.0}, hs_results, hs_curves)
    all_results['hardswish'] = hs_results
    all_curves['hardswish'] = aggregate_training_curves(hs_curves)
    
    # 3. Run Best Lion Configuration
    # Based on Table 1: Lion, wd=0.03, lr=20.0 achieved 0.9258 accuracy
    print("\n" + "="*80)
    print("CONFIGURATION 3: LION (Best Config: wd=0.03, lr=20.0)")
    print("="*80)
    lion_acc, lion_time, lion_curves = run_experiment('lion', 20.0, N_RUNS, wd=0.03, beta2=0.99, capture_curves=True)
    lion_results = analyze_configuration('lion', lion_acc, lion_time, base_acc, base_time)
    print_statistics(lion_results)
    lion_log_dir = save_log('lion', lion_acc, lion_time,
                           {'mode': 'lion', 'lr': 20.0, 'wd': 0.03, 'beta2': 0.99}, lion_results, lion_curves)
    all_results['lion'] = lion_results
    all_curves['lion'] = aggregate_training_curves(lion_curves)
    
    # 4. Final Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY - ALL CONFIGURATIONS")
    print("="*80)
    
    print(f"\n{'Configuration':<20} {'Acc Mean':<12} {'Acc CI (t)':<25} {'Time Mean':<12} {'Time CI (t)':<25} {'Speedup':<10}")
    print("-" * 120)
    
    for name, res in all_results.items():
        acc = res['accuracy']
        time = res['time']
        acc_ci = f"[{acc['ci_95_t'][0]:.4f}, {acc['ci_95_t'][1]:.4f}]"
        time_ci = f"[{time['ci_95_t'][0]:.4f}, {time['ci_95_t'][1]:.4f}]"
        
        speedup = ""
        if 'vs_baseline' in time:
            speedup = f"{time['vs_baseline']['speedup_percent']:+.2f}%"
        
        print(f"{name:<20} {acc['mean']:.4f}      {acc_ci:<25} {time['mean']:.4f}s      {time_ci:<25} {speedup:<10}")
    
    # Save combined summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'n_runs_per_config': N_RUNS,
        'bootstrap_iterations': BOOTSTRAP_ITERATIONS,
        'configurations': all_results,
    }
    
    summary_path = os.path.join('logs', f"benchmark_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n💾 Summary saved to: {os.path.abspath(summary_path)}")
    
    # Generate training curve plots
    print("\n" + "="*80)
    print("GENERATING TRAINING CURVES")
    print("="*80)
    curves_path = os.path.join('logs', f"training_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plot_training_curves(all_curves, curves_path)
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)