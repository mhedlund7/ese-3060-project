# ESE 3060 Final Project Fall 2025

Contributors: Marcus Hedlund, Sofia Hedlund

## Part 1: CIFAR-10 Speedrun

### Quick Benchmark: Best HardSwish Configuration

To run a quick benchmark of our best HardSwish modification (similar to `orig_airbench94.py`):

```bash
git clone https://github.com/mhedlund7/ese-3060-project.git
cd ese-3060-project
pip install -r requirements.txt
python airbench94_lion_hardswish_experiments.py --mode hardswish --lr 12.0 --runs 25
```

This runs 25 trials with HardSwish activation and learning rate 12.0, matching the format of the original `orig_airbench94.py` benchmark.

### Final Statistical Benchmark

To reproduce our final benchmark results we used for our final statistics in our report, run the comprehensive benchmark script:

```bash
python final_benchmark.py
```

This script runs 200 independent trials for each configuration:

- **Baseline**: GELU activation with Nesterov SGD (LR=11.5)
- **HardSwish**: HardSwish activation with Nesterov SGD (LR=12.0)
- **Lion**: Lion optimizer with best configuration (LR=20.0, WD=0.03, beta2=0.99)

The script generates:

- Statistical analysis (mean, std, 95% confidence intervals, p-values)
- Training curve plots with error bars
- Comprehensive logs in `logs/` directory (both `.pt` and `.json` formats)

### Experimentation Process

Our final report detailing our full process can be found [here]().

**Code Evolution:**

1. Started with the base `airbench94.py` file (now saved as `orig_airbench94.py`) and added runtime tracking
2. Created `airbench94_compilation_trials.py` and `airbench94_preliminary_experiments.py` to test various modifications
3. Focused on two main ablations: **Lion optimizer** and **HardSwish activation**
4. Final experiments implemented in `airbench94_lion_hardswish_experiments.py`
5. Comprehensive statistical analysis performed with `final_benchmark.py`

**Hardware:**

- All experiments run on RunPod using A100 PCIe GPUs
- Small screening experiments: spot instances
- Final 200-run benchmarks: on-demand instances for consistency

### Command Line Arguments for Experiments

The `airbench94_lion_hardswish_experiments.py` script supports the following command-line arguments:

**Mode Selection:**

- `--mode`: Experiment mode (default: `baseline`)
  - `baseline`: GELU activation + Nesterov SGD
  - `hardswish`: HardSwish activation + Nesterov SGD
  - `lion`: GELU activation + Lion optimizer
  - `lion_lookahead`: Lion optimizer with Lookahead wrapper
  - `lion_hardswish`: Lion optimizer + HardSwish activation

**Hyperparameters:**

- `--runs`: Number of training runs to perform (default: 5)
- `--lr`: Override base learning rate (default: uses value from `hyp` dict)
- `--wd`: Override weight decay (default: uses value from `hyp` dict)
- `--beta2`: Override beta2 parameter for Lion optimizer (default: 0.99)
- `--bias_scaler`: Override bias scaler (default: 64.0)
- `--label_smoothing`: Override label smoothing rate (default: 0.2)
- `--epochs`: Override training epochs (default: 9.9)

**Example Commands:**

```bash
# Run baseline with default settings (5 runs)
python airbench94_lion_hardswish_experiments.py --mode baseline

# Run HardSwish with custom learning rate (25 runs)
python airbench94_lion_hardswish_experiments.py --mode hardswish --lr 12.0 --runs 25

# Run Lion optimizer with custom hyperparameters
python airbench94_lion_hardswish_experiments.py --mode lion --lr 20.0 --wd 0.03 --beta2 0.99 --runs 10

# Run HardSwish with custom epochs and label smoothing
python airbench94_lion_hardswish_experiments.py --mode hardswish --lr 12.0 --epochs 10.0 --label_smoothing 0.25 --runs 5
```

**Output:**

- Per-epoch training metrics printed to console
- Summary statistics (mean ± std) for accuracy and runtime printed to console
- Logs saved to `logs/Cifar10Logs/{uuid}/log.txt` containing:
  - Full code snapshot
  - Raw accuracy and runtime arrays (as PyTorch tensors)
  - Hyperparameters used
  - Experiment mode
- **Note**: Statistics are computed and displayed in the console output, but the log file contains only raw data. To recompute statistics, load the `.pt` file and calculate from the `accs` and `times` arrays.

**For `final_benchmark.py` only:**

- Comprehensive statistics are saved in both `.pt` and `.json` formats
- The `.json` file is human-readable and contains all computed statistics (mean, std, confidence intervals, p-values, etc.)
- Statistics can be viewed directly in the JSON file without needing to load and process the data

## Part 2: NanoGPT Speedrun

### Quick Benchmark: Best AdAGo Configuration

To run a quick benchmark with our best AdAGo optimizer configuration:

```bash
git clone https://github.com/mhedlund7/ese-3060-project.git
cd ese-3060-project
pip install -r requirements.txt

# Download the FineWeb-10B dataset (if not already downloaded)
python cached_fineweb10B.py 9
mkdir -p data/fineweb10B
mv fineweb10B/fineweb_train_*.bin fineweb10B/fineweb_val_*.bin data/fineweb10B/

# Run training with AdAGo optimizer
torchrun --standalone --nproc_per_node=8 adago_check_train_gpt.py \
    --learning-rate 0.0036 \
    --block-lr-mult 5.0 \
    --adago-gamma 1.0 \
    --adago-eps 5e-08
```

This runs the full 5100-step training with AdAGo optimizer for transformer blocks and AdamW for the language model head.

### Short Run for Testing

For faster testing and development, use the short run mode (500 iterations):

```bash
SHORT_RUN=1 torchrun --standalone --nproc_per_node=8 adago_check_train_gpt.py \
    --learning-rate 0.0036 \
    --block-lr-mult 100.0 \
    --adago-gamma 1.0 \
    --adago-eps 0.0005
```

### Experimentation Process

Our final report detailing our full process can be found ~~[here]().~~

**Code Evolution:**

1. Started with the base `train_gpt.py` file (now saved as `orig_train_gpt.py`) and added comprehensive logging
2. Created `adago_check_train_gpt.py` to implement the AdAGo optimizer (AdaGrad-style adaptive gradient orthogonalization)
3. Performed hyperparameter sweeps for learning rate, epsilon (`eps`), gamma, and block learning rate multiplier
4. Tested different optimizer configurations: Muon (baseline), Adagrad, and AdAGo
5. Final experiments focused on AdAGo with optimized hyperparameters

**Hardware:**

- All experiments run on RunPod using 8× A100 PCIe GPUs with on-demand instances
- Distributed training using PyTorch's `torchrun` with 8 processes

### Command Line Arguments for Experiments

The `adago_check_train_gpt.py` script supports the following command-line arguments:

**Optimizer Selection:**

- `BLOCK_OPTIMIZER` environment variable: Choose optimizer for transformer blocks (default: `adago`)
  - `muon`: Muon optimizer (baseline - momentum with orthogonalized updates)
  - `adagrad`: Standard Adagrad optimizer
  - `adago`: AdAGo optimizer (AdaGrad-style adaptive gradient orthogonalization)

**Hyperparameters:**

- `--learning-rate`: Base learning rate (default: `0.0036`)
- `--block-lr-mult`: Learning rate multiplier for transformer block optimizer (default: `100.0`)
  - Final block optimizer LR = `block_lr_mult * 0.1 * learning_rate`
- `--adago-gamma`: Gamma parameter for AdAGo (gradient norm capping, default: `1.0`)
- `--adago-eps`: Epsilon parameter for AdAGo (minimum step size floor, default: `5e-4`)

**Environment Variables:**

- `SHORT_RUN=1`: Enable short run mode (500 iterations instead of 5100, reduced validation tokens)

**Example Commands:**

```bash
# Run baseline Muon optimizer
BLOCK_OPTIMIZER=muon torchrun --standalone --nproc_per_node=8 adago_check_train_gpt.py \
    --learning-rate 0.0036

# Run AdAGo with default hyperparameters
torchrun --standalone --nproc_per_node=8 adago_check_train_gpt.py \
    --learning-rate 0.0036 \
    --block-lr-mult 100.0 \
    --adago-gamma 1.0 \
    --adago-eps 0.0005

# Run AdAGo with custom hyperparameters
torchrun --standalone --nproc_per_node=8 adago_check_train_gpt.py \
    --learning-rate 0.0036 \
    --block-lr-mult 5.0 \
    --adago-gamma 1.0 \
    --adago-eps 0.0005

# Short run for quick testing
SHORT_RUN=1 torchrun --standalone --nproc_per_node=8 adago_check_train_gpt.py \
    --learning-rate 0.0036 \
    --block-lr-mult 100.0
```

**Output:**

- **Console Output:**
  - Step-by-step training logs with `train_loss`, cumulative `train_time` (ms), and per-step average time
  - Periodic validation logs (every 125 steps by default) with `val_loss`, cumulative `train_time`, and per-step average time
  - Final summary:
    - Peak GPU memory usage (MiB)
    - Final training loss
    - Final validation loss
    - Total training time (ms and seconds)
    - Paths to saved log files and training curves

- **Text Log File** - `logs/NanoGPT/{uuid}.txt`:
  - Experiment metadata (run ID, timestamps, random seed, DDP world size)
  - Full hyperparameter dump (`Hyperparameters` dataclass)
  - Model configuration (layers, heads, embedding dim, vocab size)
  - Optimizer configuration (AdamW for `lm_head`, AdAGo/Muon/Adagrad for transformer blocks)
  - Full training script source code snapshot
  - `nvidia-smi` output for hardware context
  - All training and validation log lines (same format as console)
  - Final results block (peak memory, final losses, total training time)

- **JSON Log File** - `logs/NanoGPT/{uuid}.json`:
  - Same high-level metadata (run ID, start/end time, hyperparameters, model & optimizer config)
  - Scalar results:
    - `training_time_ms`, `peak_memory_mib`, `final_train_loss`, `final_val_loss`
  - Full training traces under `training_data`:
    - `train_losses` (per logged step)
    - `val_losses` (per validation evaluation)
    - `train_times_ms` (cumulative train time at each validation)
    - `steps` (corresponding iteration indices)

- **Training Curves** - `logs/NanoGPT/{uuid}/training_curves.png`:
  - Two-panel plot showing training loss and validation loss curves over training steps