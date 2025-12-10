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

### Experimentation Process

Our final report detailing our full process can be found ~~[here]().~~

**Code Evolution:**

1. Started with the base `train_gpt.py` file (now saved as `orig_train_gpt.py`) and added runtime tracking
2. Created `adagrad_check_train_gpt.py` and `airbench94_preliminary_experiments.py` to test various modifications
3. Performed hyperparameter sweeps for learning rate, epsilon, and gamma of AdaGO
4. ~~Final experiments implemented in `airbench94_lion_hardswish_experiments.py`~~
5. ~~Comprehensive statistical analysis performed with `final_benchmark.py`~~

**Hardware:**

- All experiments run on RunPod using 8 A100 PCIe GPUs with on-demand instances

**Output:**

- Step-by-step training logs with `train_loss`, running `train_time` (ms), and per-step average time
- Periodic and final validation logs with `val_loss`, cumulative `train_time`, and per-step average time
- Final summary:
  - Peak GPU memory usag (MiB)
  - Final training loss
  - Final validation loss
  - Total training time (ms and seconds)
- Text log file - `logs/NanoGPT/{uuid}/log.txt`
  * Experiment metadata (run ID, timestamps, random seed, world size).
  * Full hyperparameter dump (`Hyperparameters` dataclass).
  * Model configuration (layers, heads, embedding dim, vocab size).
  * Optimizer configuration (AdamW for `lm_head`, AdAGo/Muon/Adagrad for blocks).
  * Full training script source code snapshot.
  * `nvidia-smi` output for hardware context.
  * All training and validation log lines (same format as console).
  * Final results block (peak memory, final losses, total training time).
- JSON log file - `logs/NanoGPT/{uuid}/log.json`
  * Same high-level metadata (run ID, start/end time, hyperparameters, model & optimizer config).
  * Scalar results:
    * `training_time_ms`, `peak_memory_mib`, `final_train_loss`, `final_val_loss`.
  * Full training traces under `training_data`:
    * `train_losses` (per logged step)
    * `val_losses` (per validation evaluation)
    * `train_times_ms` (cumulative train time at each validation)
    * `steps` (corresponding iteration indices)

# Original Project Information

## Project Overview

This project contains two machine learning training benchmarks:

- **airbench94.py**: CIFAR-10 image classification benchmark
- **train_gpt.py**: GPT-2 training on the FineWeb-10B dataset

## Setup and Installation

### Prerequisites

- Python 3.8+
- NVIDIA GPU (A100/H100 recommended)
- CUDA 11.7 or later

### Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

## Running airbench94.py

### Overview

CIFAR-10 training benchmark achieving 94.01% average accuracy in 3.83 seconds on an NVIDIA A100. You will want to use a single node of an a100.

- CIFAR-10 dataset automatically downloaded on first run
- Cached to `cifar10/` directory as `.pt` files for faster subsequent runs

### Execution

```bash
python airbench94.py
```

Runs 25 training iterations and reports mean/standard deviation accuracy metrics.

### Output

- Per-epoch training metrics (loss, accuracy)
- Validation and test-time augmentation (TTA) accuracy
- Logs saved to `logs/{uuid}/log.pt`

### Hardware Requirements

- NVIDIA A100 GPU recommended
- CUDA 11.7+
- NVIDIA Driver 515.105.01 or compatible

### Reference

Based on: [cifar10-airbench legacy airbench94.py](https://github.com/KellerJordan/cifar10-airbench/blob/master/legacy/airbench94.py)

## Running train_gpt.py

### Overview

Trains a GPT-2 model on the FineWeb-10B dataset. You will want to use an 8xH100.

### Execution

Download the data with

```bash
python cached_fineweb10B.py 9
```

and then run the script with

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Hardware Requirements

- Tested on 8× NVIDIA H100 80GB GPUs
- PyTorch 2.4.1+ with CUDA 12.1

### Reference

Based on: [modded-nanogpt record number #5](https://github.com/KellerJordan/modded-nanogpt/blob/master/records/track_1_short/2024-10-14_ModernArch/dabaaddd-237c-4ec9-939d-6608a9ed5e27.txt)
