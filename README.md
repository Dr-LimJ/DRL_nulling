# Deep Reinforcement Learning-based Adaptive Nulling in Phased Array

Python implementation of the paper "Deep Reinforcement Learning-based Adaptive Nulling in Phased Array under Dynamic Environments" (IEEE Access, 2025).

## Overview

This project implements a PPO-based adaptive beamforming system for phased array antennas in dynamic environments. The system learns to optimize complex-valued antenna weights to maximize Signal-to-Interference Ratio (SIR) while both desired and interference signals move continuously.

## Key Features

- **End-to-end RL approach**: Direct complex weight control without pre-computed lookup tables
- **Dynamic environment**: Both desired and interference signals move with boundary bounce behavior
- **PPO algorithm**: Stable on-policy learning with clipped surrogate objective
- **TD(0) critic loss**: Temporal difference learning for value estimation
- **GAE**: Generalized Advantage Estimation for variance reduction
- **Paper-faithful architecture**: Implements exact network structure from Figure 5

## Project Structure

```
drl_beamforming/
├── requirements.txt          # Python dependencies
├── radar_env.py             # Dynamic RADAR environment (Gymnasium)
├── ppo_agent.py             # PPO agent implementation
├── train.py                 # Training script
├── evaluate.py              # Evaluation and visualization
└── README.md                # This file
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Training

```bash
python train.py
```

This will:
- Train PPO agent for 3000 episodes
- Save model and metrics to `results/ppo_training_YYYYMMDD_HHMMSS/`
- Generate training curves

### Evaluation

```bash
python evaluate.py --model_path results/ppo_training_YYYYMMDD_HHMMSS/trained_agent.pth
```

This will:
- Evaluate trained agent for 10 episodes
- Generate visualizations of:
  - Signal movement patterns
  - SIR evolution
  - Array factor pattern
  - Power levels

## Implementation Details

### Environment (RADAR_dynamic)

- **State**: 73-dimensional vector
  - 1 scalar: normalized desired angle [-1, 1]
  - 72 values: normalized covariance matrix (upper triangular)
  
- **Action**: 16-dimensional continuous
  - Real and imaginary parts of 8 complex antenna weights
  
- **Reward**: SIR (dB) with penalty for constraint violations
  - Penalty if PD ≤ 0 dB (desired signal suppressed)
  - Penalty if PI ≥ -10 dB (interference not suppressed)

### PPO Agent

**Actor Network** (6 layers, Figure 5):
```
Input (73) → 128 → 128 → 64 → 32 → 16 → [mean, log_std] (16+16)
```

**Critic Network** (5 layers, Figure 5):
```
Input (73) → 128 → 128 → 64 → 32 → 1
```

**Hyperparameters** (from paper):
- Learning rate: 3e-4
- Discount factor (γ): 0.99
- GAE lambda (λ): 0.95
- Clip range (ε): 0.2
- Buffer size: 2048
- Batch size: 128
- Epochs per update: 10

### Key Differences from MATLAB

1. **Framework**: MATLAB Deep Learning Toolbox → PyTorch
2. **Environment**: Custom MATLAB class → Gymnasium interface
3. **Efficiency**: 
   - Vectorized operations with NumPy/PyTorch
   - GPU acceleration support
   - Batch processing
4. **Modularity**: Separated components for easier experimentation

## Training Tips

### Hyperparameter Tuning

Critical hyperparameters:
- `clip_range`: Controls policy update magnitude (0.2-0.3 works well)
- `entropy_coeff`: Encourages exploration (0.01-0.05)
- `value_coeff`: Balances critic loss (0.2-0.5)
- `angle_threshold`: Filters out near-collision cases (10° recommended)

### Monitoring Training

Key metrics to watch:
- **SIR**: Should converge to ~25 dB
- **KL divergence**: Should stay low (< 0.1)
- **Clip loss**: Indicates policy changes
- **Ratio statistics**: Monitor in debug output

### Common Issues

**Problem**: Agent learns conservative policy (low SIR)
- **Solution**: Reduce penalty reward magnitude, increase entropy coefficient

**Problem**: Training unstable
- **Solution**: Reduce learning rate, decrease clip range

**Problem**: Slow convergence
- **Solution**: Increase buffer size, tune GAE lambda

## Performance Expectations

Based on paper results:
- **Average SIR**: ~25 dB
- **Convergence**: ~350,000 steps (~6 minutes wall-clock)
- **Inference time**: <1 ms per action
- **Speedup vs PSO/GA**: ~2.83×10^5 times faster

## Extending the Code

### Adding Multiple Interferers

```python
env = RADARDynamic(
    num_interferers=3,  # Change from 1 to 3
    # ... other params
)
```

### Changing Array Size

```python
env = RADARDynamic(
    num_elements=16,  # Change from 8 to 16
    # ... other params
)

# Agent action size will automatically adjust
```

### Custom Reward Function

Edit `Trainer.compute_reward()` in `train.py`:
```python
def compute_reward(self, eval_info: dict) -> float:
    # Your custom logic here
    return reward
```

## Citation

If you use this code, please cite:

```bibtex
@article{lin2025deep,
  title={Deep Reinforcement Learning-based Adaptive Nulling in Phased Array under Dynamic Environments},
  author={Lin, Ying-Dar and Chang, Jen-Hao and Lai, Yuan-Cheng},
  journal={IEEE Access},
  year={2025},
  publisher={IEEE}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- Original MATLAB implementation by the paper authors
- PyTorch team for the deep learning framework
- Gymnasium for the RL environment interface
