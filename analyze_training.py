"""
Training Analysis Script for PPO Beamforming
Analyzes saved training data from train.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import argparse
from typing import Dict, List, Tuple


class TrainingAnalyzer:
    """Analyze training results from train.py"""
    
    def __init__(self, results_dir: str):
        """
        Initialize analyzer with results directory
        
        Args:
            results_dir: Path to training results directory (e.g., ./results/ppo_training_20240101_120000)
        """
        self.results_dir = Path(results_dir)
        
        if not self.results_dir.exists():
            raise ValueError(f"Results directory not found: {results_dir}")
        
        # Load data
        self.metrics = self._load_metrics()
        self.config = self._load_config()
        
        print(f"\n{'='*60}")
        print(f"Training Analyzer Initialized")
        print(f"{'='*60}")
        print(f"Results directory: {self.results_dir}")
        print(f"Total episodes: {len(self.metrics['episode_rewards'])}")
        print(f"Total updates: {self.metrics['update_count']}")
        print(f"{'='*60}\n")
    
    def _load_metrics(self) -> Dict:
        """Load metrics from npz file"""
        metrics_path = self.results_dir / 'metrics.npz'
        
        if not metrics_path.exists():
            raise ValueError(f"Metrics file not found: {metrics_path}")
        
        data = np.load(metrics_path, allow_pickle=True)
        
        # Convert to regular dict
        metrics = {}
        for key in data.files:
            metrics[key] = data[key]
            if metrics[key].ndim == 0:  # Scalar values
                metrics[key] = metrics[key].item()
        
        return metrics
    
    def _load_config(self) -> Dict:
        """Load configuration from json file"""
        config_path = self.results_dir / 'config.json'
        
        if not config_path.exists():
            print(f"Warning: Config file not found: {config_path}")
            return {}
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return config
    
    def print_summary(self):
        """Print training summary statistics"""
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        
        # Configuration
        print("\n[Configuration]")
        for key, value in self.config.items():
            print(f"  {key:20s}: {value}")
        
        # Training statistics
        print("\n[Training Statistics]")
        print(f"  {'Total episodes':<25s}: {len(self.metrics['episode_rewards'])}")
        print(f"  {'Total steps':<25s}: {self.metrics['total_steps']}")
        print(f"  {'Stored steps':<25s}: {self.metrics['stored_steps']} "
              f"({100*self.metrics['stored_steps']/self.metrics['total_steps']:.1f}%)")
        print(f"  {'Skipped steps':<25s}: {self.metrics['skipped_steps']} "
              f"({100*self.metrics['skipped_steps']/self.metrics['total_steps']:.1f}%)")
        print(f"  {'Total updates':<25s}: {self.metrics['update_count']}")
        
        # Performance metrics
        episode_rewards = self.metrics['episode_rewards']
        episode_sirs = self.metrics['episode_sirs']
        
        print("\n[Episode Performance]")
        print(f"  {'Mean reward (all)':<25s}: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"  {'Mean reward (last 100)':<25s}: {np.mean(episode_rewards[-100:]):.2f}")
        print(f"  {'Mean SIR (all)':<25s}: {np.mean(episode_sirs):.2f} ± {np.std(episode_sirs):.2f} dB")
        print(f"  {'Mean SIR (last 100)':<25s}: {np.mean(episode_sirs[-100:]):.2f} dB")
        print(f"  {'Best episode reward':<25s}: {np.max(episode_rewards):.2f}")
        print(f"  {'Best episode SIR':<25s}: {np.max(episode_sirs):.2f} dB")
        print(f"  {'Worst episode SIR':<25s}: {np.min(episode_sirs):.2f} dB")
        
        # Update metrics (only stored transitions)
        if len(self.metrics['update_rewards']) > 0:
            update_rewards = self.metrics['update_rewards']
            update_sirs = self.metrics['update_sirs']
            
            print("\n[Update Performance (Stored Transitions)]")
            print(f"  {'Mean update reward':<25s}: {np.mean(update_rewards):.2f} ± {np.std(update_rewards):.2f}")
            print(f"  {'Mean update reward (last 10)':<25s}: {np.mean(update_rewards[-10:]):.2f}")
            print(f"  {'Mean update SIR':<25s}: {np.mean(update_sirs):.2f} ± {np.std(update_sirs):.2f} dB")
            print(f"  {'Mean update SIR (last 10)':<25s}: {np.mean(update_sirs[-10:]):.2f} dB")
        
        # Loss metrics
        if len(self.metrics['actor_losses']) > 0:
            actor_losses = self.metrics['actor_losses']
            critic_losses = self.metrics['critic_losses']
            
            print("\n[Loss Metrics]")
            print(f"  {'Mean actor loss':<25s}: {np.mean(actor_losses):.4f}")
            print(f"  {'Final actor loss (last 10)':<25s}: {np.mean(actor_losses[-10:]):.4f}")
            print(f"  {'Mean critic loss':<25s}: {np.mean(critic_losses):.4f}")
            print(f"  {'Final critic loss (last 10)':<25s}: {np.mean(critic_losses[-10:]):.4f}")
        
        # KL divergence
        if len(self.metrics['kl_history']) > 0:
            kl_history = self.metrics['kl_history']
            print(f"  {'Mean KL divergence':<25s}: {np.mean(kl_history):.6f}")
            print(f"  {'Final KL div (last 10)':<25s}: {np.mean(kl_history[-10:]):.6f}")
        
        print("\n" + "="*60)
    
    def plot_comprehensive_analysis(self, save_path: str = None):
        """Create comprehensive training analysis plots"""
        sns.set_style("whitegrid")
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Episode Rewards
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_episode_rewards(ax1)
        
        # 2. Episode SIRs
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_episode_sirs(ax2)
        
        # 3. Update Rewards
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_update_rewards(ax3)
        
        # 4. Update SIRs
        ax4 = fig.add_subplot(gs[0, 3])
        self._plot_update_sirs(ax4)
        
        # 5. Actor Loss
        ax5 = fig.add_subplot(gs[1, 0])
        self._plot_actor_loss(ax5)
        
        # 6. Critic Loss
        ax6 = fig.add_subplot(gs[1, 1])
        self._plot_critic_loss(ax6)
        
        # 7. KL Divergence
        ax7 = fig.add_subplot(gs[1, 2])
        self._plot_kl_divergence(ax7)
        
        # 8. Training Efficiency
        ax8 = fig.add_subplot(gs[1, 3])
        self._plot_training_efficiency(ax8)
        
        # 9. Reward Distribution
        ax9 = fig.add_subplot(gs[2, 0])
        self._plot_reward_distribution(ax9)
        
        # 10. SIR Distribution
        ax10 = fig.add_subplot(gs[2, 1])
        self._plot_sir_distribution(ax10)
        
        # 11. Rolling Statistics
        ax11 = fig.add_subplot(gs[2, 2])
        self._plot_rolling_statistics(ax11)
        
        # 12. Learning Progress
        ax12 = fig.add_subplot(gs[2, 3])
        self._plot_learning_progress(ax12)
        
        # Save
        if save_path is None:
            save_path = self.results_dir / 'comprehensive_analysis.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comprehensive analysis saved to {save_path}")
        plt.close()
    
    def _plot_episode_rewards(self, ax):
        """Plot episode rewards with moving average"""
        rewards = self.metrics['episode_rewards']
        episodes = range(len(rewards))
        
        ax.plot(episodes, rewards, alpha=0.3, linewidth=0.5, color='blue')
        
        # Moving average
        if len(rewards) > 50:
            window = min(50, len(rewards) // 10)
            ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(rewards)), ma, linewidth=2, 
                   color='darkblue', label=f'MA({window})')
            ax.legend()
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Episode Rewards')
        ax.grid(True, alpha=0.3)
    
    def _plot_episode_sirs(self, ax):
        """Plot episode average SIRs"""
        sirs = self.metrics['episode_sirs']
        episodes = range(len(sirs))
        
        ax.plot(episodes, sirs, alpha=0.3, linewidth=0.5, color='green')
        
        # Moving average
        if len(sirs) > 50:
            window = min(50, len(sirs) // 10)
            ma = np.convolve(sirs, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(sirs)), ma, linewidth=2, 
                   color='darkgreen', label=f'MA({window})')
            ax.legend()
        
        ax.axhline(np.mean(sirs), color='red', linestyle='--', alpha=0.5,
                  label=f'Mean: {np.mean(sirs):.2f} dB')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average SIR (dB)')
        ax.set_title('Episode SIR Performance')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_update_rewards(self, ax):
        """Plot update rewards"""
        if len(self.metrics['update_rewards']) > 0:
            rewards = self.metrics['update_rewards']
            updates = range(len(rewards))
            
            ax.plot(updates, rewards, linewidth=1.5, color='purple')
            ax.axhline(np.mean(rewards), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(rewards):.2f}')
            
            ax.set_xlabel('Update')
            ax.set_ylabel('Total Reward')
            ax.set_title('Update Rewards (Stored Transitions)')
            ax.grid(True, alpha=0.3)
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No update data', ha='center', va='center',
                   transform=ax.transAxes)
    
    def _plot_update_sirs(self, ax):
        """Plot update SIRs"""
        if len(self.metrics['update_sirs']) > 0:
            sirs = self.metrics['update_sirs']
            updates = range(len(sirs))
            
            ax.plot(updates, sirs, linewidth=1.5, color='orange')
            ax.axhline(np.mean(sirs), color='red', linestyle='--',
                      label=f'Mean: {np.mean(sirs):.2f} dB')
            
            ax.set_xlabel('Update')
            ax.set_ylabel('Average SIR (dB)')
            ax.set_title('Update SIR (Stored Transitions)')
            ax.grid(True, alpha=0.3)
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No update data', ha='center', va='center',
                   transform=ax.transAxes)
    
    def _plot_actor_loss(self, ax):
        """Plot actor loss"""
        if len(self.metrics['actor_losses']) > 0:
            losses = self.metrics['actor_losses']
            updates = range(len(losses))
            
            ax.plot(updates, losses, linewidth=1.5, color='red', alpha=0.7)
            
            # Moving average
            if len(losses) > 20:
                window = min(20, len(losses) // 5)
                ma = np.convolve(losses, np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(losses)), ma, linewidth=2,
                       color='darkred', label=f'MA({window})')
                ax.legend()
            
            ax.set_xlabel('Update')
            ax.set_ylabel('Loss')
            ax.set_title('Actor Loss')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No loss data', ha='center', va='center',
                   transform=ax.transAxes)
    
    def _plot_critic_loss(self, ax):
        """Plot critic loss"""
        if len(self.metrics['critic_losses']) > 0:
            losses = self.metrics['critic_losses']
            updates = range(len(losses))
            
            ax.plot(updates, losses, linewidth=1.5, color='blue', alpha=0.7)
            
            # Moving average
            if len(losses) > 20:
                window = min(20, len(losses) // 5)
                ma = np.convolve(losses, np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(losses)), ma, linewidth=2,
                       color='darkblue', label=f'MA({window})')
                ax.legend()
            
            ax.set_xlabel('Update')
            ax.set_ylabel('Loss')
            ax.set_title('Critic Loss')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No loss data', ha='center', va='center',
                   transform=ax.transAxes)
    
    def _plot_kl_divergence(self, ax):
        """Plot KL divergence"""
        if len(self.metrics['kl_history']) > 0:
            kl = self.metrics['kl_history']
            updates = range(len(kl))
            
            ax.plot(updates, kl, linewidth=1.5, color='green')
            ax.axhline(np.mean(kl), color='red', linestyle='--',
                      label=f'Mean: {np.mean(kl):.4f}')
            
            ax.set_xlabel('Update')
            ax.set_ylabel('KL Divergence')
            ax.set_title('KL Divergence (Old vs New Policy)')
            ax.grid(True, alpha=0.3)
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No KL data', ha='center', va='center',
                   transform=ax.transAxes)
    
    def _plot_training_efficiency(self, ax):
        """Plot training efficiency (stored vs skipped steps)"""
        stored = self.metrics['stored_steps']
        skipped = self.metrics['skipped_steps']
        total = self.metrics['total_steps']
        
        labels = ['Stored\n(Used for Training)', 'Skipped\n(Angle Threshold)']
        sizes = [stored, skipped]
        colors = ['#66b3ff', '#ff9999']
        explode = (0.05, 0.05)
        
        ax.pie(sizes, explode=explode, labels=labels, colors=colors,
               autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
        ax.set_title(f'Training Efficiency\n(Total: {total:,} steps)')
    
    def _plot_reward_distribution(self, ax):
        """Plot reward distribution across episodes"""
        rewards = self.metrics['episode_rewards']
        
        ax.hist(rewards, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        ax.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {np.mean(rewards):.2f}')
        ax.axvline(np.median(rewards), color='green', linestyle='--', linewidth=2,
                  label=f'Median: {np.median(rewards):.2f}')
        
        ax.set_xlabel('Episode Reward')
        ax.set_ylabel('Frequency')
        ax.set_title('Reward Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_sir_distribution(self, ax):
        """Plot SIR distribution across episodes"""
        sirs = self.metrics['episode_sirs']
        
        ax.hist(sirs, bins=50, edgecolor='black', alpha=0.7, color='lightgreen')
        ax.axvline(np.mean(sirs), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {np.mean(sirs):.2f} dB')
        ax.axvline(np.median(sirs), color='blue', linestyle='--', linewidth=2,
                  label=f'Median: {np.median(sirs):.2f} dB')
        
        ax.set_xlabel('Average SIR (dB)')
        ax.set_ylabel('Frequency')
        ax.set_title('SIR Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_rolling_statistics(self, ax):
        """Plot rolling mean and std of rewards"""
        rewards = self.metrics['episode_rewards']
        window = min(50, len(rewards) // 5)
        
        if len(rewards) > window:
            rolling_mean = np.convolve(rewards, np.ones(window)/window, mode='valid')
            rolling_std = np.array([
                np.std(rewards[max(0, i-window):i+1]) 
                for i in range(window-1, len(rewards))
            ])
            
            episodes = range(window-1, len(rewards))
            
            ax.plot(episodes, rolling_mean, linewidth=2, color='blue', label='Mean')
            ax.fill_between(episodes, 
                           rolling_mean - rolling_std,
                           rolling_mean + rolling_std,
                           alpha=0.3, color='blue', label='±1 Std')
            
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            ax.set_title(f'Rolling Statistics (window={window})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                   transform=ax.transAxes)
    
    def _plot_learning_progress(self, ax):
        """Plot learning progress over time"""
        sirs = self.metrics['episode_sirs']
        
        # Divide into quartiles
        n = len(sirs)
        q1 = sirs[:n//4]
        q2 = sirs[n//4:n//2]
        q3 = sirs[n//2:3*n//4]
        q4 = sirs[3*n//4:]
        
        data = [q1, q2, q3, q4]
        labels = ['Q1\n(Early)', 'Q2', 'Q3', 'Q4\n(Late)']
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                        notch=True, showmeans=True)
        
        # Color boxes
        colors = ['#ff9999', '#ffcc99', '#99ccff', '#99ff99']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel('SIR (dB)')
        ax.set_title('Learning Progress (Quartile Comparison)')
        ax.grid(True, alpha=0.3, axis='y')
    
    def compare_phases(self, phase_sizes: List[int] = None):
        """Compare performance across training phases"""
        if phase_sizes is None:
            # Default: divide into 4 equal phases
            n = len(self.metrics['episode_rewards'])
            phase_sizes = [n // 4] * 4
        
        rewards = self.metrics['episode_rewards']
        sirs = self.metrics['episode_sirs']
        
        print("\n" + "="*60)
        print("PHASE COMPARISON")
        print("="*60)
        
        start = 0
        for i, size in enumerate(phase_sizes):
            end = min(start + size, len(rewards))
            phase_rewards = rewards[start:end]
            phase_sirs = sirs[start:end]
            
            print(f"\nPhase {i+1} (Episodes {start+1}-{end}):")
            print(f"  Mean reward: {np.mean(phase_rewards):.2f} ± {np.std(phase_rewards):.2f}")
            print(f"  Mean SIR:    {np.mean(phase_sirs):.2f} ± {np.std(phase_sirs):.2f} dB")
            print(f"  Max SIR:     {np.max(phase_sirs):.2f} dB")
            print(f"  Min SIR:     {np.min(phase_sirs):.2f} dB")
            
            start = end
        
        print("\n" + "="*60)
    
    def analyze_convergence(self, window: int = 100):
        """Analyze convergence behavior"""
        sirs = self.metrics['episode_sirs']
        
        if len(sirs) < window * 2:
            print("Insufficient data for convergence analysis")
            return
        
        print("\n" + "="*60)
        print("CONVERGENCE ANALYSIS")
        print("="*60)
        
        # Calculate moving average
        ma = np.convolve(sirs, np.ones(window)/window, mode='valid')
        
        # Find trend (linear regression on last half)
        second_half = ma[len(ma)//2:]
        x = np.arange(len(second_half))
        coeffs = np.polyfit(x, second_half, 1)
        slope = coeffs[0]
        
        print(f"\nWindow size: {window} episodes")
        print(f"Trend (last half): {slope:.4f} dB/episode")
        
        if abs(slope) < 0.01:
            print("Status: ✓ CONVERGED (slope < 0.01 dB/episode)")
        elif slope > 0:
            print("Status: ↗ IMPROVING")
        else:
            print("Status: ↘ DEGRADING")
        
        # Variance analysis
        first_quarter_var = np.var(sirs[:len(sirs)//4])
        last_quarter_var = np.var(sirs[3*len(sirs)//4:])
        
        print(f"\nVariance analysis:")
        print(f"  First quarter: {first_quarter_var:.2f}")
        print(f"  Last quarter:  {last_quarter_var:.2f}")
        print(f"  Reduction:     {100*(first_quarter_var - last_quarter_var)/first_quarter_var:.1f}%")
        
        print("\n" + "="*60)
    
    def export_data(self, export_dir: str = None):
        """Export processed data for further analysis"""
        if export_dir is None:
            export_dir = self.results_dir / 'exported_data'
        
        export_dir = Path(export_dir)
        export_dir.mkdir(exist_ok=True)
        
        # Export episode data
        episode_data = {
            'episode': np.arange(len(self.metrics['episode_rewards'])),
            'reward': self.metrics['episode_rewards'],
            'sir_db': self.metrics['episode_sirs'],
        }
        np.savez(export_dir / 'episode_data.npz', **episode_data)
        
        # Export update data
        if len(self.metrics['update_rewards']) > 0:
            update_data = {
                'update': np.arange(len(self.metrics['update_rewards'])),
                'reward': self.metrics['update_rewards'],
                'sir_db': self.metrics['update_sirs'],
                'actor_loss': self.metrics['actor_losses'],
                'critic_loss': self.metrics['critic_losses'],
                'kl_div': self.metrics['kl_history'],
            }
            np.savez(export_dir / 'update_data.npz', **update_data)
        
        # Export summary statistics
        summary = {
            'config': self.config,
            'episode_stats': {
                'mean_reward': float(np.mean(self.metrics['episode_rewards'])),
                'std_reward': float(np.std(self.metrics['episode_rewards'])),
                'mean_sir': float(np.mean(self.metrics['episode_sirs'])),
                'std_sir': float(np.std(self.metrics['episode_sirs'])),
                'max_sir': float(np.max(self.metrics['episode_sirs'])),
                'min_sir': float(np.min(self.metrics['episode_sirs'])),
            },
            'training_stats': {
                'total_steps': int(self.metrics['total_steps']),
                'stored_steps': int(self.metrics['stored_steps']),
                'skipped_steps': int(self.metrics['skipped_steps']),
                'update_count': int(self.metrics['update_count']),
            }
        }
        
        with open(export_dir / 'summary_stats.json', 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"✓ Data exported to {export_dir}")


def main():
    """Main analysis function"""
    parser = argparse.ArgumentParser(
        description='Analyze PPO training results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze specific training run
  python analyze_training.py --dir ./results/ppo_training_20240101_120000
  
  # Analyze latest training run
  python analyze_training.py --latest
  
  # Generate all plots and export data
  python analyze_training.py --latest --export
        """
    )
    
    parser.add_argument('--dir', type=str, default=None,
                       help='Path to training results directory')
    parser.add_argument('--latest', action='store_true',
                       help='Use the latest training results')
    parser.add_argument('--export', action='store_true',
                       help='Export processed data')
    parser.add_argument('--convergence', action='store_true',
                       help='Perform convergence analysis')
    parser.add_argument('--phases', type=int, default=4,
                       help='Number of phases for comparison (default: 4)')
    
    args = parser.parse_args(['--latest', '--export']) # example: args = parser.parse_args(['--dir', './results/...', '--export'])

    # Determine results directory
    if args.latest:
        results_base = Path('./results')
        if not results_base.exists():
            print("Error: ./results directory not found")
            return
        
        # Find latest training directory
        training_dirs = sorted(results_base.glob('ppo_training_*'))
        if not training_dirs:
            print("Error: No training results found in ./results")
            return
        
        results_dir = training_dirs[-1]
        print(f"Using latest training: {results_dir.name}")
    elif args.dir:
        results_dir = args.dir
    else:
        print("Error: Must specify --dir or --latest")
        parser.print_help()
        return
    
    # Create analyzer
    try:
        analyzer = TrainingAnalyzer(results_dir)
    except Exception as e:
        print(f"Error loading results: {e}")
        return
    
    # Print summary
    analyzer.print_summary()
    
    # Phase comparison
    analyzer.compare_phases()
    
    # Convergence analysis
    if args.convergence:
        analyzer.analyze_convergence()
    
    # Generate plots
    print("\nGenerating comprehensive analysis plots...")
    analyzer.plot_comprehensive_analysis()
    
    # Export data
    if args.export:
        print("\nExporting data...")
        analyzer.export_data()
    
    print("\n" + "="*60)
    print("Analysis completed!")
    print("="*60)


if __name__ == '__main__':
    main()