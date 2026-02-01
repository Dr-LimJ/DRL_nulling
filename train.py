"""
Training script for PPO-based beamforming
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
import json

from radar_env import RADARDynamic
from ppo_agent import PPOAgent


class Trainer:
    """Training manager for PPO beamforming"""
    
    def __init__(
        self,
        env: RADARDynamic,
        agent: PPOAgent,
        angle_threshold: float = 10.0,
        penalty_reward: float = -10.0,
        pd_threshold: float = -5.0,
        eta_rl: float = -15.0,
        save_dir: str = './results',
    ):
        self.env = env
        self.agent = agent
        self.angle_threshold = angle_threshold
        self.penalty_reward = penalty_reward
        self.pd_threshold = pd_threshold
        self.eta_rl = eta_rl
        self.current_update_reward = 0.0
        self.current_update_sirs = []

        # Create save directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_dir = os.path.join(save_dir, f'ppo_training_{timestamp}')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Tracking variables
        self.episode_rewards = []
        self.episode_sirs = []
        self.update_rewards = []
        self.update_sirs = []
        
        # Statistics
        self.total_steps = 0
        self.skipped_steps = 0
        self.stored_steps = 0
        
        print(f"\nTrainer initialized")
        print(f"Save directory: {self.save_dir}")
        print(f"Angle threshold: {angle_threshold}°")
        print(f"Penalty reward: {penalty_reward}")
        print(f"PD threshold: {pd_threshold} dB")
        print(f"PI threshold (eta_rl): {eta_rl} dB\n")
    
    def compute_reward(self, eval_info: dict) -> float:
        """Compute reward based on evaluation info"""
        PD_dB = eval_info['PD_dB']
        PI_dB = eval_info['PI_dB']
        
        # Check constraints
        # if PD_dB <= self.pd_threshold or PI_dB >= self.eta_rl:
        #     return self.penalty_reward
        # else:
        #     return eval_info['SIR_dB']

        return eval_info['SIR_dB']

    
    def check_angle_threshold(self, state_info: dict) -> bool:
        """Check if angle difference meets threshold"""
        desired_angle = state_info['desired_angle']
        int_angles = state_info['int_angles_deg']
        
        min_angle_diff = np.min(np.abs(desired_angle - int_angles))
        return min_angle_diff >= self.angle_threshold
    
    def train_episode(self) -> dict:
        """Train for one episode"""

        options = {
            'int_angles_deg': [0.0],
            'desired_angle': 20.0
        }
        
        state, info = self.env.reset(options=options)
        done = False
        truncated = False
        steps = 0
        
        episode_reward = 0.0
        sir_values = []
        episode_skipped = 0
        episode_stored = 0
        
        # Current update tracking
        # current_update_reward = 0.0
        # current_update_sirs = []
        
        while not (done or truncated) and steps < 500:
            # Check angle threshold
            current_angle_diff = np.min(
                np.abs(info['desired_angle'] - info['int_angles_deg'])
            )
            
            if current_angle_diff >= self.angle_threshold:
                # Get action
                action, action_info = self.agent.select_action(state)
                
                # Take step
                next_state, _, done, truncated, step_info = self.env.step(action)
                
                # Compute reward
                reward = self.compute_reward(step_info)
                
                # Store transition
                self.agent.store_transition(
                    state, action, reward, next_state, done, action_info
                )
                
                # Update tracking
                episode_reward += reward
                sir_values.append(step_info['SIR_dB'])
                # current_update_reward += reward
                # current_update_sirs.append(step_info['SIR_dB'])
                self.current_update_reward += reward
                self.current_update_sirs.append(step_info['SIR_dB'])
                
                episode_stored += 1
                self.stored_steps += 1
                
                # Check if ready to update
                if self.agent.ready_to_update():
                    actor_loss, critic_loss = self.agent.update()
                    
                    # Store update metrics
                    self.update_rewards.append(self.current_update_reward)
                    self.update_sirs.append(np.mean(self.current_update_sirs))
                    self.current_update_reward = 0.0
                    self.current_update_sirs = []
                
                # Update state
                state = next_state
                info = step_info
            else:
                # Skip but continue episode
                next_state, _, done, truncated, step_info = self.env.step(
                    np.zeros(self.env.action_space.shape)  # Dummy action
                )
                state = next_state
                info = step_info
                episode_skipped += 1
                self.skipped_steps += 1
            
            steps += 1
            self.total_steps += 1
        
        # Episode statistics
        avg_sir = np.mean(sir_values) if sir_values else 0.0
        
        self.episode_rewards.append(episode_reward)
        self.episode_sirs.append(avg_sir)
        self.agent.episode_end(episode_reward, steps, avg_sir)
        
        return {
            'episode_reward': episode_reward,
            'avg_sir': avg_sir,
            'steps': steps,
            'stored': episode_stored,
            'skipped': episode_skipped,
        }
    
    def train(self, num_episodes: int = 3000, log_interval: int = 100):
        """Train for multiple episodes"""
        print("=" * 60)
        print(f"Starting training for {num_episodes} episodes")
        print("=" * 60)
        
        pbar = tqdm(range(num_episodes), desc="Training")
        
        for episode in pbar:
            episode_stats = self.train_episode()
            
            # Update progress bar
            if (episode + 1) % log_interval == 0:
                recent_rewards = self.episode_rewards[-log_interval:]
                recent_sirs = self.episode_sirs[-log_interval:]
                
                pbar.set_postfix({
                    'Avg Reward': f"{np.mean(recent_rewards):.2f}",
                    'Avg SIR': f"{np.mean(recent_sirs):.2f} dB",
                    'Updates': self.agent.update_count,
                })
                
                # Periodic logging
                print(f"\n[Episode {episode+1}/{num_episodes}]")
                print(f"  Avg Reward (last {log_interval}): {np.mean(recent_rewards):.2f}")
                print(f"  Avg SIR (last {log_interval}): {np.mean(recent_sirs):.2f} dB")
                print(f"  Total Updates: {self.agent.update_count}")
                print(f"  Stored/Skipped: {self.stored_steps}/{self.skipped_steps}")
        
        print("\n" + "=" * 60)
        print("Training completed!")
        print("=" * 60)
        
        # Save results
        self.save_results()
        self.plot_results()
    
    def save_results(self):
        """Save training results"""
        print("\nSaving results...")
        
        # Save agent
        agent_path = os.path.join(self.save_dir, 'trained_agent.pth')
        self.agent.save(agent_path)
        
        # Save metrics
        metrics = {
            'episode_rewards': self.episode_rewards,
            'episode_sirs': self.episode_sirs,
            'update_rewards': self.update_rewards,
            'update_sirs': self.update_sirs,
            'actor_losses': self.agent.actor_losses,
            'critic_losses': self.agent.critic_losses,
            'kl_history': self.agent.kl_history,
            'total_steps': self.total_steps,
            'stored_steps': self.stored_steps,
            'skipped_steps': self.skipped_steps,
            'update_count': self.agent.update_count,
        }
        
        metrics_path = os.path.join(self.save_dir, 'metrics.npz')
        np.savez(metrics_path, **metrics)
        
        # Save configuration
        config = {
            'num_elements': self.env.num_elements,
            'num_interferers': self.env.num_interferers,
            'angle_threshold': self.angle_threshold,
            'penalty_reward': self.penalty_reward,
            'pd_threshold': self.pd_threshold,
            'eta_rl': self.eta_rl,
            'gamma': self.agent.gamma,
            'lambda_gae': self.agent.lambda_gae,
            'clip_range': self.agent.clip_range,
            'learning_rate': self.agent.learning_rate,
            'buffer_size': self.agent.buffer_size,
            'batch_size': self.agent.batch_size,
        }
        
        config_path = os.path.join(self.save_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        # Save summary
        summary_path = os.path.join(self.save_dir, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("Training Summary\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("Configuration:\n")
            for key, value in config.items():
                f.write(f"  {key}: {value}\n")
            
            f.write(f"\nTraining Statistics:\n")
            f.write(f"  Total episodes: {len(self.episode_rewards)}\n")
            f.write(f"  Total steps: {self.total_steps}\n")
            f.write(f"  Stored steps: {self.stored_steps} ({100*self.stored_steps/self.total_steps:.1f}%)\n")
            f.write(f"  Skipped steps: {self.skipped_steps} ({100*self.skipped_steps/self.total_steps:.1f}%)\n")
            f.write(f"  Total updates: {self.agent.update_count}\n")
            
            f.write(f"\nPerformance Metrics:\n")
            f.write(f"  Final avg reward (last 100): {np.mean(self.episode_rewards[-100:]):.2f}\n")
            f.write(f"  Final avg SIR (last 100): {np.mean(self.episode_sirs[-100:]):.2f} dB\n")
            f.write(f"  Best episode SIR: {np.max(self.episode_sirs):.2f} dB\n")
            
            if len(self.update_rewards) > 0:
                f.write(f"  Final avg update reward (last 10): {np.mean(self.update_rewards[-10:]):.2f}\n")
                f.write(f"  Final avg update SIR (last 10): {np.mean(self.update_sirs[-10:]):.2f} dB\n")
        
        print(f"✓ Results saved to {self.save_dir}")
    
    def plot_results(self):
        """Plot training results"""
        print("\nGenerating plots...")
        
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Episode rewards
        ax = axes[0, 0]
        ax.plot(self.episode_rewards, alpha=0.3, linewidth=0.5)
        if len(self.episode_rewards) > 100:
            window = 100
            moving_avg = np.convolve(
                self.episode_rewards, 
                np.ones(window)/window, 
                mode='valid'
            )
            ax.plot(range(window-1, len(self.episode_rewards)), moving_avg, 
                   linewidth=2, label=f'MA({window})')
            ax.legend()
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Episode Rewards')
        
        # Episode SIRs
        ax = axes[0, 1]
        ax.plot(self.episode_sirs, alpha=0.3, linewidth=0.5)
        if len(self.episode_sirs) > 100:
            window = 100
            moving_avg = np.convolve(
                self.episode_sirs, 
                np.ones(window)/window, 
                mode='valid'
            )
            ax.plot(range(window-1, len(self.episode_sirs)), moving_avg, 
                   linewidth=2, label=f'MA({window})')
            ax.legend()
        ax.set_xlabel('Episode')
        ax.set_ylabel('SIR (dB)')
        ax.set_title('Episode SIRs')
        
        # Update rewards
        ax = axes[0, 2]
        if len(self.update_rewards) > 0:
            ax.plot(self.update_rewards, linewidth=1.5)
            ax.set_xlabel('Update')
            ax.set_ylabel('Total Reward')
            ax.set_title('Update Rewards (Stored Transitions)')
        
        # Actor losses
        ax = axes[1, 0]
        if len(self.agent.actor_losses) > 0:
            ax.plot(self.agent.actor_losses, linewidth=1.5, color='red')
            ax.set_xlabel('Update')
            ax.set_ylabel('Loss')
            ax.set_title('Actor Loss')
        
        # Critic losses
        ax = axes[1, 1]
        if len(self.agent.critic_losses) > 0:
            ax.plot(self.agent.critic_losses, linewidth=1.5, color='blue')
            ax.set_xlabel('Update')
            ax.set_ylabel('Loss')
            ax.set_title('Critic Loss')
        
        # Update SIRs
        ax = axes[1, 2]
        if len(self.update_sirs) > 0:
            ax.plot(self.update_sirs, linewidth=1.5, color='green')
            ax.axhline(np.mean(self.update_sirs), color='red', linestyle='--',
                      label=f'Mean: {np.mean(self.update_sirs):.2f} dB')
            ax.set_xlabel('Update')
            ax.set_ylabel('Avg SIR (dB)')
            ax.set_title('Update SIRs (Stored Transitions)')
            ax.legend()
        
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(self.save_dir, 'training_curves.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training curves saved to {fig_path}")
        
        plt.close()


def main():
    """Main training function"""
    # Set random seeds
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Create environment
    env = RADARDynamic(
        num_elements=8,
        snr_db=5.0,
        isr_db=0.0,
        num_interferers=1,
        output_mode='triu_norm',
        max_steps=1024,
        desired_delta_deg=0.0,
        int_delta_deg=0.0,
    )
    
    # Create agent
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    
    agent = PPOAgent(
        state_size=state_size,
        action_size=action_size,
        gamma=0.99,
        lambda_gae=0.95,
        clip_range=0.3,
        value_coeff=0.2,
        entropy_coeff=0.001,
        learning_rate=3e-4,
        buffer_size=2048,
        batch_size=128,
        n_epochs=10,
        debug_verbose=10,
    )
    
    # Create trainer
    trainer = Trainer(
        env=env,
        agent=agent,
        angle_threshold=10.0,
        penalty_reward=-10.0,
        eta_rl=-15.0,
        pd_threshold=-5.0
    )
    
    # Train
    trainer.train(num_episodes=500, log_interval=3)
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print(f"Results saved to: {trainer.save_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
