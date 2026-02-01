"""
Evaluation script for trained PPO agent
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from radar_env import RADARDynamic
from ppo_agent import PPOAgent
import os


def evaluate_agent(
    agent: PPOAgent,
    env: RADARDynamic,
    num_episodes: int = 10,
    visualize: bool = True,
    save_dir: str = './eval_results',
):
    """Evaluate trained agent"""
    agent.training = False
    
    episode_rewards = []
    episode_sirs = []
    episode_lengths = []
    
    print(f"\nEvaluating agent for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        truncated = False
        steps = 0
        
        episode_reward = 0.0
        sir_values = []
        
        # Track trajectory for visualization
        trajectory = {
            'desired_angles': [],
            'int_angles': [],
            'sirs': [],
            'pd_dbs': [],
            'pi_dbs': [],
        }
        
        while not (done or truncated) and steps < 500:
            # Get action (deterministic)
            action, _ = agent.select_action(state, deterministic=True)
            
            # Take step
            next_state, _, done, truncated, step_info = env.step(action)
            
            # Record
            episode_reward += step_info['SIR_dB']
            sir_values.append(step_info['SIR_dB'])
            
            trajectory['desired_angles'].append(step_info['theta_deg'])
            trajectory['int_angles'].append(step_info['int_angles_deg'][0])
            trajectory['sirs'].append(step_info['SIR_dB'])
            trajectory['pd_dbs'].append(step_info['PD_dB'])
            trajectory['pi_dbs'].append(step_info['PI_dB'])
            
            state = next_state
            steps += 1
        
        avg_sir = np.mean(sir_values) if sir_values else 0.0
        
        episode_rewards.append(episode_reward)
        episode_sirs.append(avg_sir)
        episode_lengths.append(steps)
        
        print(f"Episode {episode+1}: Reward={episode_reward:.2f}, "
              f"Avg SIR={avg_sir:.2f} dB, Steps={steps}")
        
        # Visualize first episode
        if episode == 0 and visualize:
            visualize_episode(trajectory, save_dir)
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean SIR: {np.mean(episode_sirs):.2f} ± {np.std(episode_sirs):.2f} dB")
    print(f"Mean length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Min SIR: {np.min(episode_sirs):.2f} dB")
    print(f"Max SIR: {np.max(episode_sirs):.2f} dB")
    print("=" * 60)
    
    return {
        'episode_rewards': episode_rewards,
        'episode_sirs': episode_sirs,
        'episode_lengths': episode_lengths,
    }


def visualize_episode(trajectory: dict, save_dir: str):
    """Visualize episode trajectory"""
    os.makedirs(save_dir, exist_ok=True)
    
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    steps = range(len(trajectory['sirs']))
    
    # Signal angles
    ax = axes[0, 0]
    ax.plot(steps, trajectory['desired_angles'], 'b-', linewidth=2, label='Desired')
    ax.plot(steps, trajectory['int_angles'], 'r-', linewidth=2, label='Interference')
    ax.set_xlabel('Step')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title('Signal Movement')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # SIR over time
    ax = axes[0, 1]
    ax.plot(steps, trajectory['sirs'], 'g-', linewidth=2)
    ax.axhline(np.mean(trajectory['sirs']), color='red', linestyle='--',
              label=f'Mean: {np.mean(trajectory["sirs"]):.2f} dB')
    ax.set_xlabel('Step')
    ax.set_ylabel('SIR (dB)')
    ax.set_title('Signal-to-Interference Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Power levels
    ax = axes[1, 0]
    ax.plot(steps, trajectory['pd_dbs'], 'b-', linewidth=2, label='PD (Desired)')
    ax.plot(steps, trajectory['pi_dbs'], 'r-', linewidth=2, label='PI (Interference)')
    ax.axhline(0, color='black', linestyle='--', alpha=0.5, label='0 dB')
    ax.axhline(-10, color='orange', linestyle='--', alpha=0.5, label='-10 dB threshold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Power (dB)')
    ax.set_title('Power Levels')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # SIR histogram
    ax = axes[1, 1]
    ax.hist(trajectory['sirs'], bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(trajectory['sirs']), color='red', linestyle='--',
              linewidth=2, label=f'Mean: {np.mean(trajectory["sirs"]):.2f} dB')
    ax.set_xlabel('SIR (dB)')
    ax.set_ylabel('Frequency')
    ax.set_title('SIR Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    fig_path = os.path.join(save_dir, 'episode_visualization.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Episode visualization saved to {fig_path}")
    
    plt.close()


def visualize_array_pattern(
    agent: PPOAgent,
    env: RADARDynamic,
    save_dir: str = './eval_results',
):
    """Visualize array factor pattern"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Reset environment
    state, info = env.reset()
    
    # Get action
    action, _ = agent.select_action(state, deterministic=True)
    
    # Evaluate
    eval_info = env.evaluate(action)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    angles = eval_info['angles']
    AF_dB = eval_info['AF_pattern_dB']
    theta_deg = eval_info['theta_deg']
    PD_dB = eval_info['PD_dB']
    int_angles = eval_info['int_angles_deg']
    PI_dB_single = eval_info['PI_dB_single']
    
    # Main pattern
    ax.plot(angles, AF_dB, 'b-', linewidth=2, label='Array Factor')
    
    # Mark desired direction
    ax.plot(theta_deg, PD_dB, 'go', markersize=10, linewidth=2,
           label=f'Desired ({theta_deg:.1f}°, {PD_dB:.2f} dB)')
    
    # Mark interference directions
    for q, (int_ang, pi_db) in enumerate(zip(int_angles, PI_dB_single)):
        ax.plot(int_ang, pi_db, 'rx', markersize=10, linewidth=2,
               label=f'Interf {q+1} ({int_ang:.1f}°, {pi_db:.2f} dB)')
    
    # Threshold lines
    ax.axhline(-10, color='red', linestyle='--', linewidth=1,
              label='Null Threshold (-10 dB)')
    ax.axhline(0, color='black', linestyle='--', linewidth=1,
              label='0 dB Reference')
    
    ax.set_xlabel('Angle (degrees)', fontsize=12)
    ax.set_ylabel('Array Gain (dB)', fontsize=12)
    ax.set_title(f'Array Factor Pattern (SIR = {eval_info["SIR_dB"]:.2f} dB)', fontsize=14)
    ax.set_xlim([-90, 90])
    ax.set_ylim([np.min(AF_dB) - 5, np.max(AF_dB) + 5])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    plt.tight_layout()
    
    # Save
    fig_path = os.path.join(save_dir, 'array_pattern.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Array pattern saved to {fig_path}")
    
    plt.close()


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained PPO agent')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--num_episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--save_dir', type=str, default='./eval_results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create environment
    env = RADARDynamic(
        num_elements=8,
        snr_db=5.0,
        isr_db=10.0,
        num_interferers=1,
        output_mode='triu_norm',
        max_steps=1000,
        desired_delta_deg=0.1,
        int_delta_deg=0.1,
    )
    
    # Create agent
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    
    agent = PPOAgent(
        state_size=state_size,
        action_size=action_size,
    )
    
    # Load trained model
    agent.load(args.model_path)
    
    # Evaluate
    results = evaluate_agent(
        agent=agent,
        env=env,
        num_episodes=args.num_episodes,
        visualize=True,
        save_dir=args.save_dir,
    )
    
    # Visualize array pattern
    visualize_array_pattern(
        agent=agent,
        env=env,
        save_dir=args.save_dir,
    )
    
    print(f"\nEvaluation completed! Results saved to {args.save_dir}")


if __name__ == '__main__':
    main()
