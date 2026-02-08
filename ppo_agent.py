"""
PPO Agent with TD(0) Critic Loss
Implementation based on paper Figure 5 architecture
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List, Dict
from collections import deque


class ActorNetwork(nn.Module):
    """Actor network with 6 layers (paper Figure 5)"""
    
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        
        # Paper architecture: 6 layers
        self.shared = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.01),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.01),
        )
        
        # Output: mean and log_std for Gaussian policy
        self.mean_head = nn.Linear(16, action_size)
        self.log_std_head = nn.Linear(16, action_size)
        
        # Initialize weights (small values for stability)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and log_std"""
        features = self.shared(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        
        # Clamp log_std for stability
        # log_std = torch.clamp(log_std, min=-5.0, max=-0.5)
        
        return mean, log_std


class CriticNetwork(nn.Module):
    """Critic network with 5 layers (paper Figure 5)"""
    
    def __init__(self, state_size: int):
        super().__init__()
        
        # Paper architecture: 5 layers
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.01),
            nn.Linear(32, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass returning state value"""
        return self.network(state)


class PPOAgent:
    """PPO Agent with TD(0) critic loss"""
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        # PPO hyperparameters
        gamma: float = 0.99,
        lambda_gae: float = 0.95,
        clip_range: float = 0.2,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.01,
        learning_rate: float = 3e-4,
        # Training parameters
        batch_size: int = 128,
        n_epochs: int = 10,
        buffer_size: int = 2048,
        debug_verbose: int = 10,
    ):
        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.clip_range = clip_range
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.learning_rate = learning_rate
        
        # Training parameters
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.buffer_size = buffer_size
        self.debug_verbose = debug_verbose
        
        # Networks
        self.actor = ActorNetwork(state_size, action_size).to(device)
        self.critic = CriticNetwork(state_size).to(device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3*learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3*learning_rate)
        
        # Experience buffer
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'values': [],
            'next_values': [],
            'log_probs': [],
        }
        
        # Statistics
        self.total_steps = 0
        self.episode_count = 0
        self.update_count = 0
        
        # Loss history
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_history = []
        self.kl_history = []
        
        # Episode metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_sirs = []
        
        # Training mode
        self.training = True
        
        print("=" * 50)
        print("PPO Agent Initialized")
        print("=" * 50)
        print(f"Device: {device}")
        print(f"State size: {state_size}")
        print(f"Action size: {action_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Clip range: {clip_range}")
        print(f"Buffer size: {buffer_size}")
        print(f"Batch size: {batch_size}")
        print("=" * 50)
    
    def select_action(
        self, 
        state: np.ndarray, 
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Dict]:
        """Select action using actor network"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mean, log_std = self.actor(state_tensor)
            std = torch.exp(log_std)
            
            if self.training and not deterministic:
                # Sample from Gaussian
                normal = torch.distributions.Normal(mean, std)
                action_tensor = normal.sample()
            else:
                # Deterministic (mean)
                action_tensor = mean
            
            action = action_tensor.cpu().numpy()[0]
            
            # Compute log probability
            log_prob = self._compute_log_prob(action_tensor, mean, log_std)
            
            # Compute entropy
            entropy = self._compute_entropy(log_std)
            
            # Get value estimate
            value = self.critic(state_tensor).cpu().item()
        
        info = {
            'log_prob': log_prob.cpu().item(),
            'entropy': entropy.cpu().item(),
            'value': value,
            'mean': mean.cpu().numpy()[0],
            'std': std.cpu().numpy()[0],
        }
        
        return action, info
    
    def _compute_log_prob(
        self, 
        action: torch.Tensor, 
        mean: torch.Tensor, 
        log_std: torch.Tensor
    ) -> torch.Tensor:
        """Compute log probability of action under Gaussian"""
        std = torch.exp(log_std)
        var = std ** 2
        log_prob = -0.5 * (
            ((action - mean) ** 2) / var + 2 * log_std + np.log(2 * np.pi)
        ).sum(dim=-1)
        return log_prob
    
    def _compute_entropy(self, log_std: torch.Tensor) -> torch.Tensor:
        """Compute entropy of Gaussian distribution"""
        entropy = 0.5 * (1 + np.log(2 * np.pi) + 2 * log_std).sum(dim=-1)
        return entropy
    
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: Dict,
    ):
        """Store transition in buffer"""
        # Compute next value with current critic
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            next_value = 0.0 if done else self.critic(next_state_tensor).cpu().item()
        
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['next_states'].append(next_state)
        self.buffer['dones'].append(done)
        self.buffer['values'].append(info['value'])
        self.buffer['next_values'].append(next_value)
        self.buffer['log_probs'].append(info['log_prob'])
        
        self.total_steps += 1
    
    def ready_to_update(self) -> bool:
        """Check if buffer is ready for update"""
        return len(self.buffer['states']) >= self.buffer_size
    
    def update(self) -> Tuple[float, float]:
        """Perform PPO update"""
        if len(self.buffer['states']) < self.buffer_size:
            return float('nan'), float('nan')
        
        # Convert buffer to numpy arrays
        states = np.array(self.buffer['states'][:self.buffer_size])
        actions = np.array(self.buffer['actions'][:self.buffer_size])
        rewards = np.array(self.buffer['rewards'][:self.buffer_size])
        dones = np.array(self.buffer['dones'][:self.buffer_size])
        old_values = np.array(self.buffer['values'][:self.buffer_size])
        old_next_values = np.array(self.buffer['next_values'][:self.buffer_size])
        old_log_probs = np.array(self.buffer['log_probs'][:self.buffer_size])
        
        # Compute advantages using GAE
        advantages = self._compute_gae(rewards, old_values, old_next_values, dones)
        
        # Debug advantage statistics (first few updates)
        if self.update_count < self.debug_verbose:
            self._debug_advantages(advantages)
        
        # Normalize advantages (IMPORTANT!)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Multiple epochs of updates
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_kl = 0.0
        
        for epoch in range(self.n_epochs):
            # Create mini-batches
            indices = np.random.permutation(self.buffer_size)
            num_batches = self.buffer_size // self.batch_size
            
            for batch in range(num_batches):
                batch_indices = indices[batch * self.batch_size:(batch + 1) * self.batch_size]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_rewards = rewards[batch_indices]
                batch_next_states = np.array(self.buffer['next_states'])[batch_indices]
                batch_dones = dones[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Update actor
                debug_this = (epoch == 0 and batch == 0 and 
                             self.update_count < self.debug_verbose)
                actor_loss, kl = self._update_actor(
                    batch_states, batch_actions, batch_advantages, 
                    batch_old_log_probs, debug_this
                )
                
                # Update critic
                critic_loss = self._update_critic(
                    batch_states, batch_rewards, batch_next_states, batch_dones
                )
                
                total_actor_loss += actor_loss
                total_critic_loss += critic_loss
                total_kl += kl
        
        # Average losses
        avg_actor_loss = total_actor_loss / (self.n_epochs * num_batches)
        avg_critic_loss = total_critic_loss / (self.n_epochs * num_batches)
        avg_kl = total_kl / (self.n_epochs * num_batches)
        
        # Store losses
        self.actor_losses.append(avg_actor_loss)
        self.critic_losses.append(avg_critic_loss)
        self.kl_history.append(avg_kl)
        
        # Clear buffer
        self.buffer = {key: [] for key in self.buffer.keys()}
        self.update_count += 1
        
        # Summary
        if self.update_count <= self.debug_verbose:
            print("\n" + "-" * 50)
            print(f"Update #{self.update_count} Summary:")
            print(f"  Actor Loss: {avg_actor_loss:.4f}")
            print(f"  Critic Loss: {avg_critic_loss:.4f}")
            print(f"  Avg KL Divergence: {avg_kl:.6f}")
            print("=" * 50)
        
        return avg_actor_loss, avg_critic_loss
    
    def _compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        next_values: np.ndarray,
        dones: np.ndarray,
    ) -> np.ndarray:
        """Compute Generalized Advantage Estimation"""
        N = len(rewards)
        advantages = np.zeros(N)
        
        last_advantage = 0.0
        for t in reversed(range(N)):
            if dones[t]:
                last_advantage = 0.0
                next_value = 0.0
            else:
                next_value = next_values[t]
            
            # TD error
            delta = rewards[t] + self.gamma * next_value - values[t]
            
            # GAE
            last_advantage = delta + self.gamma * self.lambda_gae * last_advantage
            advantages[t] = last_advantage
        
        return advantages
    
    def _debug_advantages(self, advantages: np.ndarray):
        """Debug advantage statistics"""
        print("\n" + "=" * 50)
        print(f"Update #{self.update_count + 1}: Advantage Analysis")
        print("=" * 50)
        print(f"Mean: {advantages.mean():.4f}, Std: {advantages.std():.4f}")
        print(f"Min/Max: [{advantages.min():.4f}, {advantages.max():.4f}]")
        pos_ratio = (advantages > 0).sum() / len(advantages)
        print(f"Positive ratio: {pos_ratio*100:.2f}% ({(advantages>0).sum()}/{len(advantages)})")
    
    def _update_actor(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        old_log_probs: np.ndarray,
        debug: bool = False,
    ) -> Tuple[float, float]:
        """Update actor network"""
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        
        # Forward pass
        mean, log_std = self.actor(states_tensor)
        
        # Compute new log probabilities
        new_log_probs = self._compute_log_prob(actions_tensor, mean, log_std)
        
        # Probability ratio
        ratio = torch.exp(new_log_probs - old_log_probs_tensor)
        
        # KL divergence (approximation)
        kl_div = (old_log_probs_tensor - new_log_probs).mean()
        
        # Surrogate objectives
        surr1 = ratio * advantages_tensor
        surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages_tensor
        
        # Debug detailed analysis (first batch of first epoch)
        if debug:
            self._debug_clip_loss(ratio, advantages_tensor, surr1, surr2, kl_div)
        
        # Clip loss
        clip_loss = -torch.min(surr1, surr2).mean()
        
        # Entropy loss
        entropy = self._compute_entropy(log_std).mean()
        entropy_loss = -self.entropy_coeff * entropy
        
        # Total actor loss
        # actor_loss = clip_loss + entropy_loss
        actor_loss = clip_loss

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item(), kl_div.item()
    
    def _debug_clip_loss(
        self,
        ratio: torch.Tensor,
        advantages: torch.Tensor,
        surr1: torch.Tensor,
        surr2: torch.Tensor,
        kl_div: torch.Tensor,
    ):
        """Debug clip loss details"""
        print("\n" + "-" * 50)
        # print("Clip Loss Detailed Analysis (First Batch):")
        print("Detailed Analysis (First Batch):")

        print("-" * 50)
        
        ratio_np = ratio.detach().cpu().numpy()
        adv_np = advantages.detach().cpu().numpy()
        surr1_np = surr1.detach().cpu().numpy()
        surr2_np = surr2.detach().cpu().numpy()
        
        # Ratio statistics
        # print("\n1. Probability Ratio Statistics:")
        # print(f"   Mean: {ratio_np.mean():.4f}, Std: {ratio_np.std():.4f}")
        # print(f"   Min: {ratio_np.min():.4f}, Max: {ratio_np.max():.4f}")
        # print(f"   Median: {np.median(ratio_np):.4f}")
        
        # Clipping analysis
        # print("\n1. Clipping Analysis:")
        clipped = np.abs(ratio_np - np.clip(ratio_np, 1-self.clip_range, 1+self.clip_range)) > 1e-8
        # print(f"   Ratio hit boundary: {clipped.mean()*100:.2f}% ({clipped.sum()}/{len(clipped)})")
        
        surr2_used = surr2_np < surr1_np
        # print(f"   Surr2 actually used: {surr2_used.mean()*100:.2f}% ({surr2_used.sum()}/{len(surr2_used)})")
        
        # Advantage-wise analysis
        pos_adv = adv_np > 0
        neg_adv = adv_np < 0
        
        print("\n1. Analysis by Advantage Sign:")
        print(f"\n   Positive advantages ({pos_adv.sum()} samples, {pos_adv.mean()*100:.1f}%):")
        # if pos_adv.sum() > 0:
            # print(f"     Ratio: mean={ratio_np[pos_adv].mean():.4f}, median={np.median(ratio_np[pos_adv]):.4f}")
            # print(f"     Boundary hit: {clipped[pos_adv].mean()*100:.2f}%")
            # print(f"     Surr2 used: {surr2_used[pos_adv].mean()*100:.2f}%")
        
        print(f"\n   Negative advantages ({neg_adv.sum()} samples, {neg_adv.mean()*100:.1f}%):")
        # if neg_adv.sum() > 0:
            # print(f"     Ratio: mean={ratio_np[neg_adv].mean():.4f}, median={np.median(ratio_np[neg_adv]):.4f}")
            # print(f"     Boundary hit: {clipped[neg_adv].mean()*100:.2f}%")
            # print(f"     Surr2 used: {surr2_used[neg_adv].mean()*100:.2f}%")
        
        # print(f"\n3. KL Divergence: {kl_div.item():.6f}")
        # print("-" * 50)
    
    def _update_critic(
        self,
        states: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> float:
        """Update critic network with TD(0) loss"""
        states_tensor = torch.FloatTensor(states).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        
        # Current values
        values = self.critic(states_tensor).squeeze()
        
        # Next values (detached for TD target)
        with torch.no_grad():
            next_values = self.critic(next_states_tensor).squeeze()
            next_values = next_values * (1 - dones_tensor)
        
        # TD target
        td_targets = rewards_tensor + self.gamma * next_values
        
        # TD(0) loss
        critic_loss = 0.5 * ((td_targets - values) ** 2).mean()
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return critic_loss.item()
    
    def episode_end(self, episode_reward: float, episode_length: int, avg_sir: float):
        """Record episode statistics"""
        self.episode_count += 1
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.episode_sirs.append(avg_sir)
    
    def get_info(self) -> Dict:
        """Get agent statistics"""
        info = {
            'total_steps': self.total_steps,
            'episode_count': self.episode_count,
            'update_count': self.update_count,
            'buffer_fill': len(self.buffer['states']) / self.buffer_size,
        }
        
        if len(self.episode_rewards) > 0:
            info['avg_reward'] = np.mean(self.episode_rewards[-100:])
            info['avg_sir'] = np.mean(self.episode_sirs[-100:])
            info['avg_actor_loss'] = np.mean(self.actor_losses[-10:]) if self.actor_losses else 0
            info['avg_critic_loss'] = np.mean(self.critic_losses[-10:]) if self.critic_losses else 0
        
        return info
    
    def save(self, filepath: str):
        """Save agent"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'episode_count': self.episode_count,
            'update_count': self.update_count,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'kl_history': self.kl_history,
            'episode_rewards': self.episode_rewards,
            'episode_sirs': self.episode_sirs,
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load agent"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.total_steps = checkpoint['total_steps']
        self.episode_count = checkpoint['episode_count']
        self.update_count = checkpoint['update_count']
        self.actor_losses = checkpoint['actor_losses']
        self.critic_losses = checkpoint['critic_losses']
        self.kl_history = checkpoint['kl_history']
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_sirs = checkpoint['episode_sirs']
        print(f"Model loaded from {filepath}")
