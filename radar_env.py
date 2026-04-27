"""
RADAR Dynamic Environment
Fully dynamic environment where both desired and interference signals move continuously
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Optional


class RADARDynamic(gym.Env):
    """Dynamic RADAR environment with moving signals"""
    
    def __init__(
        self,
        num_elements: int = 8,
        snr_db: float = 5.0,
        isr_db: float = 10.0,
        num_interferers: int = 1,
        output_mode: str = 'triu_norm',
        gain_mode: str = 'peak',
        angle_min: float = -40.0,
        angle_max: float = 40.0,
        max_steps: int = 1000,
        desired_delta_deg: float = 0.1,
        int_delta_deg: float = 0.1,
        fc: float = 9.5e9,
        lsnap: int = 200,
        duty: float = 0.2,
        bw: float = 1e6,
        fddc: float = 4e6,
    ):
        super().__init__()
        
        # Basic parameters
        self.num_elements = num_elements
        self.snr_db = snr_db
        self.isr_db = isr_db
        self.num_interferers = num_interferers
        self.output_mode = output_mode
        self.gain_mode = gain_mode
        
        # Angle boundaries
        self.angle_min = angle_min
        self.angle_max = angle_max
        
        # Episode parameters
        self.max_steps = max_steps
        self.desired_delta_deg = desired_delta_deg
        self.int_delta_deg = int_delta_deg
        
        # RF/Array parameters
        self.fc = fc
        self.c = 3e8  # speed of light
        self.lambda_ = self.c / self.fc
        self.d_over_lambda = 0.5
        
        # Snapshot parameters
        self.lsnap = lsnap
        self.duty = duty
        self.bw = bw
        self.fddc = fddc
        
        # Generate LFM reference signal
        self._generate_lfm_reference()

        # Precompute steering matrix over evaluation angle grid (vectorized evaluate)
        self._eval_angles = np.arange(-90, 90.1, 0.1)
        u = np.sin(np.deg2rad(self._eval_angles))
        n = np.arange(num_elements)
        self._eval_A = np.exp(
            1j * 2 * np.pi * self.d_over_lambda * np.outer(u, n)
        )  # shape: (len(angles), M)

        # Define spaces
        # Observation is the steered covariance Gamma' = D·Gamma·D^H, so theta_D
        # is implicit in the transform and not part of the state.
        if output_mode == 'triu_norm':
            K = num_elements * (num_elements + 1) // 2
            state_size = 2 * K  # real(z) + imag(z)
        else:
            state_size = 2 * num_elements * num_elements
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2 * num_elements,), dtype=np.float32
        )
        
        # Internal state
        self.cur_step = 0
        self.desired_angle = 0.0
        self.desired_direction = 1
        self.int_angles_deg = np.zeros(num_interferers)
        self.int_direction = np.ones(num_interferers)
        self.current_gamma = None
        self.current_theta = 0.0
        
    def _generate_lfm_reference(self):
        """Generate LFM (Linear Frequency Modulation) reference signal"""
        PRI = self.lsnap / self.fddc
        T_pulse = self.duty * PRI
        K = self.bw / T_pulse
        t = np.arange(self.lsnap) / self.fddc
        
        # Generate LFM pulse
        rect_pulse = np.abs(t - T_pulse/2) <= T_pulse/2
        phase = 2 * np.pi * (0.5 * K * (t - T_pulse/2)**2)
        iq0 = rect_pulse * np.exp(1j * phase)
        
        # Apply delay
        delay = int(self.lsnap / 5)
        self.lfm_ref = np.roll(iq0, delay)
        
    def _steer_vector(self, deg: float) -> np.ndarray:
        """Generate steering vector for given angle"""
        u = np.sin(np.deg2rad(deg))
        n = np.arange(self.num_elements)
        a = np.exp(1j * 2 * np.pi * self.d_over_lambda * n * u)
        return a
    
    def _compute_gamma(self, theta_deg: float) -> np.ndarray:
        """Compute covariance matrix for current signal configuration"""
        M = self.num_elements
        L = self.lsnap
        
        # Signal power
        sig_pwr = 1.0
        jam_pwr = sig_pwr * 10 ** (self.isr_db / 10)
        noise_var = sig_pwr / 10 ** (self.snr_db / 10)
        
        # Desired signal
        a_D = self._steer_vector(theta_deg)
        s_D = np.sqrt(sig_pwr / self.duty) * self.lfm_ref
        X = np.outer(a_D, s_D)
        
        # Interference signals
        for q in range(self.num_interferers):
            a_i = self._steer_vector(self.int_angles_deg[q])
            s_i = np.sqrt(jam_pwr) * (
                np.random.randn(L) + 1j * np.random.randn(L)
            ) / np.sqrt(2)
            X = X + np.outer(a_i, s_i)
        
        # Noise
        N = np.sqrt(noise_var / 2) * (
            np.random.randn(M, L) + 1j * np.random.randn(M, L)
        )
        X = X + N
        
        # Covariance matrix
        Gamma = (X @ X.conj().T) / L
        return Gamma
    
    def reset(
        self, 
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        range_deg = self.angle_max - self.angle_min
        
        # Initialize desired signal
        if options and 'desired_angle' in options:
            self.desired_angle = options['desired_angle']
        else:
            self.desired_angle = np.random.rand() * range_deg + self.angle_min
        
        self.desired_direction = np.random.choice([-1, 1])
        
        # Initialize interference signals
        if options and 'int_angles_deg' in options:
            self.int_angles_deg = np.array(options['int_angles_deg'])
        else:
            self.int_angles_deg = (
                np.random.rand(self.num_interferers) * range_deg + self.angle_min
            )
        
        self.int_direction = np.random.choice([-1, 1], size=self.num_interferers)
        
        # Reset episode
        self.cur_step = 0
        self.current_gamma = None
        self.current_theta = self.desired_angle
        
        obs = self._get_observation()
        info = {
            'desired_angle': self.desired_angle,
            'desired_direction': self.desired_direction,
            'int_angles_deg': self.int_angles_deg.copy(),
            'int_direction': self.int_direction.copy(),
        }
        
        return obs, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        theta_deg = self.desired_angle

        # Compute or retrieve Gamma
        if self.current_gamma is None or self.current_theta != theta_deg:
            Gamma = self._compute_gamma(theta_deg)
            self.current_gamma = Gamma
            self.current_theta = theta_deg
        else:
            Gamma = self.current_gamma

        # Steer to desired direction: Gamma' = D · Gamma · D^H, D = diag(a*(theta_D))
        # In the transformed frame, the desired signal sits at broadside,
        # so the agent doesn't need theta_D as an input.
        a_D = self._steer_vector(theta_deg)
        d = a_D.conj()
        Gamma_prime = (d[:, None] * Gamma) * d.conj()[None, :]

        if self.output_mode == 'triu_norm':
            M = self.num_elements
            triu_indices = np.triu_indices(M)
            z = Gamma_prime[triu_indices]
            z_norm = z / (np.linalg.norm(z) + 1e-8)
            obs = np.concatenate([z_norm.real, z_norm.imag])
        else:
            Gamma_flat = Gamma_prime.flatten()
            Gamma_norm = Gamma_flat / (np.linalg.norm(Gamma_flat) + 1e-8)
            obs = np.concatenate([Gamma_norm.real, Gamma_norm.imag])

        return obs.astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment"""
        self.cur_step += 1
        
        # Move desired signal with bounce
        self.desired_angle += self.desired_delta_deg * self.desired_direction
        
        if self.desired_angle >= self.angle_max or self.desired_angle <= self.angle_min:
            self.desired_direction = -self.desired_direction
            self.desired_angle = np.clip(self.desired_angle, self.angle_min, self.angle_max)
        
        # Move interference signals with bounce
        self.int_angles_deg += self.int_delta_deg * self.int_direction
        
        for i in range(self.num_interferers):
            if self.int_angles_deg[i] >= self.angle_max or self.int_angles_deg[i] <= self.angle_min:
                self.int_direction[i] = -self.int_direction[i]
                self.int_angles_deg[i] = np.clip(
                    self.int_angles_deg[i], self.angle_min, self.angle_max
                )
        
        # Compute new Gamma
        theta_deg = self.desired_angle
        Gamma = self._compute_gamma(theta_deg)
        self.current_gamma = Gamma
        self.current_theta = theta_deg
        
        # Get observation
        obs = self._get_observation()
        
        # Evaluate action (weights)
        eval_info = self.evaluate(action)
        
        # Compute reward (will be done in training script)
        reward = 0.0  # Placeholder
        
        # Check termination
        terminated = (self.cur_step >= self.max_steps)
        truncated = False
        
        info = {
            'PD_dB': eval_info['PD_dB'],
            'PI_dB': eval_info['PI_dB'],
            'SIR_dB': eval_info['SIR_dB'],
            'desired_angle': theta_deg,  # Consistent with reset()
            'int_angles_deg': self.int_angles_deg.copy(),
        }
        
        return obs, reward, terminated, truncated, info
    
    def evaluate(self, action: np.ndarray) -> Dict:
        """Evaluate beamforming weights"""
        M = self.num_elements

        # Action is a unit-norm weight w' in the steered (broadside) frame.
        # Inverse-transform to the physical weight: w = D^H · w' = a(theta_D) ⊙ w'.
        w_prime = action[:M] + 1j * action[M:2*M]
        w_prime = w_prime / (np.linalg.norm(w_prime) + 1e-8)

        theta_deg = self.desired_angle
        a_D = self._steer_vector(theta_deg)
        w = a_D * w_prime

        # Compute array pattern (vectorized)
        angles = self._eval_angles
        AF_all = np.abs(self._eval_A @ w.conj()) ** 2
        
        # Reference gain
        max_gain = max(np.max(AF_all), 1e-10)
        if self.gain_mode == 'directivity':
            ref_gain = max(np.mean(AF_all), 1e-10)
        else:  # 'peak'
            ref_gain = max_gain
        AF_pattern_dB = 10 * np.log10(AF_all / ref_gain + 1e-10)

        # Desired signal gain - lookup from AF_all
        desired_idx = int(round((theta_deg + 90) / 0.1))
        desired_idx = np.clip(desired_idx, 0, len(angles) - 1)
        PD = AF_all[desired_idx] / ref_gain
        PD_dB = 10 * np.log10(max(PD, 1e-10))

        # Interference gain - lookup from AF_all
        PI_total = 0.0
        PI_dB_single = []

        for q in range(self.num_interferers):
            int_idx = int(round((self.int_angles_deg[q] + 90) / 0.1))
            int_idx = np.clip(int_idx, 0, len(angles) - 1)
            PI_q = AF_all[int_idx] / ref_gain
            PI_total += PI_q
            PI_dB_single.append(10 * np.log10(max(PI_q, 1e-10)))

        PI_dB = 10 * np.log10(max(PI_total, 1e-10))
        SIR_dB = PD_dB - PI_dB
        
        return {
            'PD_dB': PD_dB,
            'PI_dB': PI_dB,
            'SIR_dB': SIR_dB,
            'theta_deg': theta_deg,
            'int_angles_deg': self.int_angles_deg.copy(),
            'w': w,
            'PI_dB_single': PI_dB_single,
            'angles': angles,
            'AF_pattern_dB': AF_pattern_dB,
        }
