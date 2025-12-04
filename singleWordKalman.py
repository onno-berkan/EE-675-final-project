import numpy as np
import matplotlib.pyplot as plt

class SingleWordKalmanTracker:
    def __init__(self, n_channels):
        """
        Kalman Filter optimized for tracking neural drift of a single word.
        Assumes A=I, H=I, and diagonal covariance matrices for speed.
        """
        self.n_channels = n_channels
        self.n_timesteps = 0
        
        # Model Parameters (to be fit via EM)
        self.Q = None  # Process Noise (Drift) Variance (Vector: diagonal of Q matrix)
        self.R = None  # Measurement Noise Variance (Vector: diagonal of R matrix)
        self.mu_0 = None # Initial state mean
        self.P_0 = None  # Initial state covariance
        
        # Storage for the E-step results
        self.x_smooth = None
        self.P_smooth = None
        self.P_lag_smooth = None

    def initialize_parameters(self, data):
        """
        Smart initialization based on the first few trials.
        data: (T, n_channels)
        """
        # Initialize Mean
        self.mu_0 = np.mean(data, axis=0)
        self.P_0 = np.ones(self.n_channels) * 10.0
        
        # Initialize R (Noise)
        self.R = np.var(data, axis=0) + 1.0
        
        # Initialize Q (Drift) to be small relative to R
        self.Q = self.R * 0.01 
        
    def e_step(self, data):
        """
        Runs Forward Filter and Backward Smoother to estimate latent states.
        Uses Diagonal Matrix optimizations (element-wise operations).
        """
        T, C = data.shape
        self.n_timesteps = T
        
        # --- 1. Forward Pass (Kalman Filter) ---
        x_pred = np.zeros((T, C))
        P_pred = np.zeros((T, C))
        x_filt = np.zeros((T, C))
        P_filt = np.zeros((T, C))
        
        # Initial step
        x_pred[0] = self.mu_0
        P_pred[0] = self.P_0 + self.Q
        
        # Update step t=0
        K = P_pred[0] / (P_pred[0] + self.R)
        x_filt[0] = x_pred[0] + K * (data[0] - x_pred[0])
        P_filt[0] = (1 - K) * P_pred[0]
        
        for t in range(1, T):
            # Predict (A=I)
            x_pred[t] = x_filt[t-1]
            P_pred[t] = P_filt[t-1] + self.Q
            
            # Update (H=I)
            K = P_pred[t] / (P_pred[t] + self.R)
            x_filt[t] = x_pred[t] + K * (data[t] - x_pred[t])
            P_filt[t] = (1 - K) * P_pred[t]

        # --- 2. Backward Pass (RTS Smoother) ---
        self.x_smooth = np.zeros((T, C))
        self.P_smooth = np.zeros((T, C))
        self.P_lag_smooth = np.zeros((T, C)) # Covariance between t and t-1
        
        # Last step initialization
        self.x_smooth[-1] = x_filt[-1]
        self.P_smooth[-1] = P_filt[-1]
        
        # The lag covariance for the last step (simplified approx for A=I)
        # P_{T, T-1} = (I - K_T) * A * P_{T-1|T-1}
        # Since A=I, P_{T, T-1} = P_filt[T] @ J_{T-1}^inv ? 
        # Actually, standard formula: P_{t, t-1}^s = P_t^s * J_{t-1}
        
        for t in range(T-2, -1, -1):
            # Smoother Gain: J = P_{t|t} * A^T * P_{t+1|t}^-1
            # With diagonal matrices and A=I:
            J = P_filt[t] / P_pred[t+1]
            
            self.x_smooth[t] = x_filt[t] + J * (self.x_smooth[t+1] - x_pred[t+1])
            self.P_smooth[t] = P_filt[t] + (J**2) * (self.P_smooth[t+1] - P_pred[t+1])
            
            # Lag-1 Covariance: Cov(x_t+1, x_t)
            # Standard RTS Lag-1 formula
            self.P_lag_smooth[t+1] = self.P_smooth[t+1] * J

    def m_step(self, data):
        """
        Updates parameters Q and R to maximize likelihood.
        """
        T, C = data.shape
        
        # --- Update Q (Drift) ---
        # Q = 1/(T-1) * Sum [ E(dx^2) ]
        # E[(x_t - x_{t-1})^2] = (x_t - x_{t-1})^2 + P_t + P_{t-1} - 2*P_{t,t-1}
        
        diff_sq = (self.x_smooth[1:] - self.x_smooth[:-1])**2
        var_terms = self.P_smooth[1:] + self.P_smooth[:-1] - 2 * self.P_lag_smooth[1:]
        
        self.Q = np.mean(diff_sq + var_terms, axis=0)
        
        # Prevent Q from collapsing to 0 (numerical stability)
        self.Q = np.maximum(self.Q, 1e-6)

        # --- Update R (Observation Noise) ---
        # R = 1/T * Sum [ E(error^2) ]
        # E[(y_t - x_t)^2] = (y_t - x_t)^2 + P_t
        
        res_sq = (data - self.x_smooth)**2
        self.R = np.mean(res_sq + self.P_smooth, axis=0)

        # --- Update Initial Conditions ---
        self.mu_0 = self.x_smooth[0]
        self.P_0 = self.P_smooth[0]

    def fit(self, data, max_iter=100, tol=1e-4):
        """
        Runs the EM loop until convergence.
        """
        self.initialize_parameters(data)
        
        prev_log_lik = -np.inf
        
        print(f"Starting EM for {data.shape[0]} trials on {data.shape[1]} channels...")
        
        for i in range(max_iter):
            self.e_step(data)
            self.m_step(data)
            
            # Compute Log Likelihood (Approximate using observation errors)
            # A full LL calc is expensive, usually we monitor parameter convergence
            # Here we just track the norm of Q to see it stabilizes
            q_norm = np.linalg.norm(self.Q)
            print(f"Iter {i+1}: Mean Drift (Q) = {np.mean(self.Q):.4f}, Mean Noise (R) = {np.mean(self.R):.4f}")
            
            # Check convergence (simplified)
            if np.abs(q_norm - prev_log_lik) < tol:
                print("Converged.")
                break
            prev_log_lik = q_norm

        return self.x_smooth