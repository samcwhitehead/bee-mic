# src/bee-mic/tracking.py
"""
Tools for tracking moving acoustic sources

"""
import numpy as np

# Kalman filter definition
class KinematicKalmanFilter:
    def __init__(self, dt, initial_pos, process_noise=5.0, measurement_noise=0.1):
        """
        3D Constant Velocity Kalman Filter to smooth erratic acoustic localizations.
        """
        self.dt = dt

        # State Vector: [X, Y, Z, Vx, Vy, Vz]
        self.x = np.array([initial_pos[0], initial_pos[1], initial_pos[2], 0.0, 0.0, 0.0])

        # State Transition Matrix (Physics: pos = pos + velocity * dt)
        self.F = np.eye(6)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        # Measurement Matrix (We only observe position, not velocity)
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # Process Noise Covariance (How erratically can the bee accelerate?)
        # Higher = trusts acoustic data more. Lower = smoother/stiffer flight path.
        q = process_noise
        self.Q = np.eye(6) * q

        # Measurement Noise Covariance (How much do we trust SRP-PHAT?)
        # Variance of the acoustic error (e.g., 0.1 meters)
        self.R = np.eye(3) * measurement_noise

        # Covariance Matrix
        self.P = np.eye(6) * 1.0

    def predict(self):
        # Predict next state based on physics
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3]

    def update(self, measurement):
        # Calculate Kalman Gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state with new acoustic measurement
        y = measurement - (self.H @ self.x)
        self.x = self.x + (K @ y)

        # Update covariance
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P
        return self.x[:3]

