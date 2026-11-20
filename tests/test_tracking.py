import numpy as np
from bee_mic.tracking import KinematicKalmanFilter


def test_kalman_coasting_behavior():
    """
    Ensures that if acoustic measurements suddenly jump
    (e.g., during a boundary singularity), the filter resists the noise.
    """
    kf = KinematicKalmanFilter(dt=0.02, initial_pos=[0.0, 0.0, 0.5], process_noise=0.01, measurement_noise=5.0)

    # 1. Simulate 5 steps of steady flight moving +0.1 in X per step
    for i in range(1, 6):
        kf.predict()
        kf.update(np.array([i * 0.1, 0.0, 0.5]))

    # 2. Inject a massive acoustic error (simulating a boundary failure)
    kf.predict()
    smoothed_pos = kf.update(np.array([5.0, 0.0, 0.5]))  # Raw jumps to X=5.0!

    # 3. The filter should reject this based on physical inertia
    # X should be close to 0.6, resisting the pull toward 5.0
    assert smoothed_pos[0] < 1.0, f"Kalman filter failed to reject noise. X = {smoothed_pos[0]}"
    assert smoothed_pos[0] > 0.5, f"Kalman filter over-damped trajectory. X = {smoothed_pos[0]}"


def test_kalman_initialization():
    """Verifies state vector initializes correctly."""
    start_pos = [1.0, -1.0, 0.5]
    kf = KinematicKalmanFilter(dt=0.02, initial_pos=start_pos)
    assert np.allclose(kf.x[:3], start_pos), "Initial position mismatch"
    assert np.allclose(kf.x[3:], [0, 0, 0]), "Initial velocity should be zero"

if __name__ == "__main__":
    test_kalman_initialization()
    test_kalman_coasting_behavior()