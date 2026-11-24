import numpy as np
from bee_mic.simulation import SourceSimulator
from bee_mic.geometry import ArrayGeometry
from bee_mic.dsp import AcousticSolver

def validate_tdoa_physics():
    """
    Validates solver math, TDOA accuracy, statistical bias, and spatial GDOP
    before trusting the simulator for new array geometries.
    """
    print("=== 1. DETERMINISTIC & ORACLE CHECKS (NOISELESS) ===")
    test_pos = np.array([1.8, -0.6, 1.2])
    sim = SourceSimulator(fs=48000)
    solver = AcousticSolver(fs=48000)
    mic_array = ArrayGeometry.perimeter_array()
    test_signal = sim.generate_pollinator_buzz(duration=3)

    # Compute exact analytical ground-truth TDOAs
    dists = np.linalg.norm(mic_array - test_pos, axis=1)
    true_tdoas = (dists - dists[0]) / sim.c_sound

    # Test A: Multilateration solver alone (fed exact TDOAs)
    solver_only_pos = solver.multilateration(mic_array, true_tdoas)
    solver_err = np.linalg.norm(test_pos - solver_only_pos)
    print(f"Solver-only error (exact TDOAs):  {solver_err * 1e6:.4f} µm  [Target: < 1 µm]")

    # Test B: Full pipeline at ultra-high SNR (100 dB = virtually noiseless)
    rx_clean, _ = sim.simulate_stationary_propagation(test_pos, mic_array, test_signal, snr_db=100)
    est_tdoas = solver.estimate_tdoa(rx_clean, ref_mic_idx=0)
    tdoa_err_us = np.max(np.abs(true_tdoas - est_tdoas)) * 1e6

    full_clean_pos = solver.multilateration(mic_array, est_tdoas)
    full_clean_err = np.linalg.norm(test_pos - full_clean_pos)
    print(f"Max TDOA error (100 dB SNR):      {tdoa_err_us:.4f} µs   [Target: < 0.1 µs]")
    print(f"Full pipeline error (100 dB SNR): {full_clean_err * 1000:.4f} mm  [Target: < 1.0 mm]")

    print("\n=== 2. MONTE CARLO NOISE & BIAS CHECK (50 TRIALS @ 20 dB SNR) ===")
    estimates = []
    for _ in range(50):
        rx_noisy, _ = sim.simulate_stationary_propagation(test_pos, mic_array, test_signal, snr_db=20)
        tdoas = solver.estimate_tdoa(rx_noisy, ref_mic_idx=0)
        estimates.append(solver.multilateration(mic_array, tdoas))

    estimates = np.array(estimates)
    mean_est = np.mean(estimates, axis=0)
    bias_cm = np.linalg.norm(mean_est - test_pos) * 100
    std_xyz_cm = np.std(estimates, axis=0) * 100

    print(f"Mean estimated position:          {np.round(mean_est, 4)} (True: {test_pos})")
    print(f"Systematic bias (mean error):     {bias_cm:.2f} cm   [Target: < 2.0 cm]")
    print(f"XYZ scatter (1-sigma std dev):    {np.round(std_xyz_cm, 2)} cm")

    print("\n=== 3. SPATIAL RANGE & GDOP SANITY CHECK (100 dB SNR) ===")
    for r in [0.5, 1.0, 2.0, 4.0]:
        # Place target along a fixed diagonal ray at increasing distances
        direction = np.array([1.0, -0.5, 0.8])
        direction /= np.linalg.norm(direction)
        pos = direction * r

        rx, _ = sim.simulate_stationary_propagation(pos, mic_array, test_signal, snr_db=100)
        tdoas = solver.estimate_tdoa(rx, ref_mic_idx=0)
        est = solver.multilateration(mic_array, tdoas)

        # Decompose error into Angular (bearing) vs Radial (range) components
        est_dir = est / np.linalg.norm(est)
        angle_err_deg = np.degrees(np.arccos(np.clip(np.dot(direction, est_dir), -1.0, 1.0)))
        range_err_cm = abs(np.linalg.norm(est) - r) * 100
        print(f"Range {r:3.1f} m -> Bearing Error: {angle_err_deg:.4f}° | Range Error: {range_err_cm:.3f} cm")

if __name__ == "__main__":
    validate_tdoa_physics()