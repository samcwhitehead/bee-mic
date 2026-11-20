import argparse
import numpy as np

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from bee_mic.simulation import SourceSimulator
from bee_mic.geometry import ArrayGeometry
from bee_mic.dsp import AcousticSolver
from bee_mic.tracking import KinematicKalmanFilter
from bee_mic.visualization import plot_flight_path_3d, plot_spatial_error_heatmap, plot_3d_geometry  # , plot_gcc_phat_diagnostic


def demo_standard_flight():
    print("=== DYNAMIC TRACKING: STANDARD FLIGHT ===")
    sim = SourceSimulator(fs=48000)
    mics = ArrayGeometry.perimeter_array()

    # 1. Generate trajectory
    duration = 0.4
    sig = sim.generate_pollinator_buzz(duration=duration, f0=250.0)
    start_pos, end_pos = np.array([-0.8, -0.8, 0.3]), np.array([0.8, 0.8, 0.6])
    trajectory = np.linspace(start_pos, end_pos, len(sig))

    # 2. Simulate moving audio
    rx_moving = sim.simulate_moving_propagation(trajectory, mics, sig, snr_db=20)

    # 3. Track with SRP-PHAT and Kalman Filter
    solver = AcousticSolver(fs=48000)
    kf = KinematicKalmanFilter(dt=0.02, initial_pos=start_pos, process_noise=8.0, measurement_noise=0.15)

    window_samples = int(0.04 * sim.fs)
    step_samples = int(0.02 * sim.fs)

    raw_positions, smoothed_positions, actual_positions = [], [], []

    for start_idx in range(0, rx_moving.shape[1] - window_samples, step_samples):
        audio_chunk = rx_moving[:, start_idx:start_idx + window_samples]

        mid_time_emit = (start_idx + window_samples / 2) / sim.fs
        if int(mid_time_emit * sim.fs) >= len(sig):
            break

        true_mid_pos = trajectory[int(mid_time_emit * sim.fs)]
        actual_positions.append(true_mid_pos)

        raw_pos, _ = solver.srp_phat(audio_chunk, mics, bounds=([-1, -1, 0], [1, 1, 1]))
        raw_positions.append(raw_pos[0])

        kf.predict()
        smoothed_pos = kf.update(raw_pos[0])
        smoothed_positions.append(smoothed_pos)

        print(
            f"T: {(start_idx / sim.fs) * 1000:4.0f}ms | Raw Err: {np.linalg.norm(true_mid_pos - raw_pos[0]) * 100:5.1f}cm | Smooth Err: {np.linalg.norm(true_mid_pos - smoothed_pos) * 100:5.1f}cm")

    plot_flight_path_3d(actual_positions, raw_positions, smoothed_positions, mic_positions=mics)


def demo_flyover_singularity():
    print("=== DYNAMIC TRACKING: FLYOVER SINGULARITY ===")
    sim = SourceSimulator(fs=48000)
    mics = ArrayGeometry.perimeter_array()

    # Trajectory passes exactly over the center mic (Mic 0) at [0,0,0.5]
    sig = sim.generate_pollinator_buzz(duration=0.4, f0=250.0)
    trajectory = np.linspace(np.array([-0.5, -0.5, 0.5]), np.array([0.5, 0.5, 0.5]), len(sig))

    rx_moving = sim.simulate_moving_propagation(trajectory, mics, sig, snr_db=30)

    # Extract a 40ms chunk from the exact middle of the simulation (the flyover)
    mid_idx = rx_moving.shape[1] // 2
    window_samples = int(0.04 * sim.fs)
    bad_chunk = rx_moving[:, mid_idx - window_samples // 2: mid_idx + window_samples // 2]

    # print("Plotting degraded cross-correlation peak for Center Mic (0) vs Perimeter Mic (1)...")
    # print("Notice the peak is smeared or split due to massive Doppler and delay shifts.")
    # plot_gcc_phat_diagnostic(bad_chunk, fs=48000, mic_A=0, mic_B=1)


def demo_multiple_stationary():
    print("=== DYNAMIC TRACKING: TWO STATIONARY BEES ===")
    sim = SourceSimulator(fs=48000)
    mics = ArrayGeometry.perimeter_array(width=2.0, length=2.0)

    duration = 0.06

    # 1. Bee A: Bumblebee hovering in the Northwest quadrant
    sig_A = sim.generate_pollinator_buzz(duration=duration, f0=230.0)
    pos_A = np.array([-0.5, 0.5, 0.5])
    rx_A, _ = sim.simulate_propagation(pos_A, mics, sig_A, snr_db=25)

    # 2. Bee B: Honeybee hovering in the Southeast quadrant
    sig_B = sim.generate_pollinator_buzz(duration=duration, f0=240.0)
    pos_B = np.array([0.5, -0.5, 0.5])
    rx_B, _ = sim.simulate_propagation(pos_B, mics, sig_B, snr_db=25)

    # 3. Superposition
    rx_combined = rx_A + rx_B

    solver = AcousticSolver(fs=48000)

    print(f"True Position A: {pos_A}")
    print(f"True Position B: {pos_B}")
    print("Tracking combined audio. Watch the raw SRP-PHAT output:")

    # raw_pos is your coordinates, energy_grid is the (41, 41, 21) array
    est_positions, energy_grid = solver.srp_phat(rx_combined, mics, bounds=([-1, -1, 0], [1, 1, 1]), num_targets=2)

    # 4. Match estimates to ground truth (order doesn't matter)
    for est in est_positions:
        err_A = np.linalg.norm(pos_A - est) * 100
        err_B = np.linalg.norm(pos_B - est) * 100

        if err_A < err_B:
            print(f"Located Target near Bee A | Error: {err_A:.1f} cm")
        else:
            print(f"Located Target near Bee B | Error: {err_B:.1f} cm")

    # 5. Visualize positions in 3D space
    plot_3d_geometry(mics, np.vstack((pos_A, pos_B)), est_positions)


# --------------------------------------------------------------
def demo_multiple_moving():
    print("=== DYNAMIC TRACKING: MULTI-TARGET DATA ASSOCIATION ===")
    sim = SourceSimulator(fs=48000)
    mics = ArrayGeometry.perimeter_array(width=2.0, length=2.0)

    duration = 0.5

    # Both bees have the EXACT same frequency to prove spatial separation works
    sig_A = sim.generate_pollinator_buzz(duration=duration, f0=250.0)
    traj_A = np.linspace(np.array([-0.9, -0.5, 0.4]), np.array([0.9, -0.5, 0.4]), len(sig_A))
    rx_A = sim.simulate_moving_propagation(traj_A, mics, sig_A, snr_db=25)

    sig_B = sim.generate_pollinator_buzz(duration=duration, f0=250.0)
    traj_B = np.linspace(np.array([0.9, 0.5, 0.6]), np.array([-0.9, 0.5, 0.6]), len(sig_B))
    rx_B = sim.simulate_moving_propagation(traj_B, mics, sig_B, snr_db=25)

    rx_combined = rx_A + rx_B
    solver = AcousticSolver(fs=48000)

    # Initialize TWO independent Kalman filters
    kf_A = KinematicKalmanFilter(dt=0.02, initial_pos=traj_A[0], process_noise=0.05, measurement_noise=0.1)
    kf_B = KinematicKalmanFilter(dt=0.02, initial_pos=traj_B[0], process_noise=0.05, measurement_noise=0.1)

    window_samples = int(0.04 * sim.fs)
    step_samples = int(0.02 * sim.fs)

    all_raw_detections = []
    smoothed_A_list, smoothed_B_list = [], []

    for start_idx in range(0, rx_combined.shape[1] - window_samples, step_samples):
        audio_chunk = rx_combined[:, start_idx:start_idx + window_samples]

        # 1. Get raw acoustic peaks (up to 2)
        raw_pos, _ = solver.srp_phat(audio_chunk, mics, bounds=([-1, -1, 0], [1, 1, 1]), num_targets=2)
        all_raw_detections.extend(raw_pos)

        # 2. PREDICT where both bees should be right now
        kf_A.predict()
        kf_B.predict()

        predictions = np.array([kf_A.x[:3], kf_B.x[:3]])
        measurements = np.array(raw_pos)

        # 3. DATA ASSOCIATION (Global Nearest Neighbor)
        if len(measurements) > 0:
            # Create a 2xN matrix of Euclidean distances between predictions and measurements
            cost_matrix = cdist(predictions, measurements)

            # The Hungarian algorithm optimally pairs filters to measurements to minimize total distance
            filter_indices, measurement_indices = linear_sum_assignment(cost_matrix)

            # 4. UPDATE the filters with their properly assigned measurements
            for f_idx, m_idx in zip(filter_indices, measurement_indices):
                # Optional Distance Gate: Reject measurement if it's absurdly far away (e.g., > 1.0m jump)
                if cost_matrix[f_idx, m_idx] < 1.0:
                    matched_meas = measurements[m_idx]
                    if f_idx == 0:
                        kf_A.update(matched_meas)
                    else:
                        kf_B.update(matched_meas)

        # Record the current state (if a filter didn't get a measurement this frame,
        # it just coasts naturally on its physics prediction)
        smoothed_A_list.append(kf_A.x[:3].copy())
        smoothed_B_list.append(kf_B.x[:3].copy())

    plot_flight_path_3d(
        actual_positions=[traj_A[::step_samples], traj_B[::step_samples]],
        raw_positions=all_raw_detections,
        smoothed_positions=[smoothed_A_list, smoothed_B_list],
        mic_positions=mics
    )

# --------------------------------------------------------------
def demo_gdop_heatmap():
    """
    Demo evaluating Geometric Dilution of Precision (GDOP)
    :return:
    """
    print("=== ARRAY GEOMETRY EVALUATION: GDOP HEATMAP ===")
    sim = SourceSimulator(fs=48000)
    solver = AcousticSolver(fs=48000)

    # Easily swap this out for ArrayGeometry.circular_array() to see how the blindspots change
    mics = ArrayGeometry.perimeter_array(width=2.0, length=2.0)
    print(f"mics shape: {mics.shape}")
    # Use a generic 250 Hz buzz for the test signal
    test_signal = sim.generate_pollinator_buzz(duration=0.1, f0=250.0)

    plot_spatial_error_heatmap(
        sim=sim,
        solver=solver,
        mic_positions=mics,
        test_signal=test_signal,
        resolution=20  # Bumped up slightly for a smoother plot
    )


# ============================================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bee-Mic Demonstration Scripts")
    parser.add_argument('scenario',
                        choices=['standard', 'flyover', 'multiple_stationary', 'multiple_moving', 'heatmap'],
                        help="Demonstration to run")
    args = parser.parse_args()

    if args.scenario == 'standard':
        demo_standard_flight()
    elif args.scenario == 'flyover':
        demo_flyover_singularity()
    elif args.scenario == 'multiple_stationary':
        demo_multiple_stationary()
    elif args.scenario == 'multiple_moving':
        demo_multiple_moving()
    elif args.scenario == 'heatmap':
        demo_gdop_heatmap()