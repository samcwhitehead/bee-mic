# src/bee_mic/visualization.py
"""
Plotting funcitons for bioacoustics

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------
def plot_3d_geometry(mic_positions,
                     true_pos,
                     est_pos,
                     title="3D Array Localization"):
    """
    Plots the microphone array, true target position(s), and solver estimate(s).
    Accepts either a single [x, y, z] coordinate or an (N, 3) array of coordinates.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Normalize inputs to 2D arrays of shape (N, 3) so single and multi-target share logic
    true_pts = np.atleast_2d(true_pos)
    est_pts = np.atleast_2d(est_pos)

    # 1. Plot Microphones
    ax.scatter(mic_positions[:, 0], mic_positions[:, 1], mic_positions[:, 2],
               c='black', marker='^', s=100, label='Microphones')

    # 2. Plot Ground Truth(s)
    if len(true_pts) > 0:
        ax.scatter(true_pts[:, 0], true_pts[:, 1], true_pts[:, 2],
                   c='green', marker='*', s=200, label='True Position')
        for i, pt in enumerate(true_pts):
            ax.text(pt[0], pt[1], pt[2] + 0.04, f' True {i+1}', color='green', fontweight='bold')

    # 3. Plot Estimate(s)
    if len(est_pts) > 0:
        ax.scatter(est_pts[:, 0], est_pts[:, 1], est_pts[:, 2],
                   c='red', marker='X', s=150, label='Estimated Position')
        for i, pt in enumerate(est_pts):
            ax.text(pt[0], pt[1], pt[2] - 0.05, f' Est {i+1}', color='red')

    # 4. Draw Error Vectors (pairing each estimate to its nearest ground truth)
    if len(true_pts) > 0 and len(est_pts) > 0:
        dist_matrix = cdist(est_pts, true_pts)
        nearest_true_indices = np.argmin(dist_matrix, axis=1)

        for est_idx, true_idx in enumerate(nearest_true_indices):
            t_pt = true_pts[true_idx]
            e_pt = est_pts[est_idx]
            # Label only the first line so 'Error Vector' doesn't duplicate in the legend
            label = 'Error Vector' if est_idx == 0 else None
            ax.plot([t_pt[0], e_pt[0]], [t_pt[1], e_pt[1]], [t_pt[2], e_pt[2]],
                    'r--', alpha=0.5, label=label)

    # Draw a faint 2x2m ground reference plane at Z=0
    ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], [0, 0, 0, 0, 0],
            color='gray', linestyle=':', alpha=0.4)

    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title(title)
    ax.legend()
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()



# ------------------------------------------------------------------------------------------
def plot_cross_correlation(received_signals, fs, ref_mic_idx=0,
                           target_mic_idx=1, bandpass=None, use_gcc_phat=True):
    """
    Visualizes the cross-correlation function between two microphones.
    Essential for diagnosing cycle-skipping on periodic pollinator wingbeats.
    """
    # Apply bandpass if requested
    if bandpass is not None:
        sos = signal.butter(4, bandpass, btype='bandpass', fs=fs, output='sos')
        proc_signals = signal.sosfiltfilt(sos, received_signals, axis=1)
    else:
        proc_signals = received_signals

    ref_sig = proc_signals[ref_mic_idx]
    tar_sig = proc_signals[target_mic_idx]

    # Calculate standard cross-correlation
    n = len(ref_sig)
    n_fft = 1 << (2 * n - 1).bit_length()
    REF_FFT = np.fft.rfft(ref_sig, n=n_fft)
    TAR_FFT = np.fft.rfft(tar_sig, n=n_fft)

    cross_spec = TAR_FFT * np.conj(REF_FFT)

    # Apply Generalized Cross-Correlation with Phase Transform (GCC-PHAT)
    if use_gcc_phat:
        mag = np.abs(cross_spec)
        # Add a small regularization term (epsilon) to avoid division by zero on silent frames
        cross_spec = cross_spec / (mag + 0.05 * np.max(mag))

    corr = np.fft.fftshift(np.fft.irfft(cross_spec, n=n_fft))
    lags = np.arange(-n_fft // 2, n_fft // 2) / fs * 1000  # in milliseconds

    # Zoom in on the central +/- 10 ms
    mask = (lags >= -10) & (lags <= 10)

    fig = plt.figure(figsize=(10, 4))
    plt.plot(lags[mask], corr[mask], color='blue')

    # Highlight the peak
    peak_idx = np.argmax(corr[mask])
    peak_lag = lags[mask][peak_idx]
    plt.axvline(peak_lag, color='red', linestyle='--', alpha=0.7, label=f'Peak: {peak_lag:.2f} ms')

    plt.title(f"Cross-Correlation (Mic {ref_mic_idx} vs Mic {target_mic_idx}) | GCC-PHAT: {use_gcc_phat}")
    plt.xlabel("Time Lag (ms)")
    plt.ylabel("Correlation Amplitude")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------------------------
def plot_spatial_error_heatmap(sim, solver, mic_positions, test_signal, z_height=0.55,
                               grid_size=2.0, resolution=15, snr_db=15,
                               bandpass=(150, 1200), jitter_sec=50e-6):
    """
    Maps spatial GDOP by querying the simulator across a grid.
    """
    print(f"\nGenerating {resolution}x{resolution} heatmap... (this may take a few seconds)")

    x_vals = np.linspace(-grid_size, grid_size, resolution)
    y_vals = np.linspace(-grid_size, grid_size, resolution)
    X, Y = np.meshgrid(x_vals, y_vals)
    error_grid = np.zeros_like(X)

    for i in range(resolution):
        for j in range(resolution):
            pos = np.array([X[i, j], Y[i, j], z_height])

            # 1. Generate audio using the simulator
            rx, _ = sim.simulate_stationary_propagation(pos, mic_positions, test_signal,
                                                        snr_db=snr_db, jitter_sec=jitter_sec)

            # 2. Process audio using the DSP solver
            tdoas = solver.estimate_tdoa(rx, mic_positions=mic_positions,
                                         bandpass=bandpass, use_gcc_phat=True)

            est = solver.multilateration(mic_positions, tdoas, bounds=([-5.0, -5.0, 0.0], [5.0, 5.0, 5.0]))

            error_grid[i, j] = np.linalg.norm(pos - est) * 100

    plt.figure(figsize=(9, 7))

    max_err = max(5.0, np.max(error_grid))
    levels = np.linspace(0, max_err, 26)

    cp = plt.contourf(X, Y, error_grid, levels=levels, cmap='viridis_r', extend='max')
    plt.colorbar(cp, label="Localization Error (cm)")

    plt.scatter(mic_positions[:, 0], mic_positions[:, 1], c='red', marker='^', s=100, label='Microphones')
    plt.plot([-1.0, 1.0, 1.0, -1.0, -1.0], [-1.0, -1.0, 1.0, 1.0, -1.0], 'r--', alpha=0.5, label='2x2m Patch')

    plt.title(f"Spatial Error (SNR {snr_db}dB + {jitter_sec * 1e6:.0f}µs Jitter, Z={z_height}m)")
    plt.xlabel("X Position (meters)")
    plt.ylabel("Y Position (meters)")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------------------------
def plot_srp_phat_slices(grid_data, est_positions, true_positions=None, mic_positions=None):
    """
    Slices the 3D SRP-PHAT acoustic power grid at the Z-height of each estimated target
    to visualize the spatial hotspots.
    """
    X, Y, Z, power_3d = grid_data
    z_coords = Z[0, 0, :]  # Extract 1D array of Z grid coordinates

    num_targets = len(est_positions)
    fig, axes = plt.subplots(1, num_targets, figsize=(6 * num_targets, 5))

    # Handle single target case (axes is not an iterable)
    if num_targets == 1:
        axes = [axes]

    for i, est_pos in enumerate(est_positions):
        ax = axes[i]

        # Find the Z-index closest to the estimated target's height
        est_z = est_pos[2]
        z_idx = np.argmin(np.abs(z_coords - est_z))
        actual_z = z_coords[z_idx]

        # Extract the 2D power slice at that Z-height
        power_slice = power_3d[:, :, z_idx]
        X_slice = X[:, :, z_idx]
        Y_slice = Y[:, :, z_idx]

        # Normalize the slice for consistent color mapping
        slice_min, slice_max = np.min(power_slice), np.max(power_slice)
        norm_slice = (power_slice - slice_min) / (slice_max - slice_min + 1e-12)

        # Plot the acoustic heatmap
        cp = ax.contourf(X_slice, Y_slice, norm_slice, levels=30, cmap='inferno')
        plt.colorbar(cp, ax=ax, label="Normalized Acoustic Power")

        # Plot the estimated position
        ax.scatter(est_pos[0], est_pos[1], c='cyan', marker='X', s=150,
                   edgecolor='black', label='Estimated Target')

        # Overlay True Positions if provided
        if true_positions is not None:
            for true_pos in true_positions:
                # Only plot true positions that are roughly at this Z-height
                if abs(true_pos[2] - actual_z) < 0.15:
                    ax.scatter(true_pos[0], true_pos[1], c='lime', marker='*', s=200,
                               edgecolor='black', label='True Target')

        # Overlay Microphones if provided
        if mic_positions is not None:
            ax.scatter(mic_positions[:, 0], mic_positions[:, 1], c='white', marker='^',
                       s=100, edgecolor='black', label='Microphones')

            # Draw the 2x2m patch boundary
            ax.plot([-1.0, 1.0, 1.0, -1.0, -1.0], [-1.0, -1.0, 1.0, 1.0, -1.0],
                    'w--', alpha=0.5, label='Array Boundary')

        ax.set_title(f"Target {i + 1} Hotspot (Z = {actual_z:.2f} m)")
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.legend(loc='upper right', fontsize='small')

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------
def _normalize_trajectories(data):
    """
    Normalizes input into a list of 2D arrays of shape (T, 3).
    Handles:
      - A single trajectory: shape (T, 3) or list of [x, y, z]
      - Multiple trajectories: list of (T, 3) arrays or 3D array (num_bees, T, 3)
    """
    if data is None or len(data) == 0:
        return []

    arr = np.asarray(data)  # , dtype=object)
    # Single trajectory passed as (T, 3) numeric array or list of 3-element points
    if arr.ndim == 2 and arr.shape[1] == 3 and np.issubdtype(arr.dtype, np.number):
        return [arr.astype(float)]

    # Multiple trajectories (or ragged list of trajectories)
    trajectories = []
    for item in data:
        traj = np.asarray(item, dtype=float)
        if traj.ndim == 2 and traj.shape[1] == 3 and len(traj) > 0:
            trajectories.append(traj)
    return trajectories


# -----------------------------------------------------------------------------------------------
def plot_flight_path_3d(actual_positions, raw_positions, smoothed_positions, mic_positions=None):
    """
    Plots 3D flight trajectories comparing ground truth(s), raw acoustic detections,
    and Kalman-smoothed path(s). Supports single or multiple targets.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Distinct colors for multi-target ground truth vs smoothed tracks
    true_colors = ['green', 'darkgreen', 'limegreen', 'olive']
    smooth_colors = ['blue', 'darkorange', 'purple', 'cyan']

    # 1. Plot True Trajectories
    actual_trajs = _normalize_trajectories(actual_positions)
    for idx, actual in enumerate(actual_trajs):
        color = true_colors[idx % len(true_colors)]
        label = f'True Trajectory {idx + 1}' if len(actual_trajs) > 1 else 'True Trajectory'

        ax.plot(actual[:, 0], actual[:, 1], actual[:, 2],
                label=label, color=color, linewidth=2, linestyle='--')
        ax.scatter(actual[0, 0], actual[0, 1], actual[0, 2], color=color, marker='o', s=50)

        start_tag = f' Start {idx + 1}' if len(actual_trajs) > 1 else ' Start'
        ax.text(actual[0, 0], actual[0, 1], actual[0, 2], start_tag, color=color, fontweight='bold')

    # 2. Plot Raw SRP-PHAT Detections (as an unlinked point cloud)
    if raw_positions is not None and len(raw_positions) > 0:
        raw = np.atleast_2d(np.asarray(raw_positions, dtype=float))
        ax.scatter(raw[:, 0], raw[:, 1], raw[:, 2],
                   label='Raw SRP-PHAT', color='red', alpha=0.45, marker='x', s=40)
        # Only connect raw points with a faint line when tracking a single target
        if len(actual_trajs) <= 1 and len(_normalize_trajectories(smoothed_positions)) <= 1:
            ax.plot(raw[:, 0], raw[:, 1], raw[:, 2], color='red', alpha=0.2, linewidth=1)

    # 3. Plot Kalman Smoothed Paths
    smoothed_trajs = _normalize_trajectories(smoothed_positions)
    for idx, smoothed in enumerate(smoothed_trajs):
        color = smooth_colors[idx % len(smooth_colors)]
        label = f'Kalman Smoothed {idx + 1}' if len(smoothed_trajs) > 1 else 'Kalman Smoothed'

        ax.plot(smoothed[:, 0], smoothed[:, 1], smoothed[:, 2],
                label=label, color=color, linewidth=3)
        ax.scatter(smoothed[:, 0], smoothed[:, 1], smoothed[:, 2],
                   color=color, s=20)

    # 4. Plot Microphones and Ground Plane
    if mic_positions is not None:
        mics = np.asarray(mic_positions, dtype=float)
        ax.scatter(mics[:, 0], mics[:, 1], mics[:, 2],
                   label='Microphones', color='black', marker='^', s=100)

        # Draw a faint 2x2m box at Z=0 to represent the garden patch
        ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], [0, 0, 0, 0, 0],
                color='gray', linestyle=':', alpha=0.5)

    ax.set_title('3D Flight Path Tracking: Raw vs Smoothed')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.legend(loc='upper left')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------------------------
def count_pollinators_visually(received_signals, fs, ref_mic_idx=0, max_freq=800):
    """
    Generates a high-resolution spectrogram of the reference microphone
    so you can visually count the distinct wingbeat frequency tracks.
    """
    ref_audio = received_signals[ref_mic_idx]

    # Use a large window (e.g., 4096 samples at 48kHz = ~85ms)
    # to get very high frequency resolution for separating closely pitched bees
    nperseg = min(4096, len(ref_audio))

    freqs, times, Sxx = signal.spectrogram(
        ref_audio,
        fs=fs,
        window='hann',
        nperseg=nperseg,
        noverlap=nperseg // 2,
        scaling='density'
    )

    # Convert to log scale (dB) for visualization
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    # Filter to the relevant insect frequencies
    freq_mask = freqs <= max_freq
    freqs = freqs[freq_mask]
    Sxx_db = Sxx_db[freq_mask, :]

    plt.figure(figsize=(10, 5))
    # cmap='magma' or 'inferno' highlights the distinct harmonic tracks best
    plt.pcolormesh(times, freqs, Sxx_db, shading='gouraud', cmap='magma')

    plt.title(f"Mic {ref_mic_idx} Spectrogram: Count the distinct horizontal tracks")
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (seconds)')

    # Add subtle grid lines to make estimating the bandpass frequencies easier
    plt.grid(color='white', alpha=0.2, linestyle='--')

    plt.colorbar(label='Power Spectral Density (dB/Hz)')
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------------------------
def plot_gcc_phat_diagnostic(chunk, fs, mic_A=0, mic_B=1):
    """
    Computes and plots the GCC-PHAT cross-correlation specifically to diagnose
    peak degradation (smearing, splitting) caused by Doppler shift or rapid
    Time-Difference-Of-Arrival (TDOA) changes during the window.
    """
    sig_A = chunk[mic_A]
    sig_B = chunk[mic_B]

    n = len(sig_A)
    n_fft = 1 << (2 * n - 1).bit_length()

    # Compute FFTs
    fft_A = np.fft.rfft(sig_A, n=n_fft)
    fft_B = np.fft.rfft(sig_B, n=n_fft)

    # Cross-power spectrum
    cross_spec = fft_A * np.conj(fft_B)

    # PHAT weighting (Phase Transform) with small epsilon to prevent div by zero
    mag = np.abs(cross_spec)
    cross_spec_phat = cross_spec / (mag + 1e-10)

    # Inverse FFT and shift to center zero-lag
    gcc = np.fft.fftshift(np.fft.irfft(cross_spec_phat, n=n_fft))

    # Calculate time lags in milliseconds
    lags = np.arange(-n_fft // 2, n_fft // 2) / fs * 1000

    # Zoom in tightly to +/- 5 ms (for small arrays, true TDOA is usually < 2ms)
    mask = (lags >= -5) & (lags <= 5)

    plt.figure(figsize=(10, 5))
    plt.plot(lags[mask], gcc[mask], color='crimson', linewidth=1.5, label='Degraded GCC-PHAT')

    # Highlight the maximum peak found
    peak_idx = np.argmax(gcc[mask])
    peak_lag = lags[mask][peak_idx]
    peak_val = gcc[mask][peak_idx]

    plt.axvline(peak_lag, color='black', linestyle='--', alpha=0.6,
                label=f'Apparent Peak: {peak_lag:.2f} ms')
    plt.scatter([peak_lag], [peak_val], color='black', zorder=5)

    plt.title(
        f"Flyover Singularity Diagnostic (Mic {mic_A} vs Mic {mic_B})\nNotice the widened base or split peaks caused by rapid TDOA shift")
    plt.xlabel("Time Lag (ms)")
    plt.ylabel("Correlation Amplitude")

    # Add a zero-line for reference
    plt.axhline(0, color='gray', linewidth=0.5, alpha=0.5)

    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()