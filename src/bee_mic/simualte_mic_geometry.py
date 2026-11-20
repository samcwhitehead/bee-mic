import numpy as np
import scipy.signal as signal
from scipy.optimize import least_squares
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ============================================
##  HELPER FUNCTIONS
# ============================================
def run_verification_suite(tracker, mic_array, test_signal, bandpass=None):
    """
    Validates solver math, TDOA accuracy, statistical bias, and spatial GDOP
    before trusting the simulator for new array geometries.
    """
    print("=== 1. DETERMINISTIC & ORACLE CHECKS (NOISELESS) ===")
    test_pos = np.array([1.8, -0.6, 1.2])

    # Compute exact analytical ground-truth TDOAs
    dists = np.linalg.norm(mic_array - test_pos, axis=1)
    true_tdoas = (dists - dists[0]) / tracker.c_sound

    # Test A: Multilateration solver alone (fed exact TDOAs)
    solver_only_pos = tracker.multilateration(mic_array, true_tdoas)
    solver_err = np.linalg.norm(test_pos - solver_only_pos)
    print(f"Solver-only error (exact TDOAs):  {solver_err * 1e6:.4f} µm  [Target: < 1 µm]")

    # Test B: Full pipeline at ultra-high SNR (100 dB = virtually noiseless)
    rx_clean, _ = tracker.simulate_propagation(test_pos, mic_array, test_signal, snr_db=100)
    est_tdoas = tracker.estimate_tdoa(rx_clean, ref_mic_idx=0, bandpass=bandpass)
    tdoa_err_us = np.max(np.abs(true_tdoas - est_tdoas)) * 1e6

    full_clean_pos = tracker.multilateration(mic_array, est_tdoas)
    full_clean_err = np.linalg.norm(test_pos - full_clean_pos)
    print(f"Max TDOA error (100 dB SNR):      {tdoa_err_us:.4f} µs   [Target: < 0.1 µs]")
    print(f"Full pipeline error (100 dB SNR): {full_clean_err * 1000:.4f} mm  [Target: < 1.0 mm]")

    print("\n=== 2. MONTE CARLO NOISE & BIAS CHECK (50 TRIALS @ 20 dB SNR) ===")
    estimates = []
    for _ in range(50):
        rx_noisy, _ = tracker.simulate_propagation(test_pos, mic_array, test_signal, snr_db=20)
        tdoas = tracker.estimate_tdoa(rx_noisy, ref_mic_idx=0, bandpass=bandpass)
        estimates.append(tracker.multilateration(mic_array, tdoas))

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

        rx, _ = tracker.simulate_propagation(pos, mic_array, test_signal, snr_db=100)
        tdoas = tracker.estimate_tdoa(rx, ref_mic_idx=0, bandpass=bandpass)
        est = tracker.multilateration(mic_array, tdoas)

        # Decompose error into Angular (bearing) vs Radial (range) components
        est_dir = est / np.linalg.norm(est)
        angle_err_deg = np.degrees(np.arccos(np.clip(np.dot(direction, est_dir), -1.0, 1.0)))
        range_err_cm = abs(np.linalg.norm(est) - r) * 100
        print(f"Range {r:3.1f} m -> Bearing Error: {angle_err_deg:.4f}° | Range Error: {range_err_cm:.3f} cm")


# ==========================================
## VISUALIZATION DIAGNOSTICS
# ==========================================
def plot_3d_geometry(mic_positions, true_pos, est_pos, title="3D Array Localization"):
    """
    Plots the microphone array, the true target position, and the solver's estimate.
    Helps visualize spatial relationships and Z-axis (height) ambiguity.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Microphones
    ax.scatter(mic_positions[:, 0], mic_positions[:, 1], mic_positions[:, 2],
               c='black', marker='^', s=100, label='Microphones')

    # Plot Ground Truth
    ax.scatter(true_pos[0], true_pos[1], true_pos[2],
               c='green', marker='*', s=200, label='True Position')

    # Plot Estimate
    ax.scatter(est_pos[0], est_pos[1], est_pos[2],
               c='red', marker='X', s=150, label='Estimated Position')

    # Draw a line between true and estimated to highlight error magnitude
    ax.plot([true_pos[0], est_pos[0]], [true_pos[1], est_pos[1]], [true_pos[2], est_pos[2]],
            'r--', alpha=0.5, label='Error Vector')

    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title(title)
    ax.legend()
    plt.show()


def plot_cross_correlation(tracker, received_signals, ref_mic_idx=0,
                           target_mic_idx=1, bandpass=None, use_gcc_phat=True):
    """
    Visualizes the cross-correlation function between two microphones.
    Essential for diagnosing cycle-skipping on periodic pollinator wingbeats.
    """
    # Apply bandpass if requested
    if bandpass is not None:
        sos = signal.butter(4, bandpass, btype='bandpass', fs=tracker.fs, output='sos')
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

    if use_gcc_phat:
        mag = np.abs(cross_spec)
        cross_spec = cross_spec / (mag + 0.05 * np.max(mag))

    corr = np.fft.fftshift(np.fft.irfft(cross_spec, n=n_fft))
    lags = np.arange(-n_fft // 2, n_fft // 2) / tracker.fs * 1000  # in milliseconds

    # Zoom in on the central +/- 10 ms
    mask = (lags >= -10) & (lags <= 10)

    plt.figure(figsize=(10, 4))
    plt.plot(lags[mask], corr[mask], color='blue')
    plt.title(f"Cross-Correlation (Mic {ref_mic_idx} vs Mic {target_mic_idx}) | GCC-PHAT: {use_gcc_phat}")
    plt.xlabel("Time Lag (ms)")
    plt.ylabel("Correlation Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_spatial_error_heatmap(tracker, mic_positions, test_signal, z_height=0.55,
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

            # Pass jitter directly into the physics simulator
            rx, _ = tracker.simulate_propagation(pos, mic_positions, test_signal,
                                                 snr_db=snr_db, jitter_sec=jitter_sec)

            tdoas = tracker.estimate_tdoa(rx, mic_positions=mic_positions,
                                          bandpass=bandpass, use_gcc_phat=True)

            est = tracker.multilateration(mic_positions, tdoas, bounds=([-5.0, -5.0, 0.0], [5.0, 5.0, 5.0]))

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


# ============================================
## CLASS DEFINITION
# ============================================
class InsectLocalizationArray:
    """
    A Python port of the core Array WAH multilateration pipeline,
    adapted for pollinator insect wingbeats and sonication buzzes.
    """
    # ---------------------------------------------------------------------
    def __init__(self, fs=44100, c_sound=343.0):
        # 44.1 kHz is sufficient for insects, unlike the 250+ kHz needed for bats.
        self.fs = fs
        self.c_sound = c_sound

    # ==========================================
    # 1. GEOMETRY METHODS
    # ==========================================
    def setup_tetrahedral_array(self, radius=0.15):
        """Compact 3D array (Verified for Bats)"""
        return np.array([
            [0, 0, radius],
            [2 * np.sqrt(2) * radius / 3, 0, -radius / 3],
            [-np.sqrt(2) * radius / 3, np.sqrt(6) * radius / 3, -radius / 3],
            [-np.sqrt(2) * radius / 3, -np.sqrt(6) * radius / 3, -radius / 3]
        ])


    def setup_perimeter_array(self, width=2.0, length=2.0, low_z=0.15, high_z=0.85, add_center_mic=True):
        """Staggered 3D perimeter array framing a patch (For Pollinators)"""
        hx, hy = width / 2.0, length / 2.0
        mics = [
            [-hx, -hy, low_z],  # Mic 0 (Reference): Southwest, low
            [hx, -hy, high_z],  # Mic 1: Southeast, elevated
            [hx, hy, low_z],  # Mic 2: Northeast, low
            [-hx, hy, high_z],  # Mic 3: Northwest, elevated
        ]
        if add_center_mic:
            mics.append([0.0, 0.0, (low_z + high_z) / 2.0])
        return np.array(mics, dtype=float)

    def calibrate_array(self, rough_mic_positions, known_pop_locations, pop_audio_clips):
        """
        Uses least-squares optimization to refine microphone coordinates based on
        the Time Difference of Arrival (TDOA) of impulsive sounds at known locations.

        rough_mic_positions: [num_mics, 3] array of estimated positions
        known_pop_locations: list of [X, Y, Z] coordinates for each balloon pop
        pop_audio_clips: list of audio arrays [num_mics, samples], one for each pop
        """
        import scipy.optimize as optimize

        num_mics = rough_mic_positions.shape[0]
        num_pops = len(known_pop_locations)
        observed_tdoas = []

        # 1. Extract the exact arrival times of the balloon pops
        for k in range(num_pops):
            audio = pop_audio_clips[k]
            arrival_samples = np.argmax(np.abs(audio), axis=1)

            # Convert sample differences to time differences (seconds) relative to Mic 0
            tdoas_sec = (arrival_samples - arrival_samples[0]) / self.fs
            observed_tdoas.append(tdoas_sec)

        observed_tdoas = np.array(observed_tdoas)

        # 2. Define the Error Function for the optimizer
        def tdoa_error(flat_mics):
            # Reshape the flat array back into [num_mics, 3] coordinates
            mics = flat_mics.reshape(num_mics, 3)
            errors = []

            for k in range(num_pops):
                pop_pos = known_pop_locations[k]

                # Calculate theoretical distances from this pop to all mics
                distances = np.linalg.norm(mics - pop_pos, axis=1)

                # Convert to theoretical TDOAs relative to Mic 0
                theoretical_tdoas = (distances - distances[0]) / self.c_sound

                # The error is the difference between theoretical and real audio delays
                errors.extend(theoretical_tdoas - observed_tdoas[k])

            # To prevent the whole array from mathematically "floating" away,
            # we force Mic 0 to stay exactly where the user measured it.
            anchor_error = np.linalg.norm(mics[0] - rough_mic_positions[0]) * 10
            errors.append(anchor_error)

            return errors

        # 3. Run the Optimization
        # Flatten the initial guesses into a 1D array for the solver
        initial_guess = rough_mic_positions.flatten()

        print("Optimizing microphone coordinates...")
        result = optimize.least_squares(tdoa_error, initial_guess, method='lm')

        refined_positions = result.x.reshape(num_mics, 3)
        return refined_positions


    # ==========================================
    # 2. SIGNAL GENERATORS
    # ==========================================
    def generate_bat_call(self, duration=0.005, f_start=80000, f_end=30000, method='logarithmic'):
        """
        Generates a downward frequency-modulated (FM) echolocation chirp,
        matching the standard acoustic model in Array_WAH.

        Parameters:
        - duration: Call length in seconds (typically 2-10 ms for FM bats).
        - f_start: Starting ultrasonic frequency in Hz (e.g., 80 kHz).
        - f_end: Ending frequency in Hz (e.g., 30 kHz).
        - method: 'linear' or 'logarithmic' (hyperbolic/exponential sweep).
        """
        if self.fs < 2 * max(f_start, f_end):
            raise ValueError(f"Sampling rate ({self.fs} Hz) violates Nyquist limit for {max(f_start, f_end)} Hz call.")

        t = np.linspace(0, duration, int(self.fs * duration), endpoint=False)

        # Generate fundamental FM sweep
        sig = signal.chirp(t, f0=f_start, f1=f_end, t1=duration, method=method)

        # Add a weaker second harmonic (common in real bat calls, if within Nyquist limit)
        if self.fs > 2 * (2 * f_start):
            sig += 0.3 * signal.chirp(t, f0=f_start * 2, f1=f_end * 2, t1=duration, method=method)

        # Apply a Tukey (tapered cosine) window to shape the call envelope smoothly
        sig *= signal.windows.tukey(len(sig), alpha=0.25)

        # Zero-pad with 2 ms of silence before and after
        pad_len = int(self.fs * 0.002)
        return np.pad(sig, (pad_len, pad_len), 'constant')

    # ---------------------------------------------------------------------
    def generate_pollinator_buzz(self,
                                 duration=0.15,
                                 f0=235.0,
                                 fm_jitter_hz=8.0):
        """
        Simulates a realistic bee/pollinator flight buzz:
        - Fundamental wingbeat (180-280 Hz) + strong 2nd, 3rd, 4th harmonics
        - Slight natural frequency modulation (flight maneuvers) which breaks pure
          periodicity and helps cross-correlation lock onto the true delay.
        """
        t = np.linspace(0, duration, int(self.fs * duration), endpoint=False)

        # Slight organic pitch drift during flight (helps prevent cycle-skipping)
        freq_drift = fm_jitter_hz * np.sin(2 * np.pi * 6.0 * t)
        phase = 2 * np.pi * np.cumsum(f0 + freq_drift) / self.fs

        # Harmonic series typical of Apis / Bombus wingbeats
        sig = (
                1.00 * np.sin(phase) +
                0.75 * np.sin(2 * phase + 0.4) +
                0.45 * np.sin(3 * phase + 1.1) +
                0.25 * np.sin(4 * phase + 2.0)
        )

        sig *= signal.windows.tukey(len(sig), alpha=0.2)
        pad_len = int(self.fs * 0.02)  # 20 ms silence padding
        return np.pad(sig, (pad_len, pad_len), 'constant')

    # ==========================================
    # 3. VERIFIED PHYSICS CORE
    # ==========================================
    def simulate_propagation(self, source_pos, mic_positions, clean_sig, snr_db=20, jitter_sec=0.0):
        """Verified FFT exact phase-shift propagation with optional environmental jitter"""
        received_signals = []
        distances = np.linalg.norm(mic_positions - source_pos, axis=1)
        delays_sec = distances / self.c_sound

        # Inject physical path jitter (wind, thermal gradients) before waveform generation
        if jitter_sec > 0:
            delays_sec += np.random.normal(0, jitter_sec, len(delays_sec))
            # Ensure no negative delays in edge cases
            delays_sec = np.maximum(delays_sec, 0.0)

        max_delay_samples = int(np.ceil(np.max(delays_sec) * self.fs))
        total_len = len(clean_sig) + max_delay_samples + 10

        padded_sig = np.pad(clean_sig, (0, total_len - len(clean_sig)))
        SIG_FFT = np.fft.rfft(padded_sig)
        freqs = np.fft.rfftfreq(total_len, d=1.0 / self.fs)

        active_rms_1m = np.sqrt(np.mean(clean_sig[clean_sig != 0] ** 2))

        for dist, tau in zip(distances, delays_sec):
            amplitude = 1.0 / max(dist, 0.05)
            phase_shift = np.exp(-1j * 2 * np.pi * freqs * tau)
            rx = np.fft.irfft(SIG_FFT * phase_shift, n=total_len) * amplitude

            sig_power_ref = (active_rms_1m / 1.0) ** 2
            noise_power = sig_power_ref / (10 ** (snr_db / 10.0))
            noise = np.random.normal(0, np.sqrt(noise_power), total_len)
            received_signals.append(rx + noise)

        return np.array(received_signals), distances


    # ------------------------------------------------------------------------------------------------------
    def estimate_tdoa(self,
                      received_signals,
                      ref_mic_idx=0,
                      bandpass=None,
                      upsample_factor=32,
                      use_gcc_phat=False,
                      mic_positions=None):
        """Verified FFT upsampled cross-correlation, extended for pollinators"""
        num_mics = received_signals.shape[0]
        tdoas = np.zeros(num_mics)

        if bandpass is not None:
            sos = signal.butter(4, bandpass, btype='bandpass', fs=self.fs, output='sos')
            proc_signals = signal.sosfiltfilt(sos, received_signals, axis=1)
        else:
            proc_signals = received_signals

        ref_sig = proc_signals[ref_mic_idx]
        n = len(ref_sig)
        n_fft = 1 << (2 * n - 1).bit_length()
        REF_FFT = np.fft.rfft(ref_sig, n=n_fft)
        n_fft_up = n_fft * upsample_factor

        for i in range(num_mics):
            if i == ref_mic_idx:
                continue

            SIG_FFT = np.fft.rfft(proc_signals[i], n=n_fft)
            cross_spec = SIG_FFT * np.conj(REF_FFT)

            # Feature Extension: GCC-PHAT whitening for periodic insect buzzes
            if use_gcc_phat:
                mag = np.abs(cross_spec)
                floor = 0.05 * np.max(mag)
                cross_spec = cross_spec / (mag + floor)

            cross_spec_up = np.zeros(n_fft_up // 2 + 1, dtype=complex)
            cross_spec_up[:len(cross_spec)] = cross_spec
            corr_up = np.fft.fftshift(np.fft.irfft(cross_spec_up, n=n_fft_up))
            center = n_fft_up // 2

            # Feature Extension: Clamping search window to physical array bounds
            if mic_positions is not None:
                max_dist = np.linalg.norm(mic_positions[i] - mic_positions[ref_mic_idx])
                max_lag_up = int(np.ceil((max_dist / self.c_sound) * self.fs * upsample_factor * 1.1))
                search_window = corr_up[center - max_lag_up: center + max_lag_up + 1]
                peak_idx = np.argmax(search_window) + (center - max_lag_up)
            else:
                peak_idx = np.argmax(corr_up)

            lag_samples = (peak_idx - center) / upsample_factor
            tdoas[i] = lag_samples / self.fs

        return tdoas

    # ------------------------------------------------------------------------------------------------------
    def srp_phat(self, received_signals, mic_positions, bounds=([-1.0, -1.0, 0.0], [1.0, 1.0, 1.0]),
                 resolution=0.05, bandpass=None, num_targets=2, exclusion_radius=0.4):
        """
        Scans a 3D volume to find multiple acoustic sources simultaneously
        by mapping spatial acoustic power.
        """
        import itertools

        # 1. Filter signals
        if bandpass is not None:
            sos = signal.butter(4, bandpass, btype='bandpass', fs=self.fs, output='sos')
            proc_signals = signal.sosfiltfilt(sos, received_signals, axis=1)
        else:
            proc_signals = received_signals

        num_mics = proc_signals.shape[0]
        n_samples = proc_signals.shape[1]
        n_fft = 1 << (2 * n_samples - 1).bit_length()

        # 2. Compute FFTs once for all microphones
        ffts = [np.fft.rfft(proc_signals[i], n=n_fft) for i in range(num_mics)]

        # 3. Compute GCC-PHAT for EVERY unique pair of microphones (not just vs reference)
        mic_pairs = list(itertools.combinations(range(num_mics), 2))
        gcc_funcs = []

        for i, j in mic_pairs:
            cross_spec = ffts[j] * np.conj(ffts[i])
            mag = np.abs(cross_spec)
            floor = 0.05 * np.max(mag)
            cross_spec = cross_spec / (mag + floor)

            corr = np.fft.fftshift(np.fft.irfft(cross_spec, n=n_fft))
            gcc_funcs.append(corr)

        gcc_funcs = np.array(gcc_funcs)
        center_idx = n_fft // 2

        # 4. Generate the 3D scanning grid
        x_min, y_min, z_min = bounds[0]
        x_max, y_max, z_max = bounds[1]

        x = np.arange(x_min, x_max + resolution, resolution)
        y = np.arange(y_min, y_max + resolution, resolution)
        z = np.arange(z_min, z_max + resolution, resolution)

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

        # 5. Steer the array (Vectorized distance and lag calculation)
        # Calculate distances from every grid point to every microphone
        distances = np.linalg.norm(grid_points[:, np.newaxis, :] - mic_positions[np.newaxis, :, :], axis=2)
        power_grid = np.zeros(grid_points.shape[0])

        for pair_idx, (i, j) in enumerate(mic_pairs):
            # Calculate theoretical TDOA for this specific grid point and mic pair
            tdoa = (distances[:, j] - distances[:, i]) / self.c_sound
            lag_samples = np.round(tdoa * self.fs).astype(int)

            # Sum the GCC-PHAT correlation energy at that specific lag
            indices = np.clip(center_idx + lag_samples, 0, n_fft - 1)
            power_grid += gcc_funcs[pair_idx, indices]

        # 6. Extract targets using Local Maxima and Dynamic Thresholding
        # Reshape power grid back to 3D matrix (X, Y, Z)
        power_3d = power_grid.reshape(X.shape)

        # A. Find true local peaks (must be the highest point in a surrounding 3D box)
        # size=9 on a 5cm grid means it checks a 45cm neighborhood
        local_max_mask = (power_3d == ndimage.maximum_filter(power_3d, size=9))

        # B. Apply a dynamic threshold (e.g., must be at least 60% as strong as the loudest peak)
        global_max = np.max(power_3d)
        threshold_mask = (power_3d > 0.40 * global_max)

        # C. Find grid indices where BOTH conditions are true
        valid_peaks_indices = np.argwhere(local_max_mask & threshold_mask)

        # D. Sort the valid peaks by their acoustic power (loudest first)
        peak_powers = [power_3d[tuple(idx)] for idx in valid_peaks_indices]
        sorted_order = np.argsort(peak_powers)[::-1]

        estimated_positions = []
        for i in range(min(num_targets, len(sorted_order))):
            idx = valid_peaks_indices[sorted_order[i]]
            # Convert matrix indices back to physical spatial coordinates
            pos = np.array([x[idx[0]], y[idx[1]], z[idx[2]]])
            estimated_positions.append(pos)

        # Optional: Print warning if it couldn't find enough distinct peaks
        if len(estimated_positions) < num_targets:
            print(f"Warning: Only found {len(estimated_positions)} distinct targets above the noise threshold.")

        return np.array(estimated_positions), (X, Y, Z, power_3d)


    # ------------------------------------------------------------------------------------------------------
    def cancel_interferer(self, received_signals, mic_positions, interferer_pos):
        """
        Uses frequency-domain Delay-and-Sum beamforming to estimate the clean waveform
        of a known loud target, then subtracts its acoustic footprint from the raw audio.
        """
        num_mics = received_signals.shape[0]
        n_samples = received_signals.shape[1]

        # 1. Calculate propagation delays from the known interferer to each mic
        distances = np.linalg.norm(mic_positions - interferer_pos, axis=1)
        delays_sec = distances / self.c_sound

        ffts = np.fft.rfft(received_signals, axis=1)
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / self.fs)

        # 2. Steer array backwards to estimate the clean source waveform (Delay-and-Sum)
        aligned_ffts = []
        for i in range(num_mics):
            # Advance phase to cancel propagation delay, shifting source to t=0
            phase_shift = np.exp(1j * 2 * np.pi * freqs * delays_sec[i])
            # Multiply by distance to reverse the 1/r amplitude attenuation
            aligned_ffts.append(ffts[i] * phase_shift * max(distances[i], 0.05))

        # Average the aligned signals to isolate Bee A (Bee B is smeared out)
        estimated_source_fft = np.mean(aligned_ffts, axis=0)

        # 3. Project the estimated waveform back to the mics and subtract
        residual_signals = np.zeros_like(received_signals)
        for i in range(num_mics):
            # Retard phase to simulate forward propagation to the mic
            phase_shift = np.exp(-1j * 2 * np.pi * freqs * delays_sec[i])
            # Re-apply 1/r amplitude attenuation
            amplitude = 1.0 / max(distances[i], 0.05)

            mic_interferer_fft = estimated_source_fft * phase_shift * amplitude
            mic_interferer_audio = np.fft.irfft(mic_interferer_fft, n=n_samples)

            # Mathematically cancel the interferer from the raw audio
            residual_signals[i] = received_signals[i] - mic_interferer_audio

        return residual_signals


    # ------------------------------------------------------------------------------------------------------
    def multilateration(self, mic_positions, tdoas, ref_mic_idx=0, initial_guess=None, bounds=None):
        """Verified Multi-start Levenberg-Marquardt, extended to support TRF bounds"""
        ref_pos = mic_positions[ref_mic_idx]

        def objective_function(guess):
            d_ref = np.linalg.norm(guess - ref_pos)
            errors = []
            for i, mic in enumerate(mic_positions):
                if i == ref_mic_idx:
                    continue
                d_i = np.linalg.norm(guess - mic)
                errors.append(((d_i - d_ref) / self.c_sound) - tdoas[i])
            return np.array(errors)

        if initial_guess is not None:
            guesses = [np.array(initial_guess, dtype=float)]
        else:
            # Generate guesses covering both the compact bat array and 2m pollinator array
            guesses = [np.array([x, y, z], dtype=float)
                       for x in (-0.5, 0.5, 1.0) for y in (-0.5, 0.5, 1.0) for z in (-0.5, 0.5, 1.0)]

        # Switch solver method safely if bounds are passed in
        method = 'trf' if bounds is not None else 'lm'
        kwargs = {'bounds': bounds} if bounds is not None else {}

        best_result = None
        for guess in guesses:
            try:
                res = least_squares(objective_function, guess, method=method, **kwargs)
                if best_result is None or res.cost < best_result.cost:
                    best_result = res
            except ValueError:
                # Skip guesses that fall outside provided bounds
                continue

        return best_result.x


# ============================================
## MAIN
# ============================================
# --- Execution Example ---
if __name__ == "__main__":
    np.random.seed(47)
    sim = InsectLocalizationArray(fs=48000)
    # pol_mics = sim.setup_perimeter_array(width=2.0, length=2.0)
    pol_mics = sim.setup_tetrahedral_array(radius=0.75)

    print("=== SRP-PHAT: IDENTICAL FREQUENCY OVERLAP ===")

    # 1. Generate two insects with the exact same wingbeat (250 Hz)
    sig_A = sim.generate_pollinator_buzz(duration=2.15, f0=240.0)
    sig_B = sim.generate_pollinator_buzz(duration=2.15, f0=250.0)

    true_pos_A = np.array([0.45, -0.35, 0.55])
    true_pos_B = np.array([-0.60, 0.80, 0.30])

    # 2. Propagate and mix their signals at the array
    rx_A, _ = sim.simulate_propagation(true_pos_A, pol_mics, sig_A, snr_db=15)
    rx_B, _ = sim.simulate_propagation(true_pos_B, pol_mics, sig_B, snr_db=15)

    # Find the maximum sample length between the two signals
    max_len = max(rx_A.shape[1], rx_B.shape[1])

    # Pad the time axis (axis 1) of both arrays with zeros at the end to match max_len
    rx_A_padded = np.pad(rx_A, ((0, 0), (0, max_len - rx_A.shape[1])), 'constant')
    rx_B_padded = np.pad(rx_B, ((0, 0), (0, max_len - rx_B.shape[1])), 'constant')

    # Now they share the exact same dimensions and can be mixed
    rx_mixed = rx_A_padded + rx_B_padded

    # 4. Generate the spectrogram to count the bees manually
    print("Generating spectrogram...")
    count_pollinators_visually(rx_mixed, fs=sim.fs, ref_mic_idx=0, max_freq=800)

    # 3. Run the SRP-PHAT Grid Scanner
    print("Scanning 3D volume (this takes a moment)...")
    est_positions, grid_data = sim.srp_phat(
        rx_mixed,
        mic_positions=pol_mics,
        bounds=([-1.0, -1.0, 0.0], [1.0, 1.0, 1.0]),
        resolution=0.05,  # 5 cm grid spacing
        num_targets=2,
        exclusion_radius=0.4  # Minimum distance between bees
    )

    # 4. Match estimates to ground truth (order doesn't matter)
    for est in est_positions:
        err_A = np.linalg.norm(true_pos_A - est) * 100
        err_B = np.linalg.norm(true_pos_B - est) * 100

        if err_A < err_B:
            print(f"Located Target near Bee A | Error: {err_A:.1f} cm")
        else:
            print(f"Located Target near Bee B | Error: {err_B:.1f} cm")

    # 5. Visualize the SRP-PHAT Spatial Hotspots
    plot_srp_phat_slices(
        grid_data,
        est_positions,
        true_positions=[true_pos_A, true_pos_B],
        mic_positions=pol_mics
    )