# src/bee_mic/dsp.py
import numpy as np
import scipy.signal as signal
import scipy.optimize as optimize
import scipy.ndimage as ndimage
import itertools


# ======================================
class AcousticSolver:
    """Digital Signal Processing methods for spatial localization."""

    def __init__(self, fs=48000, c_sound=343.0):
        self.fs = fs
        self.c_sound = c_sound


    # ----------------------
    def calibrate_array(self,
                        rough_mic_positions,
                        known_pop_locations,
                        pop_audio_clips):
        """
        Uses least-squares optimization to refine microphone coordinates based on
        the Time Difference of Arrival (TDOA) of impulsive sounds (ballon pop) at known locations.

        rough_mic_positions: [num_mics, 3] array of estimated positions
        known_pop_locations: list of [X, Y, Z] coordinates for each balloon pop
        pop_audio_clips: list of audio arrays [num_mics, samples], one for each pop
        """


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
    def _evaluate_grid(self, bounds, step, mic_positions, mic_pairs, gcc_funcs, center_idx, n_fft):
        """
        Helper that generates a 3D grid within `bounds` at `step` spacing
        and accumulates steered GCC-PHAT power across all valid mic pairs.
        """
        # make grid
        x_min, y_min, z_min = bounds[0]
        x_max, y_max, z_max = bounds[1]

        x = np.arange(x_min, x_max + step, step)
        y = np.arange(y_min, y_max + step, step)
        z = np.arange(z_min, z_max + step, step)

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

        # Calculate distances from every grid point to every microphone
        distances = np.linalg.norm(
            grid_points[:, np.newaxis, :] - mic_positions[np.newaxis, :, :], axis=2
        )
        power_grid = np.zeros(grid_points.shape[0])

        for pair_idx, (i, j) in enumerate(mic_pairs):
            # Calculate theoretical TDOA for this specific grid point and mic pair
            tdoa = (distances[:, j] - distances[:, i]) / self.c_sound
            lag_samples = np.round(tdoa * self.fs).astype(int)

            # Sum the GCC-PHAT correlation energy at that specific lag
            indices = np.clip(center_idx + lag_samples, 0, n_fft - 1)
            power_grid += gcc_funcs[pair_idx, indices]

        # Reshape power grid back to 3D matrix (X, Y, Z)
        return x, y, z, X, Y, Z, power_grid.reshape(X.shape)


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

        # --- DYNAMIC SENSOR EXCLUSION ---
        # Calculate the RMS power of each microphone's audio chunk
        rms_levels = np.sqrt(np.mean(proc_signals ** 2, axis=1))
        median_rms = np.median(rms_levels)

        valid_mics = []
        for i in range(num_mics):
            # If a mic is more than 4x louder than the median, the bee is practically
            # sitting on it. Exclude it to prevent Doppler/Delay smearing.
            if rms_levels[i] > 4.0 * median_rms:
                # print(f"  [Auto-Excluding Mic {i} - Flyover Singularity detected]")
                pass
            else:
                valid_mics.append(i)

        # Only create GCC-PHAT pairs from the valid, un-smeared microphones
        mic_pairs = list(itertools.combinations(valid_mics, 2))
        # --------------------------------

        # 3. Compute GCC-PHAT for EVERY unique pair of microphones (not just vs reference)
        # mic_pairs = list(itertools.combinations(range(num_mics), 2))
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

        # 4. and 5. Generate the 3D scanning grid and steer the array
        x, y, z, X, Y, Z, power_3d = self._evaluate_grid(
            bounds, resolution, mic_positions, mic_pairs, gcc_funcs, center_idx, n_fft
        )

        # 6. Extract targets using Local Maxima and Dynamic Thresholding

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
    def srp_phat_hierarchical(self, received_signals, mic_positions, bounds=([-1.0, -1.0, 0.0], [1.0, 1.0, 1.0]),
                            resolution=0.05, coarse_resolution=0.20, bandpass=None, num_targets=2,
                            exclusion_radius=0.4):
        """
        Hierarchical 2-stage SRP-PHAT solver:
          Stage 1 (Coarse): Scans the full volume at `coarse_resolution` to find candidate peaks.
          Stage 2 (Fine):   Refines each candidate peak within a local bounding box at `resolution`.
        """
        import itertools

        # 1. Bandpass filter
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

        # 3. Dynamic sensor exclusion for flyover singularities
        rms_levels = np.sqrt(np.mean(proc_signals ** 2, axis=1))
        median_rms = np.median(rms_levels)
        valid_mics = [i for i in range(num_mics) if rms_levels[i] <= 4.0 * median_rms]
        mic_pairs = list(itertools.combinations(valid_mics, 2))

        # 4. Precompute GCC-PHAT lookup tables once
        gcc_funcs = []
        for i, j in mic_pairs:
            cross_spec = ffts[j] * np.conj(ffts[i])
            mag = np.abs(cross_spec)
            cross_spec = cross_spec / (mag + 0.05 * np.max(mag))
            corr = np.fft.fftshift(np.fft.irfft(cross_spec, n=n_fft))
            gcc_funcs.append(corr)

        gcc_funcs = np.array(gcc_funcs)
        center_idx = n_fft // 2

        # --- STAGE 1: COARSE SEARCH ---
        effective_coarse = max(coarse_resolution, resolution)
        cx, cy, cz, cX, cY, cZ, coarse_power = self._evaluate_grid(
            bounds, effective_coarse, mic_positions, mic_pairs, gcc_funcs, center_idx, n_fft
        )

        # Scale neighborhood filter size to coarse step (covers ~40-60cm neighborhood)
        filter_size = max(3, int(round(exclusion_radius / effective_coarse)) | 1)
        local_max_mask = (coarse_power == ndimage.maximum_filter(coarse_power, size=filter_size))
        threshold_mask = (coarse_power > 0.40 * np.max(coarse_power))

        valid_peaks_indices = np.argwhere(local_max_mask & threshold_mask)
        peak_powers = [coarse_power[tuple(idx)] for idx in valid_peaks_indices]
        sorted_order = np.argsort(peak_powers)[::-1]

        coarse_candidates = []
        for i in range(min(num_targets, len(sorted_order))):
            idx = valid_peaks_indices[sorted_order[i]]
            coarse_candidates.append(np.array([cx[idx[0]], cy[idx[1]], cz[idx[2]]]))

        # If coarse_resolution wasn't coarser than target resolution, skip Stage 2
        if effective_coarse <= resolution:
            return np.array(coarse_candidates), (cX, cY, cZ, coarse_power)

        # --- STAGE 2: FINE REFINEMENT ---
        # Search +/- 1.5 coarse grid cells around each candidate peak, clamped to room bounds
        half_box = effective_coarse * 1.5
        b_min = np.array(bounds[0], dtype=float)
        b_max = np.array(bounds[1], dtype=float)

        refined_targets = []
        for cand in coarse_candidates:
            sub_min = np.maximum(b_min, cand - half_box)
            sub_max = np.minimum(b_max, cand + half_box)

            fx, fy, fz, _, _, _, fine_power = self._evaluate_grid(
                (sub_min, sub_max), resolution, mic_positions, mic_pairs, gcc_funcs, center_idx, n_fft
            )

            best_idx = np.unravel_index(np.argmax(fine_power), fine_power.shape)
            refined_pos = np.array([fx[best_idx[0]], fy[best_idx[1]], fz[best_idx[2]]])
            refined_power = fine_power[best_idx]
            refined_targets.append((refined_power, refined_pos))

        # Re-sort by refined peak power (a narrow sharp peak can overtake a broad coarse peak)
        refined_targets.sort(key=lambda item: item[0], reverse=True)
        estimated_positions = [pos for _, pos in refined_targets]

        if len(estimated_positions) < num_targets:
            print(f"Warning: Only found {len(estimated_positions)} distinct targets above the noise threshold.")

        # Returns the coarse 3D grid as the diagnostic payload so callers unpacking (X, Y, Z, power_3d) still work
        return np.array(estimated_positions), (cX, cY, cZ, coarse_power)


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
                res = optimize.least_squares(objective_function, guess, method=method, **kwargs)
                if best_result is None or res.cost < best_result.cost:
                    best_result = res
            except ValueError:
                # Skip guesses that fall outside provided bounds
                continue

        return best_result.x
