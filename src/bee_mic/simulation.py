# src/bee_mic/simulation.py
"""
Generate artificial audio signal data for testing localization

"""
import numpy as np
import scipy.signal as signal
import scipy.interpolate as interpolate

class SourceSimulator:
    """Generates synthetic multi-channel audio for pollinator tracking."""

    def __init__(self, fs=44100, c_sound=343.0):
        # 44.1 kHz is sufficient for insects, unlike the 250+ kHz needed for bats.
        self.fs = fs
        self.c_sound = c_sound

    # ==========================
    # SIGNAL GENERATION
    # ==========================
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
    # SIGNAL PROPAGATION
    # ==========================================
    def simulate_propagation(self,
                             source_pos,
                             mic_positions,
                             clean_sig,
                             snr_db=20,
                             jitter_sec=0.0):
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
    def simulate_moving_propagation(self,
                                    trajectory,
                                    mic_positions,
                                    source_signal,
                                    snr_db=20):
        """
        Simulates acoustic propagation for a moving source.
        Physically models Doppler shifts by interpolating exact sample arrival times,
        and applies dynamic 1/r amplitude attenuation as the source moves.

        trajectory: [N, 3] array where N matches len(source_signal)
        """
        import scipy.interpolate as interpolate

        num_mics = mic_positions.shape[0]
        n_samples = len(source_signal)

        # Time vector for when each sample was emitted by the bee
        t_emit = np.arange(n_samples) / self.fs

        # Determine how much to pad the final audio based on the furthest possible distance
        max_dist = np.max(np.linalg.norm(trajectory[:, np.newaxis, :] - mic_positions[np.newaxis, :, :], axis=2))
        max_delay_samples = int(np.ceil((max_dist / self.c_sound) * self.fs)) + int(self.fs * 0.1)
        out_samples = n_samples + max_delay_samples

        # Time vector for the uniform recording grid of the microphones
        t_mic = np.arange(out_samples) / self.fs
        received_signals = np.zeros((num_mics, out_samples))

        for i in range(num_mics):
            # 1. Calculate changing distance to this specific microphone over time
            distances = np.linalg.norm(trajectory - mic_positions[i], axis=1)

            # 2. Calculate the exact, non-integer time each emitted sample arrives
            t_arrive = t_emit + (distances / self.c_sound)

            # 3. Apply changing amplitude (louder when closer)
            amplitudes = 1.0 / np.clip(distances, 0.05, None)
            arriving_signal = source_signal * amplitudes

            # 4. Interpolate the stretched/compressed arriving signal onto the mic's uniform clock
            # This perfectly replicates the Doppler shift
            interpolator = interpolate.interp1d(
                t_arrive, arriving_signal,
                kind='linear', bounds_error=False, fill_value=0.0
            )
            clean_audio = interpolator(t_mic)

            # 5. Add ambient noise
            active_audio = clean_audio[clean_audio != 0]
            if len(active_audio) > 0:
                sig_power = np.var(active_audio)
                noise_power = sig_power / (10 ** (snr_db / 10))
                noise = np.random.normal(0, np.sqrt(noise_power), out_samples)
                received_signals[i] = clean_audio + noise

        return received_signals