# src/bee_mic/io.py
import numpy as np
import scipy.io.wavfile as wavfile
from pathlib import Path

def load_audio_recording(filepath):
    """
    Loads a multi-channel WAV file from the real array.
    Returns: fs (int), audio_array (numpy array of shape [num_mics, num_samples])
    """
    fs, data = wavfile.read(filepath)

    # Ensure shape is [channels, samples] for the DSP math
    if data.ndim > 1 and data.shape[1] < data.shape[0]:
        data = data.T

    # Convert integer WAV data to floats between -1.0 and 1.0
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0

    return fs, data


def save_numpy_to_wav(filepath, audio_array, fs):
    """Exports simulated audio blocks to listen to them."""
    # Scale and transpose back to [samples, channels]
    scaled = np.int16(audio_array.T * 32767.0)
    wavfile.write(filepath, fs, scaled)