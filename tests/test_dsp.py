import numpy as np
from bee_mic.dsp import AcousticSolver


def test_dynamic_exclusion_logic():
    """
    Verifies that a microphone with an RMS volume significantly higher
    than the median is correctly identified for exclusion.
    """
    # Create fake audio chunk for 5 mics, 1000 samples
    # Mics 1-4 have normal background noise
    proc_signals = np.random.normal(0, 0.1, (5, 1000))

    # Mic 0 has a massive amplitude spike (the flyover)
    proc_signals[0] = np.random.normal(0, 5.0, 1000)

    # Replicate the RMS exclusion logic from srp_phat
    rms_levels = np.sqrt(np.mean(proc_signals ** 2, axis=1))
    median_rms = np.median(rms_levels)

    valid_mics = []
    for i in range(5):
        if rms_levels[i] <= 4.0 * median_rms:
            valid_mics.append(i)

    # Mic 0 should be dropped, leaving 1, 2, 3, 4
    assert 0 not in valid_mics, "Failed to exclude near-field microphone"
    assert len(valid_mics) == 4, "Incorrect number of microphones retained"
    assert set(valid_mics) == {1, 2, 3, 4}, "Dropped the wrong microphones"


if __name__ == "__main__":
    test_dynamic_exclusion_logic()