import argparse
import numpy as np
from pathlib import Path
from bee_mic.geometry import ArrayGeometry
from bee_mic.dsp import AcousticSolver
from bee_mic.io import load_audio_recording


def run_balloon_calibration(rough_config_path, audio_dir, output_path):
    print("=== ACOUSTIC ARRAY CALIBRATION ===")

    rough_mics = ArrayGeometry.load_from_json(rough_config_path)

    # Define physical pop locations
    pop_locations = [
        np.array([0.0, 0.0, 0.5]),
        np.array([0.5, 0.0, 0.5]),
        np.array([0.0, 0.5, 0.5])
    ]

    pop_audio_clips = []
    audio_dir_path = Path(audio_dir)

    for i in range(len(pop_locations)):
        wav_file = audio_dir_path / f"pop_{i + 1}.wav"
        if not wav_file.exists():
            print(f"Warning: {wav_file} not found. Skipping.")
            continue

        print(f"Loading {wav_file.name}...")
        fs, audio_data = load_audio_recording(wav_file)
        pop_audio_clips.append(audio_data)

    if len(pop_audio_clips) == len(pop_locations):
        solver = AcousticSolver(fs=fs)
        calibrated_mics = solver.calibrate_array(rough_mics, pop_locations, pop_audio_clips)

        ArrayGeometry.save_to_json(
            filepath=output_path,
            mic_positions=calibrated_mics,
            notes="Acoustically calibrated via balloon pops"
        )
    else:
        print("Calibration aborted: Missing audio files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rough', type=str, required=True, help='Path to rough JSON guesses')
    parser.add_argument('--audio', type=str, required=True, help='Directory containing pop WAVs')
    parser.add_argument('--out', type=str, required=True, help='Output JSON path')

    args = parser.parse_args()
    run_balloon_calibration(args.rough, args.audio, args.out)