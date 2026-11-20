# src/bee_mic/geometry.py
import numpy as np
import json
from pathlib import Path


class ArrayGeometry:
    """Generates and manages 3D microphone array coordinates."""

    @staticmethod
    def perimeter_array(width=2.0,
                        length=2.0,
                        low_z=0.15,
                        high_z=0.85,
                        add_center_mic=True):
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

    @staticmethod
    def circular_array(radius=1.0, num_mics=6, z_height=0.5, include_center=True):
        """Expands testing to circular/hexagonal geometries."""
        angles = np.linspace(0, 2 * np.pi, num_mics, endpoint=False)
        mics = [[radius * np.cos(a), radius * np.sin(a), z_height] for a in angles]

        if include_center:
            mics.insert(0, [0.0, 0.0, z_height])

        return np.array(mics)

    @staticmethod
    def tetrahedral(side_length=1.0):
        """Compact 3D array (Verified for Bats)"""
        return np.array([
            [0, 0, side_length],
            [2 * np.sqrt(2) * side_length / 3, 0, -side_length / 3],
            [-np.sqrt(2) * side_length / 3, np.sqrt(6) * side_length / 3, -side_length / 3],
            [-np.sqrt(2) * side_length / 3, -np.sqrt(6) * side_length / 3, -side_length / 3]
        ])

    @staticmethod
    def load_from_json(filepath):
        """Loads real-world tape-measured or acoustically calibrated coordinates."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return np.array(data['mic_positions'])

    @staticmethod
    def save_to_json(filepath, mic_positions, notes=""):
        """Saves calibrated numpy coordinates to a JSON config file."""
        import json
        from datetime import datetime

        # Build the dictionary payload
        data = {
            "calibration_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "notes": notes,
            "num_mics": len(mic_positions),
            "mic_positions": mic_positions.tolist()  # Convert numpy array to nested lists
        }

        # Ensure the target directory exists
        from pathlib import Path
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)

        print(f"Saved {len(mic_positions)} mic coordinates to {filepath}")