# bee-mic

3D acoustic localization and multi-target tracking for flying pollinators using microphone arrays.

`bee-mic` provides a modular DSP and simulation framework for localizing and tracking bees in both synthetic testbeds and real-world field environments.

---

## Installation

This project uses [Pixi](https://pixi.sh/) for reproducible dependency and environment management.

1. **Install Pixi** (if you don't already have it):
   ```bash
   curl -fsSL [https://pixi.sh/install.sh](https://pixi.sh/install.sh) | bash
   ```
2. **Clone the repository and install the environment:**
   ```bash
   git clone [https://github.com/samcwhitehead/bee-mic.git](https://github.com/samcwhitehead/bee-mic.git)
   cd bee-mic
   pixi install
   ```

---

## Interactive Exploration (JupyterLab)

To explore array geometries, listen to simulated pollinator audio, and interact with rotatable 3D trajectory plots, launch the included JupyterLab notebook:

```bash
pixi run -e notebooks explore
```

Pixi will automatically provision the isolated `notebooks` environment (`jupyterlab`, `ipympl`, `ipywidgets`) and open `notebooks/01_explore_bee_mic.ipynb`.

---

## Running Demo Simulations

The `scripts/demo_simulations.py` CLI lets you test individual tracking scenarios and benchmarks from the terminal:

```bash
# Single moving bee with Kalman tracking
pixi run demo-standard

# Flyover singularity & Doppler smearing diagnostic
pixi run demo-flyover

# Multiple overlapping targets (stationary or moving)
pixi run demo-multiple_stationary
pixi run demo-multiple_moving
```

---

## Quickstart

```python
import numpy as np
from bee_mic.simulation import SourceSimulator, ArrayGeometry
from bee_mic.dsp import AcousticSolver
from bee_mic.utils import pad_sum

# 1. Configure array and simulator
sim = SourceSimulator(fs=48000)
mics = ArrayGeometry.perimeter_array()

# 2. Simulate two overlapping bees
sig_a = sim.generate_pollinator_buzz(duration=0.2, f0=250.0)
sig_b = sim.generate_pollinator_buzz(duration=0.2, f0=250.0)

rx_a = sim.simulate_stationary_propagation(np.array([-0.5, 0.4, 0.5]), mics, sig_a, snr_db=25)
rx_b = sim.simulate_stationary_propagation(np.array([0.6, -0.3, 0.6]), mics, sig_b, snr_db=25)
rx_combined = pad_sum(rx_a, rx_b)

# 3. Localize with hierarchical SRP-PHAT
solver = AcousticSolver(fs=48000)
chunk = rx_combined[:, :int(0.04 * sim.fs)]
positions, _ = solver.srp_phat_hierarchical(chunk, mics, resolution=0.02, num_targets=2)
print("Estimated 3D positions:\n", positions)
```

---

## Running Tests

Run the `pytest` suite to verify DSP, simulation, and utility modules:

```bash
pixi run pytest
```

---

## Project Structure

```text
bee-mic/
├── notebooks/
│   └── 01_explore_bee_mic.ipynb   # Interactive walkthrough & 3D plots
├── scripts/
│   └── demo_simulations.py        # CLI scenarios and benchmarks
├── src/
│   └── bee_mic/
│       ├── __init__.py
│       ├── dsp.py                 # SRP-PHAT, hierarchical solver
│       ├── geometry.py            # ArrayGeometry
│       ├── io.py                  # Read/write data
│       ├── simulation.py          # SourceSimulator 
│       ├── tracking.py            # Kalman filter implementation
│       ├── utils.py               # Signal mixing (pad_sum) & math helpers
│       └── visualization.py       # 3D trajectories, GDOP, GCC-PHAT diagnostics
├── tests/                         # Unit tests (pytest)
├── pyproject.toml                 # Package metadata & Pixi configuration
└── README.md
```
