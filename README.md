# Quantum Assisted Simulator (QAS) for Many-Body Localization

A comprehensive implementation of the Quantum Assisted Simulator algorithm for studying many-body localization (MBL) in disordered Ising chains. This project implements and extends the QAS method from Bharti & Haug (Physical Review A 104, 042418, 2021), providing tools for simulating quantum dynamics on both gate-based quantum computers and classical simulators.

## Overview

Many-body localization is a disorder-induced phase transition that prevents thermalization in isolated quantum systems. This project combines quantum simulation techniques with classical computation to study MBL phenomena in disordered Ising chains, leveraging both Qiskit's statevector simulators and QuTiP's quantum dynamics solvers.

## Project Structure

The project is organized into two main computational approaches:

### 1. Qiskit Statevector (`qiskit_statevector/`)

Implementation of QAS using Qiskit's statevector simulator for systems with open boundary conditions. This approach enables direct simulation of quantum circuits and provides tools for analyzing quantum state evolution.

**Key components:**
- `qas_statevec_4.py`, `qas_statevec_6.py`, `qas_statevec_8.py` - QAS implementations for different system sizes (L=4, 6, 8)
- `plot_qas_vs_exact_*.py` - Comparison of QAS results against exact diagonalization
- `plot_statevec_*.py` - Visualization of quantum state properties
- `plot_from_summaries.py` - Analysis of aggregated simulation data
- `plot_mbl_qas_results.py` - Specialized plots for MBL transition analysis
- `plot_circuit.ipynb` - Interactive visualization of quantum circuits
- `td_qiskit.ipynb` - Jupyter notebook demonstrating time-dependent quantum dynamics

### 2. QuTiP Closed BC (`qutip_closedbc/`)

Implementation of QAS using QuTiP for periodic boundary conditions, including advanced calibration tools and SLURM job scripts for high-performance computing clusters.

**Key components:**
- `qas_true_disordered_closedbc.py` - Core QAS implementation with periodic boundaries
- `calibrate_qas_closedbc.py` - K-scan calibration tool for optimizing quantum parameters
- `run_qas_closedbc_single.py`, `run_qas_closedbc_l10_k8_n1000.py` - Individual simulation runners
- `merge_qas_chunks.py` - Utility for combining distributed simulation results
- `plot_*.py` - Comprehensive visualization and analysis scripts
- `qas_fig3_l10_check.py` - Verification script for published results
- `slurm_*.sh` - Batch job submission scripts for HPC clusters

## Key Features

- **Multi-System Sizes**: Analysis across system sizes L=4 to L=11 for comprehensive disorder-averaged statistics
- **Comparison Tools**: Direct comparison of QAS results with exact diagonalization (ED)
- **Disorder-Averaged Dynamics**: Study of magnetization evolution across different disorder realizations
- **K-Scan Calibration**: Automated tool for optimizing Trotter-step parameters
- **HPC Integration**: SLURM scripts for efficient distributed computation on supercomputing clusters
- **Visualization Suite**: Publication-quality plotting tools for quantum dynamics and phase transitions

## Physics Model

The simulations study a disordered Ising model with:

- **Hamiltonian**: Contains nearest-neighbor (J) and next-nearest-neighbor (J') Ising couplings
- **Disorder**: Random fields drawn from uniform distribution [-h_max, h_max]
- **Transverse Field**: Time-dependent driving field for state preparation and dynamics
- **Boundary Conditions**:
  - Open boundary conditions (Qiskit implementation)
  - Periodic boundary conditions (QuTiP implementation)

This model exhibits a many-body localization transition at critical disorder strength, where the system transitions from ergodic (thermalized) to localized (non-ergodic) behavior.

## Technologies

- **Quantum Simulation**: Qiskit (>=0.45), Qiskit-Aer (>=0.13)
- **QuTiP Dynamics**: QuTiP (>=5.0), QuTiP-QIP (>=0.4)
- **Scientific Computing**: NumPy (>=1.24), SciPy (>=1.10)
- **Visualization**: Matplotlib (>=3.7)
- **High-Performance Computing**: SLURM job scheduling system

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/thiagogiov/quantum-assisted-simulator.git
cd quantum-assisted-simulator
pip install -r requirements.txt
```

For development, consider using a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Qiskit Statevector Simulations

Run individual QAS simulations:

```bash
python qiskit_statevector/qas_statevec_8.py
python qiskit_statevector/plot_qas_vs_exact_8.py
```

Explore interactive notebooks:

```bash
jupyter notebook qiskit_statevector/plot_circuit.ipynb
jupyter notebook qiskit_statevector/td_qiskit.ipynb
```

### QuTiP Closed BC Simulations

Calibrate QAS parameters:

```bash
python qutip_closedbc/calibrate_qas_closedbc.py
```

Run single simulations:

```bash
python qutip_closedbc/run_qas_closedbc_single.py
```

Submit batch jobs to HPC clusters:

```bash
sbatch qutip_closedbc/slurm_qas_l11_k3_array.sh
sbatch qutip_closedbc/slurm_merge_qas_l11_k3.sh
```

Generate analysis plots:

```bash
python qutip_closedbc/plot_mz_vs_dj_closedbc_qas.py
python qutip_closedbc/plot_compare_qas_true_vs_exact_l11.py
```

## Output Files

The simulations generate:

- `*.npz` files: Binary NumPy archives containing quantum state data and observables
- `.png` files: Publication-quality plots (managed by .gitignore)
- `results_*/` directories: Organized simulation output (managed by .gitignore)

## References

This project is based on:

- **Bharti, K., & Haug, T.** (2021). "Quantum Assisted Simulator." *Physical Review A*, 104, 042418.
- Further extensions building on Bharti & Haug methodology for MBL studies

## Author

**Thiago Girao** - PhD candidate in Physics, researching quantum information and quantum computing.

---

For questions or contributions, please open an issue or pull request.
