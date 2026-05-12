# Representations of 3D Rotations: Mathematical Foundations and Comparative Analysis

[![Paper](https://img.shields.io/badge/arXiv-Coming%20Soon-b31b1b.svg)](https://arxiv.org/2605.08086)
[![Python](https://img.shields.io/badge/Python-3.12-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> An investigation of rotation representations for the special orthogonal group SO(3), examining mathematical foundations, computational properties, and practical applications across computer graphics, robotics, and machine learning.

<div align="center">
  <a href="https://aizierjiang.github.io/GimbalLock/" target="_blank">
    <img src="https://img.shields.io/badge/🎮_Interactive_Demo-Gimbal_Lock_Visualization-ff6b6b?style=for-the-badge&logo=webgl&logoColor=white" alt="Gimbal Lock Demo" />
  </a>
  <br>
  <sub>Experience gimbal lock singularities in real-time 3D visualization</sub>
</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Rotation Representations Covered](#rotation-representations-covered)
- [Empirical Evaluation](#empirical-evaluation)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Usage](#usage)
- [Key Results](#key-results)
- [Applications](#applications)
- [Future Directions](#future-directions)
- [Citation](#citation)
- [Contact](#contact)

---

## Overview

Rotations in three-dimensional space are fundamental to many computational fields, from robotic manipulation to viewpoint estimation in computer vision. This work provides a comparative analysis of different representations of the special orthogonal group SO(3), evaluating their:

- Mathematical formulations and algebraic properties
- Continuity and susceptibility to singularities (gimbal lock)
- Computational efficiency and storage requirements
- Interpolation properties and composition operations
- Practical applications across multiple domains

This work combines existing knowledge with reproducible numerical demonstrations.

---

## Key Features

### Comprehensive Coverage
- 7 major rotation representations analyzed in detail
- Mathematical foundations from first principles
- Complete conversion formulas between representations

### Empirical Evaluation
- 1000+ numerical experiments on random rotations
- Haar-uniform sampling from SO(3)
- Sub-microsecond timing precision using Python's `timeit`
- Edge case robustness testing (200+ scenarios)

### Quantitative Metrics
- **Numerical Stability**: Round-trip angular reconstruction error
- **Singularity Behavior**: Performance near gimbal lock and antipodal points
- **Interpolation Quality**: Path length and geodesic deviation
- **Computational Efficiency**: Composition and interpolation timing
- **Robustness**: Failure rate across identity, small/large angles, and edge cases

### Visualizations
- Performance vs. storage tradeoff analysis
- Application suitability correlation matrices
- Radar charts comparing multi-dimensional properties

---

## Rotation Representations Covered

| Representation | Parameters | Storage | Gimbal Lock | Continuity | Best Use Case |
|----------------|------------|---------|-------------|------------|---------------|
| **Euler Angles** | 3 | 24 bytes | ✗ Yes | Discontinuous | Human-readable interfaces |
| **Axis-Angle** | 3 | 24 bytes | At θ=2πk | Mostly continuous | Physics simulation |
| **Quaternions** | 4 | 32 bytes | ✓ No | Antipodal ambiguity | General-purpose (recommended) |
| **Rotation Matrices** | 9 | 72 bytes | ✓ No | Continuous | Theoretical analysis |
| **Exponential Maps** | 3 | 24 bytes | At θ=kπ | Local continuity | Incremental updates |
| **6D Continuous** | 6 | 48 bytes | ✓ No | Fully continuous | Neural networks |
| **Matrix Fisher** | 9 | 72 bytes | ✓ No | Distributional | Uncertainty modeling |

---

## Empirical Evaluation

### Methodology

Our evaluation framework runs comprehensive tests on a standard computing environment:

```
Hardware: Intel Core i7-9700K @ 3.60GHz, 16GB RAM
Software: Python 3.12 with SciPy 1.11.3 and NumPy 1.26.0
```

**Key Evaluation Metrics:**

1. **Numerical Stability** (ε_stab)
   ```
   ε_stab = (1/N) Σ ||log(R̂ᵢ⁻¹Rᵢ)||₂
   ```
   Mean angular reconstruction error over N=1000 trials

2. **Singularity Susceptibility**
   - Gimbal lock testing near β ≈ ±π/2 for Euler angles
   - Antipodal quaternion consistency verification

3. **Interpolation Quality**
   - Path length analysis (K=100 evaluation points)
   - Geodesic deviation: relative error vs. shortest path
   - Derivative continuity (smoothness assessment)

4. **Computational Efficiency**
   - Composition time: average over 1000 trials with warmup
   - Batch processing efficiency (100 rotations simultaneously)

### Performance Highlights

| Representation | Composition Time | Interpolation | Path Quality | ML Score |
|----------------|------------------|---------------|--------------|----------|
| **Quaternions** | 34.25 μs | 41.18 μs | 1.6447 (geodesic) | 0.8 |
| **Exponential Maps** | 19.43 μs | 24.43 μs | 1.6494 | 0.7 |
| **6D Continuous** | 421.95 μs | 454.62 μs | 3.7310 | 0.9 |
| **Rotation Matrices** | 306.07 μs | 343.06 μs | 1.6447 (geodesic) | 0.6 |
| **Euler Angles** | 55.36 μs | 64.74 μs | 1.6494 | 0.3 |

---

## Repository Structure

```
3DRotation/
│
├── get_appl_mtrx.py           # Application suitability matrix generator
├── get_metrics_tab.py         # Metrics table generator  
├── get_storage_comp.py        # Storage and performance comparison generator
├── CITATION.cff               # Citation metadata for GitHub
├── _config.yml                # Jekyll configuration for GitHub Pages
├── requirements.txt           # Python dependencies
├── robots.txt                 # Search engine crawling instructions
├── LICENSE                    # MIT License
├── README.md                  # This file
└── fig/                       # Resulted figures (created by scripts)
    ├── storage_performance.png
    └── application_matrix.png
```

### Key Files

- **`get_appl_mtrx.py`**: Gets the figure for the application suitability matrices
- **`get_metrics_tab.py`**: Comprehensive evaluation framework implementing metrics
- **`get_storage_comp.py`**: Produces performance visualizations

---

## Requirements

### Python Dependencies

```bash
python >= 3.12
numpy >= 1.26.0
scipy >= 1.11.3
pandas >= 2.1.0
matplotlib >= 3.8.0
seaborn >= 0.13.0
```

### Installation

```bash
# Clone the repository
git clone https://github.com/aizierjiang/3DRotation.git
cd 3DRotation

# Install dependencies
pip install -r requirements.txt
# or use:
pip install numpy scipy pandas matplotlib seaborn

# Run evaluation scripts
python get_metrics_tab.py
python get_storage_comp.py
python get_appl_mtrx.py
```

---

## Usage

### Running the Evaluation Scripts

```python
# Get metrics table
python get_metrics_tab.py

# Get storage and performance comparison
python get_storage_comp.py

# Get application suitability matrix
python get_appl_mtrx.py

# Results will be saved to the fig/ directory
```

### Quick Example: Quaternion SLERP

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

def slerp_quaternion(q1, q2, t):
    """Spherical linear interpolation between quaternions"""
    dot = np.clip(np.dot(q1, q2), -1, 1)
    
    # Handle antipodal ambiguity
    if dot < 0:
        q2 = -q2
        dot = -dot
    
    # Compute interpolation
    theta = np.arccos(dot)
    if theta < 1e-3:  # Near-parallel quaternions
        return (1-t)*q1 + t*q2  # Linear interpolation
    
    return (np.sin((1-t)*theta)*q1 + np.sin(t*theta)*q2) / np.sin(theta)

# Usage
r1, r2 = R.random(2)
q1, q2 = r1.as_quat(), r2.as_quat()
q_mid = slerp_quaternion(q1, q2, 0.5)  # Midpoint rotation
```

---

## Key Results

### Main Findings

1. **Quaternions for General Use**
   - Best balance: 34.25 μs composition, 32 bytes storage
   - No gimbal lock, efficient SLERP interpolation
   - Works well for robotics, graphics, and navigation

2. **6D Continuous for Machine Learning**
   - High ML compatibility score (0.9)
   - Fully continuous parameterization works well with gradient descent
   - Trade-off: slower composition than quaternions

3. **Exponential Maps for Real-Time Updates**
   - Fast composition (19.43 μs)
   - Direct connection to angular velocities
   - Works well for physics engines and IMU integration

4. **Probabilistic Methods for Uncertainty**
   - Matrix Fisher: measures uncertainty in sensor fusion
   - Bingham: handles antipodal symmetry in quaternion distributions
   - Applicable in autonomous systems

### Performance vs. Storage Tradeoff

```text
Storage (bytes)    Composition Time (μs)
    24        ───  Exponential (19.43)
    24        ───  Axis-Angle (35.78)
    32        ───  Quaternion (34.25)
    48        ───  6D (421.95)
    72        ───  Matrix (306.07)
```

---

## Applications

### Computer Graphics & Animation

- Quaternion SLERP prevents gimbal lock in camera paths
- Smooth character animation with constant angular velocity
- Used in Unity, Unreal Engine, Blender

### Robotics & Navigation

- SLAM systems: Quaternions for compact pose graphs
- IMU integration: Exponential maps for gyroscope updates
- Sensor fusion: Matrix Fisher distributions for uncertainty

### Machine Learning & Vision

- Neural pose estimation: 6D continuous representations
- 3D object recognition: Rotation-equivariant architectures
- Structure-from-Motion: Rotation averaging with exponential maps

### 3D Shape Registration

- Horn's algorithm: Closed-form quaternion solution for ICP
- Point cloud alignment with optimal rotation recovery
- Efficient nearest-neighbor correspondence updates

---

## Future Directions

### Research Opportunities

1. **Hybrid Representations**
   - Adaptive selection based on runtime performance
   - Combine quaternion efficiency with 6D continuity

2. **Standardized Benchmarking**
   - Public repositories with reference implementations
   - Canonical test problems across diverse scenarios

3. **Learning-Based Selection**
   - Meta-learning for automatic representation choice
   - Problem-specific optimization predictions

4. **Geometric Deep Learning**
   - Native SO(3) operations in neural architectures
   - Equivariant networks with Wigner D-matrices

5. **Quantum Computing**
   - Rotation representations for quantum algorithms
   - Quantum rotation gates and adiabatic evolution

---

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{rotation2025,
  title={Representations of 3D Rotations: Mathematical Foundations and Comparative Analysis},
  author={Aizierjiang Aiersilan, Haochen Liu, James Hahn},
  journal={arXiv preprint},
  year={2025},
  url={https://arxiv.org/2605.08086}
}
```

---

## Key References

- Euler (1776): Formulae generales pro translatione quacunque corporum rigidorum
- Hamilton (1840): On a new species of imaginary quantities connected with a theory of quaternions
- Shoemake (1985): Animating rotation with quaternion curves
- Zhou et al. (2019): On the continuity of rotation representations in neural networks
- Mohlin et al. (2020): Probabilistic orientation estimation with matrix Fisher distributions

---

## Contact

For questions or collaboration opportunities, please open an issue in this repository.

---

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/Aizierjiang/3DRotation/blob/main/LICENSE) file for details.

---

## Acknowledgments

This research builds on work in:

- Computer graphics and animation
- Robotics and control theory
- Differential geometry and Lie group theory
- Machine learning and computer vision

