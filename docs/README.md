# GF-FM: 2nd-Order Flow Matching with Geometric Fabrics Guidance

**Standalone Implementation** - No Isaac Lab dependencies required.

## Overview

GF-FM integrates **Geometric Fabrics (FABRICS)** into **Flow Matching (FM)** inference to provide physics-based safety guarantees (obstacle avoidance, joint limits) for generative motion planning.

- **Generative Model:** 2nd-order Flow Matching ($q, \dot{q}$) trained on optimal trajectories.
- **Safety Layer:** Geometric Fabrics repulsion field injected during ODE integration.
- **Platform:** Fully standalone PyTorch implementation (compatible with Isaac Lab but does not depend on it).

---

## Documentation

- **[Usage Guide (USAGE.md)](./USAGE.md):** Detailed instructions for data collection, training, inference, and visualization.
- **[Theoretical Background (THEORY.md)](./THEORY.md):** In-depth explanation of the vector field interpretation, ODE-coupled guidance, and physics integration.

---

## Quick Start

### 1. Install Dependencies
```bash
pip install torch numpy h5py pyyaml tqdm termcolor warp-lang>=1.5.0 urdfpy pybullet
```

### 2. Run Inference (Pre-trained)
```bash
# Basic inference with guidance
./isaaclab.sh -p scripts/gf_fm/run/run_inference.py \
    --checkpoint scripts/gf_fm/logs/best_model.pth \
    --guidance_scale 1.0
```

### 3. Data Collection & Training
See [USAGE.md](./USAGE.md) for full training pipeline instructions.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         GF-FM Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   cuRobo     │───▶│   HDF5       │───▶│   Training   │      │
│  │  MotionGen   │    │  Dataset     │    │   (FM)       │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                 │               │
│                                                 ▼               │
│                                          ┌──────────────┐       │
│                                          │  Inference   │       │
│                                          │  + GF Guide  │       │
│                                          └──────────────┘       │
│                                                 │               │
│                                    FABRICS ─────┘               │
│                                  (Joint Limit                   │
│                                   Repulsion)                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
scripts/gf_fm/
├── README.md                          # This file
├── docs/                              # Documentation
│   ├── USAGE.md                       # Usage guide
│   ├── THEORY.md                      # Theoretical background
│   └── archive/                       # Archived design docs
│
├── run/                               # Execution scripts
│   ├── generate_dataset.py            # Data collection (cuRobo)
│   ├── train.py                       # Training script
│   ├── run_inference.py               # Inference with GF guidance
│   └── visualize.py                   # Visualization
│
├── guidance/
│   ├── gf_guidance_field.py           # Main guidance interface
│   ├── franka_fabric.py               # Franka-specific fabric
│   ├── body_sphere_repulsion.py       # Warp kernel for repulsion
│   └── guidance_integrator.py         # ODE integration logic
│
└── policy/
    └── second_order_flow_policy.py    # FM Policy implementation
```

## License

Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
