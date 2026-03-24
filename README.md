# DR-MPC Guide-Dog / VIP Simulator

This repository is a modified version of the original DR-MPC project for socially aware navigation. The main extension in this fork is a guide-dog and visually impaired person (VIP) paired simulator with harness dynamics, pair-aware safety checks, and richer video visualization.

## Original Reference

- Original DR-MPC repository: https://github.com/James-R-Han/DR-MPC
- Paper: https://arxiv.org/abs/2410.10646
- IEEE publication: https://ieeexplore.ieee.org/document/10904316

## Demo Video

GitHub may render the video inline depending on the viewer. If not, open the file directly.

<video src="docs/media/vid_0.mp4" controls width="900"></video>

[Open demo video](docs/media/vid_0.mp4)

## What This Simulator Does

The simulator combines two subsystems:

1. Human avoidance (HA): the dog must move safely through nearby pedestrians.
2. Path tracking (PT): the dog must still stay inside the target corridor and progress toward the goal.

In this modified version, the dog is no longer simulated alone. A VIP agent is attached behind and to the right of the dog through a simplified harness model. The simulator updates the VIP state each step, checks safety for both dog and VIP, and renders both agents in the recorded video.

## Repository Layout

- `environment/`: simulator environments
- `scripts/`: training, policy, model, and config code
- `configs/guide_dog_params.yaml`: guide-dog / VIP simulator parameters
- `docs/media/vid_0.mp4`: demo video used in this README

## Local Installation

### 1. Clone this repository

```bash
git clone https://github.com/INHA-Artemis/DRMPC-HRI.git
cd DRMPC-HRI
```

### 2. Create the conda environment

```bash
conda env create -f environment.yml
conda activate social_navigation
```

If your machine uses a different CUDA or Torch setup, adjust the Torch-related packages in `environment.yml` or install compatible versions manually.

### 3. Install system packages

```bash
sudo apt update
sudo apt install -y build-essential cmake ffmpeg git
```

### 4. Install Python-RVO2

```bash
git clone https://github.com/sybrenstuvel/Python-RVO2.git
cd Python-RVO2
python setup.py build
python setup.py install
cd ..
```

### 5. Install pysteam

```bash
git clone https://github.com/utiasASRL/pysteam.git
```

### 6. Set `PYTHONPATH`

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Run The Code

Train and simulate with:

```bash
python scripts/online_continuous_task.py
```

Compare multi-run outputs with:

```bash
python scripts/compare_training_multirun.py
```

Generated videos and plots are typically written under `HA_and_PT_results/`.

## Docker Installation

A Docker setup is included in this fork.

### 1. Build the image

```bash
docker build -t drmpc-hri .
```

### 2. Run the training script inside Docker

```bash
docker run --rm -it \
  -v "$(pwd)":/opt/drmpc \
  -w /opt/drmpc \
  drmpc-hri \
  conda run --no-capture-output -n social_navigation python scripts/online_continuous_task.py
```

### 3. Open an interactive shell inside Docker

```bash
docker run --rm -it \
  -v "$(pwd)":/opt/drmpc \
  -w /opt/drmpc \
  drmpc-hri \
  bash
```

If you want GPU access, add your Docker GPU runtime options such as `--gpus all` on a compatible NVIDIA setup.

## What Was Modified

### Core Python files

- `environment/human_avoidance/subENVs/crowd_sim.py`
  - Added a separate VIP agent.
  - Added guide-pair state such as harness tension and VIP offsets.
  - Added VIP synchronization and harness-dynamics update logic.
  - Added gait variability, heading smoothing, no-backward motion constraints, and pair-length constraints.
  - Extended human spawning and ORCA-style avoidance checks to consider both dog and VIP.

- `environment/human_avoidance/human_avoidance_env.py`
  - Added a `no_pivot` option to suppress in-place dog turning at very low speed.
  - Updated the step logic so the VIP state is updated after the dog moves.
  - Extended collision and minimum-distance checks from dog-only to dog-or-VIP.
  - Synced the VIP pose at episode reset.

- `environment/HA_and_PT/human_avoidance_and_path_tracking_env.py`
  - Stored VIP state in the recorded trajectory.
  - Added VIP path-corridor and safety-corridor penalties.
  - Added soft-reset behavior that temporarily relaxes the no-pivot constraint.
  - Added rendering of the orange VIP circle and the harness line.
  - Added video overlays for dog speed, VIP speed, VIP acceleration, harness roll angle, and harness tension.

- `scripts/configs/config_HA.py`
  - Added YAML loading from `configs/guide_dog_params.yaml`.
  - Added the `guide` configuration block for pair geometry, tension, gait, and rendering settings.

- `scripts/configs/config_training.py`
  - Reduced `save_freq` from `50 * 250` to `10 * 250` so results are saved more frequently.

- `scripts/online_continuous_task.py`
  - Forced headless Matplotlib backend with `Agg` for non-GUI environments.
  - Delayed training updates until both warm-up and batch-size conditions are satisfied.

- `scripts/models/utils.py`
  - Simplified the GRU forward pass and explicitly called `flatten_parameters()` before encoding to avoid repeated warnings and improve cuDNN compatibility.

### Added support files

- `configs/guide_dog_params.yaml`
  - Centralized the guide-dog / VIP simulator parameters in YAML.

- `requirements_pip.txt`
  - Added a direct pip requirements list for environments where you want a plain pip-based install flow.

- `docs/media/vid_0.mp4`
  - Added a tracked demo video for the README.

## How The Modified Simulator Works

At every simulation step:

1. The dog receives a navigation action.
2. The HA environment optionally reduces excessive in-place pivoting at low speed.
3. The VIP is moved by a simplified harness-following model rather than by an independent navigation policy.
4. Human avoidance safety is checked against the combined dog-VIP pair.
5. Path tracking penalties are applied if either the dog or the VIP leaves the allowed corridor.
6. Video logging stores both trajectories and diagnostic values for playback.

## Guide-Dog Parameter Summary

All pair-specific parameters are stored in `configs/guide_dog_params.yaml`.

- `enable_pair`: turns the dog-VIP pair model on or off
- `vip_radius`: radius of the orange VIP agent
- `offset_right_m`, `offset_back_m`: default VIP position relative to the dog
- `offset_right_min_m`, `offset_back_min_m`: lower bounds that keep the VIP on the right-rear side
- `rod_length_m`, `rod_angle_deg_min`, `rod_angle_deg_max`: define the projected harness-length limits
- `person_mass_kg`: effective VIP mass in the simplified harness model
- `spring_k`, `damping_c`: spring-damper coefficients for harness tension
- `max_tension_n`, `min_tension_n`: lower and upper tension bounds
- `min_pair_clearance_m`: extra clearance between dog and VIP
- `max_pull_step_m`: maximum pull distance applied in one step
- `follow_gain`: how strongly the VIP follows the dog-side anchor point
- `vip_speed_max_mps`, `vip_acc_max_mps2`: VIP speed and acceleration caps
- `heading_tau_s`: smoothing time constant for VIP heading alignment
- `dog_no_pivot_min_v_mps`: low-speed threshold used to suppress aggressive in-place turning
- `vip_gait_*`: parameters that randomize stride amplitude, stride frequency, and gait-speed variation

## Notes

- This fork documents and exposes the paired simulator logic; it does not replace the original DR-MPC paper or upstream implementation.
- The upstream project should still be credited when this fork is used or extended.
