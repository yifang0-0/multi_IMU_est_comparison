# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-IMU joint angle estimation comparison project. See README.md for user documentation, methods list, and results.

## Commands

```bash
# Setup environment
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Download dataset (~5.5GB from SimTK)
python download_simtk_dataset.py

# Run estimation
python run_estimation.py --joint knee --method all --subject Subject08
python run_estimation.py --joint ankle --method vqf_olsson
python run_estimation.py --joint knee --method kf_gframe_olsson --no-plot
```

Available methods: `vqf_olsson`, `vqf_olsson_heading_correction`, `opensense`, `kf_gframe_olsson`, `kf_gframe_optimized`, `vqf_ik`, `all`

## Data Conventions

- **Quaternion format**: w, x, y, z (qmt convention)
- **Sampling rate**: 100 Hz throughout
- **Angles**: Degrees for display/RMSE, radians for computation
- **Cross-correlation alignment**: Used for phase correction before error calculation
- **Offset caching**: `offsets.json` stores computed IMU-Mocap alignment offsets; regenerates on first run if missing

## Code Style

- **Concise over verbose**: Minimize boilerplate; use libraries (e.g., `tqdm`) over manual implementations
- **Readable over clever**: Prefer `1024 * 1024` over `1 << 20`
- **Self-descriptive names**: Function names should convey purpose (e.g., `download_and_extract_simtk_dataset`)
- **One-liner docstrings**: Every function gets a single-line `"""docstring"""`
- **Inline argument docs**: Document parameters with trailing comments, one per line:
  ```python
  def estimate_joint_axis(
      q_rel,               # relative quaternion array (N, 4)
      axis_method='olsson',  # 'olsson', 'optimized', 'opensim', etc.
      gt_angles=None,      # ground truth angles in degrees
      correct_sign=True,   # flip axis to maximize correlation
  ):
      """Returns normalized joint axis vector (3,)."""
  ```
- **Minimal comments**: Let code be self-documenting; only comment non-obvious intent

## Git Preferences

- Do NOT add `Co-Authored-By` lines to commit messages
