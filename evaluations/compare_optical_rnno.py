"""Compare RNNO relative orientation against optical marker ground truth."""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import qmt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from run_estimation import prepare_data
from methods import run_rnno
from utils import compute_q_rel_optical
from constants import FS

SUBJECT = 'Subject08'

joint = 'knee'
data = prepare_data(joint, SUBJECT)
trc_path = str(ROOT / f'data/{SUBJECT}/walking/Mocap/walking.trc')

# RNNO q_rel
_, _, _, _, q_rel_rnno = run_rnno(
    data['acc_prox'], data['gyr_prox'],
    data['acc_dist'], data['gyr_dist'],
    data['fs'], axis_mode='model',
)

# Optical q_rel
q_rel_opt = compute_q_rel_optical(trc_path, joint)

# Take 60s from the middle
n = min(len(q_rel_rnno), len(q_rel_opt))
win = int(60 * FS)
start = (n - win) // 2
q_rel_rnno = qmt.quatUnwrap(q_rel_rnno[start:start + win])
q_rel_opt = qmt.quatUnwrap(q_rel_opt[start:start + win])
valid = ~np.isnan(q_rel_opt).any(axis=1)

# Relative changes from first frame (removes constant frame offset)
dq_rnno = qmt.qmult(q_rel_rnno, qmt.qinv(q_rel_rnno[0]))
dq_opt = qmt.qmult(q_rel_opt, qmt.qinv(q_rel_opt[0]))

# Procrustes alignment: dq_rnno ≈ q_L * dq_opt * qinv(q_L)
axes_rnno = qmt.quatAxis(dq_rnno[valid])
axes_opt = qmt.quatAxis(dq_opt[valid])
angles = np.abs(np.degrees(qmt.quatAngle(dq_rnno[valid])))
big = angles > 5
H = axes_opt[big].T @ axes_rnno[big]
U, _, Vt = np.linalg.svd(H)
R_L = Vt.T @ U.T
if np.linalg.det(R_L) < 0:
    Vt[-1] *= -1
    R_L = Vt.T @ U.T
q_L = qmt.quatFromRotMat(R_L.reshape(1, 3, 3))[0]

dq_opt_aligned = qmt.qmult(q_L, qmt.qmult(dq_opt, qmt.qinv(q_L)))

# Per-frame angular error
q_err = qmt.qmult(qmt.qinv(dq_rnno), dq_opt_aligned)
angle_err = np.abs(np.degrees(qmt.quatAngle(q_err)))

errs = angle_err[valid]
print(f"\n{joint}: mean={np.mean(errs):.2f}°  median={np.median(errs):.2f}°  P95={np.percentile(errs, 95):.2f}°")

# Plot
time = np.arange(win) / FS
fig, ax = plt.subplots(figsize=(14, 3))
ax.plot(time[valid], errs, alpha=0.6, linewidth=0.5)
ax.axhline(np.mean(errs), color='r', ls='--', label=f'mean={np.mean(errs):.1f}°')
ax.set_ylabel('Error (deg)')
ax.set_xlabel('Time (s)')
ax.set_title(f'{joint.title()} — RNNO vs Optical q_rel ({SUBJECT})')
ax.legend()
ax.set_ylim(0, min(30, np.percentile(errs, 95) * 2))
plt.tight_layout()
plt.savefig(ROOT / f'results/{joint}_qrel_optical_vs_rnno_{SUBJECT}.png', dpi=150)
print(f"  Saved: results/{joint}_qrel_optical_vs_rnno_{SUBJECT}.png")
