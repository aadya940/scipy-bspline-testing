"""Spline art 2: splines as a paintbrush.

A painter drags the brush along the outlines of an object, not across them.
So we first compute, at every spot of the photo, which direction the outlines
run. Then we pick a random spot, walk a short shaky path along that direction
field, fit one parametric smoothing spline through the walked points, and draw
the smooth curve as a single thick stroke in the color the photo has at that
spot. Nine thousand strokes later the photo has been repainted, and every
stroke is a fitted spline on its own little knot vector.

Needs scipy with the t= argument of make_smoothing_spline (scipy/scipy#25862).
The photo is scikit-learn's bundled sample image china.jpg (Summer Palace,
Beijing).

Usage:  python spline_art_brush.py
"""
import numpy as np
from scipy.interpolate import make_smoothing_spline
from scipy.ndimage import gaussian_filter, map_coordinates
from sklearn.datasets import load_sample_image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)

img = load_sample_image("china.jpg").astype(float) / 255.0
H, W, _ = img.shape
gray = img.mean(axis=2)

# stroke directions from the structure tensor, along the edges
gy, gx = np.gradient(gaussian_filter(gray, 1.2))
Jxx = gaussian_filter(gx * gx, 5)
Jxy = gaussian_filter(gx * gy, 5)
Jyy = gaussian_filter(gy * gy, 5)
theta = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)
ux, uy = -np.sin(theta), np.cos(theta)
coherence = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2)


def field(px, py):
    fx = map_coordinates(ux, [[py], [px]], order=1)[0]
    fy = map_coordinates(uy, [[py], [px]], order=1)[0]
    return fx, fy


N_STROKES = 9000
STEP = 2.2
N_STEPS = 11

kn_inner = np.linspace(0.25, 0.75, 3)
t_stroke = np.concatenate([[0.0] * 4, kn_inner, [1.0] * 4])
s_dense = np.linspace(0, 1, 40)

fig, ax = plt.subplots(figsize=(W / 100.0, H / 100.0))
fig.subplots_adjust(0, 0, 1, 1)
ax.set_xlim(0, W); ax.set_ylim(H, 0)
ax.axis("off")
ax.add_patch(plt.Rectangle((0, 0), W, H, color="#F6F1E5", zorder=0))

# seed more strokes where the image has structure
prob = (coherence / coherence.max()) ** 0.4 + 0.15
prob /= prob.sum()
flat = rng.choice(H * W, size=N_STROKES, p=prob.ravel(), replace=True)
seeds = np.column_stack([flat % W, flat // W]).astype(float)
seeds += rng.uniform(-0.5, 0.5, seeds.shape)

order = rng.permutation(N_STROKES)
for i in order:
    px, py = seeds[i]
    pts = [(px, py)]
    dx, dy = field(px, py)
    for _ in range(N_STEPS):
        qx, qy = pts[-1]
        fx, fy = field(np.clip(qx, 0, W - 1), np.clip(qy, 0, H - 1))
        if fx * dx + fy * dy < 0:
            fx, fy = -fx, -fy
        dx, dy = fx, fy
        pts.append((qx + STEP * fx + rng.normal(0, 0.35),
                    qy + STEP * fy + rng.normal(0, 0.35)))
    dx, dy = field(px, py)
    dx, dy = -dx, -dy
    back = [(px, py)]
    for _ in range(N_STEPS):
        qx, qy = back[-1]
        fx, fy = field(np.clip(qx, 0, W - 1), np.clip(qy, 0, H - 1))
        if fx * dx + fy * dy < 0:
            fx, fy = -fx, -fy
        dx, dy = fx, fy
        back.append((qx + STEP * fx + rng.normal(0, 0.35),
                     qy + STEP * fy + rng.normal(0, 0.35)))
    P = np.array(back[::-1] + pts[1:])

    s = np.linspace(0, 1, len(P))
    fx_s = make_smoothing_spline(s, P[:, 0], lam=1e-4, t=t_stroke)
    fy_s = make_smoothing_spline(s, P[:, 1], lam=1e-4, t=t_stroke)
    X, Y = fx_s(s_dense), fy_s(s_dense)

    cx = int(np.clip(py, 0, H - 1))
    cy = int(np.clip(px, 0, W - 1))
    col = img[cx, cy] * rng.uniform(0.92, 1.06)
    ax.plot(X, Y, color=np.clip(col, 0, 1),
            lw=rng.uniform(2.2, 4.4), alpha=0.95,
            solid_capstyle="round", zorder=1 + rng.random())

fig.savefig("spline-art-2.png", dpi=270)
print("wrote spline-art-2.png")
