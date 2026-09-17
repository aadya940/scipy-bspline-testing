"""Spline art 1: lambda as a focus knob.

An image is just a stack of horizontal lines. Fit a smoothing spline on every
line twice, once with a small lambda and once with a large lambda, and you get
a sharp and a soft version of the photo, both drawn entirely by splines.
Blending the two with a spatial mask puts the focus wherever you want it.

Needs scipy with the t= argument of make_smoothing_spline (scipy/scipy#25862).
The photo is scikit-learn's bundled sample image china.jpg (Summer Palace,
Beijing).

Usage:  python spline_art_focus.py
"""
import numpy as np
from scipy.interpolate import make_smoothing_spline
from scipy.ndimage import gaussian_filter
from sklearn.datasets import load_sample_image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

img = load_sample_image("china.jpg").astype(float) / 255.0
H, W, _ = img.shape
x = np.arange(W) / (W - 1)

# one knot every 4th pixel, the solve is sized by the knots not the pixels
inner = x[4:-4:4]
t = np.concatenate([[x[0]] * 4, inner, [x[-1]] * 4])

LAM_SHARP = 1e-12
LAM_SOFT = 1e-6


def spline_pass(image, lam):
    out = np.empty_like(image)
    for c in range(3):
        for r in range(H):
            spl = make_smoothing_spline(x, image[r, :, c], lam=lam, t=t)
            out[r, :, c] = spl(x)
    return np.clip(out, 0, 1)


sharp = spline_pass(img, LAM_SHARP)
soft = spline_pass(img, LAM_SOFT)

yy, xx = np.mgrid[0:H, 0:W]

# the pagoda sits in the left half of the frame
subject = np.exp(-(((xx - 0.30 * W) / (0.26 * W)) ** 2
                   + ((yy - 0.48 * H) / (0.42 * H)) ** 2) ** 2)

band = np.exp(-(((yy - 0.55 * H) / (0.16 * H)) ** 2) ** 2)

gray = img.mean(axis=2)
energy = gaussian_filter(gaussian_filter(gray, 1.0) ** 2, 6) \
    - gaussian_filter(gray, 6) ** 2
calm = energy < np.percentile(energy, 55)
calm = gaussian_filter(calm.astype(float), 9)

panels = [
    ("focus on the subject", subject),
    ("inverted: subject painted", 1 - subject),
    ("tilt-shift: a sharp band", band),
    ("painted where the image is calm", 1 - calm),
]

fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.2),
                         gridspec_kw=dict(wspace=0.04, hspace=0.16))
for ax, (title, m) in zip(axes.ravel(), panels):
    m3 = gaussian_filter(m, 3)[..., None]
    ax.imshow(m3 * sharp + (1 - m3) * soft)
    ax.set_title(title, fontsize=19)
    ax.axis("off")
fig.savefig("spline-art-1.png", dpi=150, bbox_inches="tight",
            facecolor="#FCFBF7")
print("wrote spline-art-1.png")
