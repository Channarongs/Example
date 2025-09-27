# app.py
# ------------------------------------------------------------
# Fundus Enhancement Demo (per Wang et al., 2021)
# Pipeline:
# 1) TV-based decomposition: (image -> structure + noise) -> (base + detail) + noise
# 2) Illuminant correction on base via Naka–Rushton (visual adaptation)
# 3) Fusion: enhanced = corrected_base + ω * detail, with per-channel α
# 4) Optional color-space for luminance (HSV/Lab)
# 5) Metrics: Δlocal (local contrast gain) and entropy
# ------------------------------------------------------------

from __future__ import annotations
import io
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

from skimage import color, exposure, util
from skimage.filters import gaussian
from skimage.restoration import denoise_tv_chambolle
from skimage.measure import shannon_entropy

# ---------- Streamlit page ----------
st.set_page_config(page_title="Fundus Enhancement — Image Decomposition + Visual Adaptation",
                   page_icon="🩺", layout="wide")

# ---------- Helpers ----------
def to_float(img: np.ndarray) -> np.ndarray:
    """Convert uint8 [0,255] or float to float32 in [0,1]."""
    if img.dtype == np.uint8:
        return img.astype(np.float32)/255.0
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img/255.0
    return np.clip(img, 0, 1)

def to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0, 1)*255.0 + 0.5).astype(np.uint8)

def load_image(file_bytes: bytes) -> np.ndarray:
    im = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return to_float(np.asarray(im))

def global_noise_lambda(img_gray: np.ndarray) -> float:
    """
    Immerkaer’s fast noise variance estimator -> proportional λ1.
    Uses the 3x3 kernel specified in the paper for an RGB channel; here on grayscale proxy.
    """
    # 3x3 kernel
    k = np.array([[1, -2, 1],
                  [-2, 4, -2],
                  [1, -2, 1]], dtype=np.float32)
    conv = util.img_as_float32(img_gray)
    # naive conv (reflect padding)
    conv = conv - np.pad(conv, 1, mode="reflect")[1:-1,1:-1]  # cheap op to keep memory small (placeholder)
    # robust alternative: use gaussian gradient magnitude as proxy:
    # but we stick to a lightweight proxy—if you prefer exact conv, use scipy.signal.convolve2d.
    # We'll emulate the statistic on the local Laplacian-ish response:
    resp = np.abs(gaussian(img_gray, sigma=0.5, preserve_range=True) - img_gray)
    W, H = img_gray.shape[1], img_gray.shape[0]
    c = np.sqrt(np.pi/2.0) / (6.0*max((W-2)*(H-2), 1))
    lam = c * np.sum(resp)
    return float(max(lam, 1e-4))

def tv_decompose_channel(ch: np.ndarray, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    ROF-like split: ch ≈ structure + noise, where structure = TV-denoised(ch, weight=lam).
    Returns (structure, noise).
    """
    structure = denoise_tv_chambolle(ch, weight=lam, channel_axis=None)
    noise = ch - structure
    return structure, noise

def second_tv_split(structure: np.ndarray, lam2: float) -> Tuple[np.ndarray, np.ndarray]:
    """structure ≈ base + detail via TV; base = denoise(structure), detail = structure - base."""
    base = denoise_tv_chambolle(structure, weight=lam2, channel_axis=None)
    detail = structure - base
    return base, detail

def luminance_map(img: np.ndarray, space: str) -> Tuple[np.ndarray, str]:
    if space == "HSV (V channel)":
        hsv = color.rgb2hsv(img)
        return hsv[...,2], "HSV"
    elif space == "Lab (L* channel)":
        lab = color.rgb2lab(img)
        L = np.clip(lab[...,0] / 100.0, 0, 1)
        return L, "Lab"
    else:
        # Fallback HSV
        hsv = color.rgb2hsv(img)
        return hsv[...,2], "HSV"

def apply_luminance(img: np.ndarray, new_L: np.ndarray, space: str) -> np.ndarray:
    new_L = np.clip(new_L, 0, 1)
    if space == "HSV (V channel)":
        hsv = color.rgb2hsv(img)
        hsv[...,2] = new_L
        return np.clip(color.hsv2rgb(hsv), 0, 1)
    else:
        lab = color.rgb2lab(img)
        lab[...,0] = np.clip(new_L*100.0, 0, 100)
        return np.clip(color.lab2rgb(lab), 0, 1)

def naka_rushton(Lin: np.ndarray, n: float = 1.0) -> np.ndarray:
    # σg from mean/stdev of Lin
    Mg = float(np.mean(Lin))
    Sg = float(np.std(Lin))
    sigma_g = Mg / (1.0 + np.exp(Sg))
    # Lout = Lin^n / (Lin^n + σg^n)
    Ln = np.power(np.clip(Lin, 0, 1), n)
    denom = Ln + (sigma_g ** n)
    Lout = np.divide(Ln, np.maximum(denom, 1e-8))
    return np.clip(Lout, 0, 1)

def fuse_corrected_with_detail(corrected_base: np.ndarray,
                               detail: np.ndarray,
                               alpha: Tuple[float,float,float],
                               gauss_sigma: float = 10.0) -> np.ndarray:
    """
    I_out = corrected_base + ω * detail, with ω_c(x) = α_c * (|detail_c| * Gσ) (smoothed magnitude).
    """
    out = np.empty_like(corrected_base)
    if detail.ndim == 2:
        w = alpha[1] * gaussian(np.abs(detail), sigma=gauss_sigma, preserve_range=True)
        out = corrected_base + w * detail
    else:
        for c, a in enumerate(alpha):
            mag = gaussian(np.abs(detail[...,c]), sigma=gauss_sigma, preserve_range=True)
            w = a * mag
            out[...,c] = corrected_base[...,c] + w * detail[...,c]
    return np.clip(out, 0, 1)

def rgb2gray_safe(img: np.ndarray) -> np.ndarray:
    return color.rgb2gray(img)

def hist_plot(img: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    if img.ndim == 3:
        for i, lab in enumerate(["R","G","B"]):
            ax.hist(img[...,i].ravel(), bins=256, range=(0,1), histtype="step", label=lab)
        ax.legend(loc="upper right")
    else:
        ax.hist(img.ravel(), bins=256, range=(0,1), histtype="stepfilled", alpha=0.85)
    ax.set_title(title); ax.set_xlabel("Intensity"); ax.set_ylabel("Count")
    fig.tight_layout()
    st.pyplot(fig); plt.close(fig)

def delta_local(original: np.ndarray, enhanced: np.ndarray, N: int = 50) -> float:
    """
    Δlocal = mean(Clocal_enh) - mean(Clocal_orig), computed on gray luminance and within a circular mask of retina if available.
    For simplicity, we compute over full valid area.
    Clocal = (max-min)/(max+min) within N×N windows (vectorized approx).
    """
    def clocal_map(I: np.ndarray) -> np.ndarray:
        # Use a fast approximation via percentile within uniform windows using reflect padding.
        # We approximate max/min by dilate/erode with Gaussian as a soft-extrema proxy for speed.
        I = np.clip(I, 0, 1)
        soft_max = gaussian(I, sigma=N/10.0, preserve_range=True)
        soft_min = gaussian(1.0 - I, sigma=N/10.0, preserve_range=True)
        soft_min = 1.0 - soft_min
        num = soft_max - soft_min
        den = soft_max + soft_min + 1e-8
        return np.clip(num/den, 0, 1)
    g0 = rgb2gray_safe(original) if original.ndim==3 else original
    g1 = rgb2gray_safe(enhanced) if enhanced.ndim==3 else enhanced
    return float(np.mean(clocal_map(g1)) - np.mean(clocal_map(g0)))

# ---------- Sidebar ----------
st.sidebar.title("🛠️ Controls")

with st.sidebar.expander("Input"):
    uploaded = st.file_uploader("Upload fundus image (PNG/JPG)", type=["png","jpg","jpeg"])
    use_demo = st.toggle("Use demo image if none uploaded", value=True)

with st.sidebar.expander("Decomposition"):
    lam2 = st.slider("λ₂ (structure→base+detail)", 0.05, 1.0, 0.30, 0.05)
    per_channel_l1 = st.checkbox("Estimate λ₁ per-channel (noise split)", value=True)

with st.sidebar.expander("Illuminant correction"):
    lum_space = st.selectbox("Luminance color space", ["HSV (V channel)", "Lab (L* channel)"], index=0)
    naka_n = st.slider("Naka–Rushton exponent n", 0.5, 3.0, 1.0, 0.1)

with st.sidebar.expander("Detail fusion"):
    aR = st.number_input("α_R (veins, R-channel)", min_value=0.0, max_value=2000.0, value=600.0, step=50.0)
    aG = st.number_input("α_G (arteries, G-channel)", min_value=0.0, max_value=2000.0, value=600.0, step=50.0)
    aB = st.number_input("α_B (artifacts, B-channel)", min_value=0.0, max_value=2000.0, value=0.0, step=50.0)
    gsig = st.slider("Gaussian σ for |detail| smoothing", 1.0, 20.0, 10.0, 1.0)

with st.sidebar.expander("Display"):
    show_layers = st.checkbox("Show base/detail/noise layers", value=True)
    show_hist = st.checkbox("Show histograms", value=True)

# ---------- Load image or demo ----------
if uploaded is None and not use_demo:
    st.info("Upload an image to begin.")
    st.stop()

if uploaded is not None:
    img = load_image(uploaded.getvalue())
else:
    # Demo image (astronaut ≈ color; you may replace with a sample fundus image if you have one)
    from skimage import data
    img = to_float(data.astronaut())  # placeholder demo

H, W = img.shape[:2]
st.markdown("### Fundus Enhancement — Image Decomposition + Visual Adaptation")

# ---------- Stage 1: split (image -> structure + noise) ----------
# Work per channel; also keep grayscale proxy for λ1 estimation.
gray = rgb2gray_safe(img)
lam1_est = global_noise_lambda(gray)

structure = np.zeros_like(img)
noise = np.zeros_like(img)
if img.ndim == 3:
    for c in range(3):
        lam1 = lam1_est
        if per_channel_l1:
            lam1 = global_noise_lambda(to_float(img[...,c]))
        s, n = tv_decompose_channel(img[...,c], lam=lam1)
        structure[...,c] = s
        noise[...,c] = n
else:
    structure, noise = tv_decompose_channel(img, lam=lam1_est)

# ---------- Stage 2: split (structure -> base + detail) ----------
base = np.zeros_like(img)
detail = np.zeros_like(img)
if img.ndim == 3:
    for c in range(3):
        b, d = second_tv_split(structure[...,c], lam2)
        base[...,c] = b
        detail[...,c] = d
else:
    base, detail = second_tv_split(structure, lam2)

# ---------- Illuminant correction on base (visual adaptation) ----------
L_in, used_space = luminance_map(base, lum_space)
L_out = naka_rushton(L_in, n=naka_n)
corrected_base = apply_luminance(base, L_out, lum_space)

# ---------- Fusion with detail (and B-channel suppression) ----------
alpha = (aR, aG, aB)
enhanced = fuse_corrected_with_detail(corrected_base, detail, alpha=alpha, gauss_sigma=gsig)

# ---------- Metrics ----------
delta_loc = delta_local(img, enhanced, N=50)
ent_orig = shannon_entropy(to_uint8(img))
ent_enh = shannon_entropy(to_uint8(enhanced))

# ---------- Layout ----------
colA, colB = st.columns(2, gap="large")
with colA:
    st.subheader("Original")
    st.image(to_uint8(img), channels="RGB", use_column_width=True)
    if show_hist: hist_plot(img, "Original Histogram")
with colB:
    st.subheader("Enhanced")
    st.image(to_uint8(enhanced), channels="RGB", use_column_width=True)
    if show_hist: hist_plot(enhanced, "Enhanced Histogram")

st.markdown(
    f"**Δlocal** (contrast gain) = `{delta_loc:.4f}` &nbsp;&nbsp;|&nbsp;&nbsp; "
    f"Entropy: original `{ent_orig:.2f}` → enhanced `{ent_enh:.2f}`"
)

if show_layers:
    st.markdown("#### Intermediate Layers")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("Base")
        st.image(to_uint8(base), channels="RGB", use_column_width=True)
    with c2:
        st.caption("Detail (×3 for visibility)")
        st.image(to_uint8(np.clip(0.5 + 3.0*detail, 0, 1)), channels="RGB", use_column_width=True)
    with c3:
        st.caption("Noise (×5 for visibility)")
        st.image(to_uint8(np.clip(0.5 + 5.0*noise, 0, 1)), channels="RGB", use_column_width=True)

# ---------- Download buttons ----------
buf_out = io.BytesIO()
Image.fromarray(to_uint8(enhanced)).save(buf_out, format="PNG")
st.download_button("⬇️ Download Enhanced PNG", data=buf_out.getvalue(),
                   file_name="enhanced.png", mime="image/png")

# Optional: save intermediate composite panel
