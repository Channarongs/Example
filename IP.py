# app.py
# ------------------------------------------------------------
# Streamlit Image Processing Lab (Grayscale/Color)
# Methods: Negative, Contrast Stretching, Piecewise Linear,
#          Log, Gamma, HistEq, AHE, CLAHE
#
# Requirements (add to requirements.txt if deploying):
# streamlit
# numpy
# pillow
# matplotlib
# scikit-image
# ------------------------------------------------------------

from __future__ import annotations
import io
from typing import Tuple, Literal

import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

# scikit-image for robust equalization/color space conversions
from skimage import exposure, color, util

st.set_page_config(
    page_title="Image Processing Lab",
    page_icon="🖼️",
    layout="wide"
)

# ----------------------- Utilities -----------------------

def load_image(file_bytes: bytes, mode: Literal["auto","grayscale","color"]) -> Tuple[np.ndarray, bool]:
    """Load image from uploaded bytes. Return float image in [0,1] and is_color flag."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")  # normalize load as RGB
    arr = np.asarray(img).astype(np.float32) / 255.0
    if mode == "grayscale":
        g = color.rgb2gray(arr)  # (H,W) float in [0,1]
        return g, False
    elif mode == "color":
        return arr, True
    else:
        # auto: preserve original if it seems grayscale
        if arr.ndim == 3:
            # detect if all channels are (almost) equal
            if np.allclose(arr[...,0], arr[...,1]) and np.allclose(arr[...,1], arr[...,2]):
                return color.rgb2gray(arr), False
            return arr, True
        return arr, False

def to_uint8(img: np.ndarray) -> np.ndarray:
    img_clip = np.clip(img, 0.0, 1.0)
    return (img_clip * 255.0 + 0.5).astype(np.uint8)

def plot_histogram(img: np.ndarray, is_color: bool, title: str):
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    if is_color:
        # Plot per-channel histogram
        for i, c in enumerate(["R", "G", "B"]):
            ax.hist(img[..., i].ravel(), bins=256, range=(0,1), histtype='step', label=c)
        ax.legend(loc="upper right")
    else:
        ax.hist(img.ravel(), bins=256, range=(0,1), histtype='stepfilled', alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Intensity (0-1)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def contrast_stretch(img: np.ndarray, low_perc: float, high_perc: float) -> np.ndarray:
    # Stretch intensities between low/high percentiles to [0,1]
    if img.ndim == 2:
        p_low, p_high = np.percentile(img, [low_perc, high_perc])
        return np.clip((img - p_low) / max(p_high - p_low, 1e-8), 0, 1)
    else:
        out = np.empty_like(img)
        for ch in range(3):
            p_low, p_high = np.percentile(img[..., ch], [low_perc, high_perc])
            out[..., ch] = np.clip((img[..., ch] - p_low) / max(p_high - p_low, 1e-8), 0, 1)
        return out

def piecewise_linear(img: np.ndarray, r1: float, s1: float, r2: float, s2: float) -> np.ndarray:
    """
    3-segment piecewise linear mapping:
    (0,0)->(r1,s1)->(r2,s2)->(1,1) with 0<=r1<r2<=1
    """
    def map_channel(x):
        x = np.clip(x, 0, 1)
        out = np.zeros_like(x)
        # Segment 1: [0, r1]
        m1 = s1 / max(r1, 1e-8)
        mask1 = x <= r1
        out[mask1] = m1 * x[mask1]
        # Segment 2: (r1, r2]
        m2 = (s2 - s1) / max((r2 - r1), 1e-8)
        mask2 = (x > r1) & (x <= r2)
        out[mask2] = s1 + m2 * (x[mask2] - r1)
        # Segment 3: (r2, 1]
        m3 = (1 - s2) / max((1 - r2), 1e-8)
        mask3 = x > r2
        out[mask3] = s2 + m3 * (x[mask3] - r2)
        return np.clip(out, 0, 1)

    if img.ndim == 2:
        return map_channel(img)
    else:
        return np.stack([map_channel(img[...,i]) for i in range(3)], axis=-1)

def log_transform(img: np.ndarray, gain: float) -> np.ndarray:
    # s = gain * log(1 + r), normalized back to [0,1] if needed
    s = gain * np.log1p(img)
    s = s / (s.max() + 1e-8)
    return np.clip(s, 0, 1)

def gamma_transform(img: np.ndarray, gamma: float, gain: float) -> np.ndarray:
    # s = gain * r^gamma -> rescale to [0,1] for display
    s = gain * np.power(np.clip(img, 0, 1), gamma)
    s = s / (s.max() + 1e-8)
    return np.clip(s, 0, 1)

def histeq(img: np.ndarray, is_color: bool) -> np.ndarray:
    if not is_color:
        return np.clip(exposure.equalize_hist(img), 0, 1)
    # Equalize luminance only (Y in YCbCr) to preserve color
    ycbcr = color.rgb2ycbcr(img)
    y = ycbcr[..., 0] / 255.0  # Y in [0,1]
    y_eq = exposure.equalize_hist(y)
    ycbcr[..., 0] = np.clip(y_eq * 255.0, 0, 255)
    rgb = color.ycbcr2rgb(ycbcr).astype(np.float32)
    return np.clip(rgb, 0, 1)

def adaptive_equalization(img: np.ndarray, is_color: bool, kernel_size: int, clip_limit: float | None) -> np.ndarray:
    """
    AHE if clip_limit is None or >= 1.0 (effectively no clipping).
    CLAHE if 0 < clip_limit < 1 (skimage equalize_adapthist uses [0,1]).
    Applied on luminance for color images.
    """
    ks = (kernel_size, kernel_size)
    if not is_color:
        return np.clip(exposure.equalize_adapthist(img, kernel_size=ks, clip_limit=clip_limit), 0, 1)
    hsv = color.rgb2hsv(img)
    v = hsv[..., 2]
    v_eq = exposure.equalize_adapthist(v, kernel_size=ks, clip_limit=clip_limit)
    hsv[..., 2] = np.clip(v_eq, 0, 1)
    rgb = color.hsv2rgb(hsv).astype(np.float32)
    return np.clip(rgb, 0, 1)

# ----------------------- Sidebar UI -----------------------

st.sidebar.title("🔧 Controls")

img_choice = st.sidebar.radio(
    "Image Type",
    options=["Auto detect", "Grayscale", "Color"],
    index=0
)

method = st.sidebar.selectbox(
    "Processing Method",
    [
        "Linear Negative",
        "Contrast Stretching",
        "Piecewise Linear",
        "Log Transform",
        "Gamma Transform",
        "Histogram Equalization",
        "Adaptive Histogram Equalization (AHE)",
        "CLAHE"
    ]
)

uploaded = st.sidebar.file_uploader("Upload an image (PNG/JPG)", type=["png","jpg","jpeg"])

# Parameters per method
if method == "Contrast Stretching":
    low = st.sidebar.slider("Low percentile", 0.0, 20.0, 1.0, 0.1)
    high = st.sidebar.slider("High percentile", 80.0, 100.0, 99.0, 0.1)
    if low >= high:
        st.sidebar.warning("Low percentile must be < High percentile.")
elif method == "Piecewise Linear":
    r1 = st.sidebar.slider("r1 (input break 1)", 0.0, 1.0, 0.25, 0.01)
    s1 = st.sidebar.slider("s1 (output at r1)", 0.0, 1.0, 0.25, 0.01)
    r2 = st.sidebar.slider("r2 (input break 2)", 0.0, 1.0, 0.75, 0.01)
    s2 = st.sidebar.slider("s2 (output at r2)", 0.0, 1.0, 0.75, 0.01)
    if r1 >= r2:
        st.sidebar.warning("Require r1 < r2.")
elif method == "Log Transform":
    gain = st.sidebar.slider("Gain", 0.1, 5.0, 1.0, 0.1)
elif method == "Gamma Transform":
    gamma = st.sidebar.slider("Gamma (γ)", 0.05, 5.0, 1.0, 0.05)
    gain_g = st.sidebar.slider("Gain", 0.1, 5.0, 1.0, 0.1)
elif method in ["Adaptive Histogram Equalization (AHE)", "CLAHE"]:
    kernel = st.sidebar.slider("Kernel size (tiles)", 4, 64, 16, 2)
    if method == "CLAHE":
        clip = st.sidebar.slider("Clip limit (0–1, lower=stronger clip)", 0.001, 0.2, 0.01, 0.001)
    else:
        # AHE: no clipping (set to 1.0 -> effectively none in skimage)
        clip = None

# ----------------------- Main Layout -----------------------

st.title("🖼️ Image Processing Lab")
st.caption("Professional, optimized Streamlit demo with per-method controls, grayscale and color support, and before/after histograms.")

if not uploaded:
    st.info("Upload an image to begin. PNG/JPG supported.")
    st.stop()

# Load image according to user selection
mode_map = {"Auto detect":"auto", "Grayscale":"grayscale", "Color":"color"}
img_float, is_color = load_image(uploaded.read(), mode=mode_map[img_choice])

# ----------------------- Processing -----------------------

if method == "Linear Negative":
    processed = 1.0 - img_float

elif method == "Contrast Stretching":
    processed = contrast_stretch(img_float, low, high)

elif method == "Piecewise Linear":
    processed = piecewise_linear(img_float, r1, s1, r2, s2)

elif method == "Log Transform":
    processed = log_transform(img_float, gain)

elif method == "Gamma Transform":
    processed = gamma_transform(img_float, gamma, gain_g)

elif method == "Histogram Equalization":
    processed = histeq(img_float, is_color)

elif method == "Adaptive Histogram Equalization (AHE)":
    processed = adaptive_equalization(img_float, is_color, kernel_size=kernel, clip_limit=clip)

elif method == "CLAHE":
    processed = adaptive_equalization(img_float, is_color, kernel_size=kernel, clip_limit=clip)

else:
    processed = img_float

# ----------------------- Display -----------------------

col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Before")
    if is_color:
        st.image(to_uint8(img_float), channels="RGB", use_column_width=True)
    else:
        st.image(to_uint8(img_float), clamp=True, use_column_width=True)
    plot_histogram(img_float, is_color, "Before Histogram")

with col2:
    st.subheader("After: " + method)
    if is_color:
        st.image(to_uint8(processed), channels="RGB", use_column_width=True)
    else:
        st.image(to_uint8(processed), clamp=True, use_column_width=True)
    plot_histogram(processed, is_color, "After Histogram")

# ----------------------- Footer & Notes -----------------------

with st.expander("ℹ️ Notes & Tips"):
    st.markdown(
        """
- **Color-safe equalization** is applied on luminance (Y/Value) to preserve hues.
- **Contrast Stretching** uses per-channel percentiles (color) for robust dynamic range expansion.
- **Piecewise Linear** gives you flexible tone mapping via two breakpoints.
- **AHE vs CLAHE**: AHE enhances locally *without* clipping (can amplify noise), while CLAHE limits contrast via `clip limit`.
- All operations run in float domain **[0,1]** and are clipped to valid ranges before display.
        """
    )
