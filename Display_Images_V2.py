import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import os

st.set_page_config(page_title="Interactive Line Scan", layout="wide")
st.title("🔬 Interactive Line-Scan Tool (with demo image)")

# ---------- helper ----------
def load_default_image():
    """Try to load image_data/sample.png relative to this script."""
    default_path = os.path.join(os.path.dirname(__file__), "image_data", "sample.png")
    if os.path.exists(default_path):
        return Image.open(default_path), "sample.png"
    return None, None


# ---------- state ----------
if "lines" not in st.session_state:
    st.session_state.lines = []
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "filename" not in st.session_state:
    st.session_state.filename = None

# ---------- upload or demo ----------
st.markdown("### 🖼️ Load Image")

col_u1, col_u2 = st.columns([3, 1])

with col_u1:
    uploaded = st.file_uploader(
        "Upload your image", type=["png", "jpg", "jpeg", "tif", "tiff"]
    )

# Spacer lines to align the demo button lower
with col_u2:
    for _ in range(3):
        st.text("")  # pushes button down
    use_demo = st.button("📂 Load demo image")

# --- handle user input ---
if uploaded:
    st.session_state.current_image = Image.open(uploaded)
    st.session_state.filename = os.path.splitext(uploaded.name)[0]

elif use_demo:
    img, name = load_default_image()
    if img is not None:
        st.session_state.current_image = img
        st.session_state.filename = os.path.splitext(name)[0]
        st.session_state.lines.clear()
        st.success("Demo image loaded successfully.")
    else:
        st.warning("No demo image found in ./image_data/sample.png")

# ---------- main app once image available ----------
if st.session_state.current_image is not None:
    img = st.session_state.current_image
    filename = st.session_state.filename
    raw = np.array(img)

    # 16-bit → 8-bit for display
    if img.mode == "I;16":
        disp = (raw / (raw.max() if raw.max() > 0 else 1) * 255).astype(np.uint8)
        img_disp = Image.fromarray(disp)
        raw_data = raw
    else:
        img_disp = img.convert("RGB")
        raw_data = np.array(img_disp)

    # ---------- layout ----------
    col_left, col_right = st.columns([2, 1], gap="large")

    # ========== LEFT COLUMN ==========
    with col_left:
        st.subheader("1️⃣ Define Line & View Image")

        cmap = st.selectbox(
            "Colormap",
            ["gray", "viridis", "plasma", "magma", "cividis", "hot", "cool", "jet"],
            index=0,
        )

        st.markdown("**Image display adjustments**")
        colb, colc = st.columns(2)
        brightness = colb.slider("Brightness", 0.5, 2.0, 1.0, 0.05)
        contrast = colc.slider("Contrast", 0.5, 2.0, 1.0, 0.05)

        enhancer = ImageEnhance.Brightness(img_disp)
        img_bright = enhancer.enhance(brightness)
        enhancer = ImageEnhance.Contrast(img_bright)
        img_adjusted = enhancer.enhance(contrast)
        disp = np.array(img_adjusted)

        st.markdown("**Line coordinates (pixels)**")
        c1, c2, c3, c4 = st.columns(4)
        x1 = c1.number_input("x₁", 0, raw_data.shape[1] - 1, 0, key="x1")
        y1 = c2.number_input("y₁", 0, raw_data.shape[0] - 1, 0, key="y1")
        x2 = c3.number_input("x₂", 0, raw_data.shape[1] - 1, raw_data.shape[1] // 2, key="x2")
        y2 = c4.number_input("y₂", 0, raw_data.shape[0] - 1, raw_data.shape[0] // 2, key="y2")

        col_btn1, col_btn2 = st.columns([1, 1])
        add_line = col_btn1.button("➕ Add line-scan")
        clear_lines = col_btn2.button("🧹 Clear all")

        if clear_lines:
            st.session_state.lines.clear()
            st.rerun()

        # --- image viewer ---
        fig = px.imshow(disp, color_continuous_scale=cmap)
        fig.update_layout(
            dragmode="pan",
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=0, b=0),
            height=550,
        )

        colors = ["lime", "cyan", "yellow", "magenta", "orange", "blue"]
        for i, line in enumerate(st.session_state.lines):
            (lx1, ly1), (lx2, ly2) = line["p1"], line["p2"]
            fig.add_trace(
                go.Scatter(
                    x=[lx1, lx2],
                    y=[ly1, ly2],
                    mode="lines+markers",
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=6),
                    name=f"Line {i+1}",
                )
            )

        # preview
        fig.add_trace(
            go.Scatter(
                x=[x1, x2],
                y=[y1, y2],
                mode="lines+markers",
                line=dict(color="red", width=2, dash="dot"),
                marker=dict(size=6, color="red"),
                name="Preview",
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- add line ---
        if add_line:
            n = int(np.hypot(x2 - x1, y2 - y1))
            if n > 0:
                xs, ys = np.linspace(x1, x2, n), np.linspace(y1, y2, n)
                xi = np.clip(xs.astype(int), 0, raw_data.shape[1] - 1)
                yi = np.clip(ys.astype(int), 0, raw_data.shape[0] - 1)
                if raw_data.ndim == 3:
                    intensity = raw_data[yi, xi].mean(axis=1)
                else:
                    intensity = raw_data[yi, xi]
                st.session_state.lines.append(
                    {"p1": (x1, y1), "p2": (x2, y2),
                     "x": xs, "y": ys, "intensity": intensity}
                )
            st.rerun()

        # --- smaller plot for line-scans ---
        if st.session_state.lines:
            st.markdown("### 2️⃣ Line-Scan Profiles")
            fig2, ax = plt.subplots(figsize=(6, 2.5))
            for i, line in enumerate(st.session_state.lines):
                ax.plot(
                    np.arange(len(line["intensity"])),
                    line["intensity"],
                    label=f"Line {i+1}",
                )
            ax.set_xlabel("Distance (px)")
            ax.set_ylabel("Intensity (a.u.)")
            ax.legend(fontsize="small", loc="upper right")
            st.pyplot(fig2)

    # ========== RIGHT COLUMN ==========
    with col_right:
        # Spacer to align with left plot bottom
        for _ in range(63):
            st.text("")

        st.subheader("3️⃣ Stored Line-Scans")

        if st.session_state.lines:
            for i, line in enumerate(st.session_state.lines):
                st.markdown(
                    f"**Line {i+1}:** ({int(line['p1'][0])}, {int(line['p1'][1])}) → "
                    f"({int(line['p2'][0])}, {int(line['p2'][1])})"
                )
                df_line = pd.DataFrame({
                    "x": line["x"],
                    "y": line["y"],
                    "intensity": line["intensity"],
                })
                csv_bytes = df_line.to_csv(index=False).encode("utf-8")
                st.download_button(
                    f"💾 Save Line {i+1}",
                    data=csv_bytes,
                    file_name=f"{filename}_Line{(i+1):02d}.csv",
                    mime="text/csv",
                    key=f"dl_{i}",
                )
                if st.button(f"❌ Delete Line {i+1}", key=f"del_{i}"):
                    st.session_state.lines.pop(i)
                    st.rerun()
        else:
            st.info("No stored lines yet – add one on the left.")

else:
    st.info("👆 Upload or load the demo image to start.")
