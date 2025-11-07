import streamlit as st
from PIL import Image
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import io
import os

st.set_page_config(page_title="Interactive Line Scan", layout="wide")
st.title("🔬 Interactive Line-Scan Tool (CSV per line)")

# ---------- state ----------
if "lines" not in st.session_state:
    st.session_state.lines = []

uploaded = st.file_uploader(
    "Upload an image", type=["png", "jpg", "jpeg", "tif", "tiff"]
)

if uploaded:
    filename = os.path.splitext(uploaded.name)[0]
    img = Image.open(uploaded)
    raw = np.array(img)

    # 16-bit → 8-bit for display
    if img.mode == "I;16":
        disp = (raw / (raw.max() if raw.max() > 0 else 1) * 255).astype(np.uint8)
        img_disp = Image.fromarray(disp)
        raw_data = raw
    else:
        img_disp = img.convert("RGB")
        raw_data = np.array(img_disp)

    # ---------- coordinate inputs ----------
    st.subheader("1️⃣ Define a straight line")
    c1, c2, c3, c4 = st.columns(4)
    x1 = c1.number_input("x₁", 0, raw_data.shape[1] - 1, 0)
    y1 = c2.number_input("y₁", 0, raw_data.shape[0] - 1, 0)
    x2 = c3.number_input("x₂", 0, raw_data.shape[1] - 1, raw_data.shape[1] // 2)
    y2 = c4.number_input("y₂", 0, raw_data.shape[0] - 1, raw_data.shape[0] // 2)

    col_btn1, col_btn2 = st.columns(2)
    add_line = col_btn1.button("➕ Add / update line-scan")
    clear_lines = col_btn2.button("🧹 Clear all lines")

    if clear_lines:
        st.session_state.lines.clear()
        st.rerun()

    # ---------- image viewer ----------
    fig = px.imshow(np.array(img_disp), binary_string=False)
    fig.update_layout(dragmode="pan", margin=dict(l=0, r=0, t=0, b=0), height=500)

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

    # ---------- add new line ----------
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

    # ---------- list, delete, download ----------
    if st.session_state.lines:
        
        st.subheader("2️⃣ Stored line-scans")
        for i, line in enumerate(st.session_state.lines):
            col_a, col_b, col_c, col_d = st.columns([4, 2, 2, 2])
            col_a.write(
                f"**Line {i + 1}:** "
                f"({int(line['p1'][0])}, {int(line['p1'][1])}) → "
                f"({int(line['p2'][0])}, {int(line['p2'][1])})"
            )

            # --- rebuild x/y arrays if they don't exist (backward compatibility) ---
            if "x" not in line or "y" not in line:
                n = len(line["intensity"])
                x1, y1 = line["p1"]
                x2, y2 = line["p2"]
                line["x"] = np.linspace(x1, x2, n)
                line["y"] = np.linspace(y1, y2, n)

            # --- per-line CSV ---
            df_line = pd.DataFrame({
                "x": line["x"],
                "y": line["y"],
                "intensity": line["intensity"],
            })
            csv_bytes = df_line.to_csv(index=False).encode("utf-8")
            col_b.download_button(
                "💾 CSV",
                data=csv_bytes,
                file_name=f"{filename}_Line{(i + 1):02d}.csv",
                mime="text/csv",
                key=f"dl_{i}",
            )

            # --- delete button ---
            if col_c.button("❌ Delete", key=f"del_{i}"):
                st.session_state.lines.pop(i)
                st.rerun()

        # ---------- combined plot ----------
        st.subheader("3️⃣ Line-scan profiles")
        fig2, ax = plt.subplots(figsize=(6, 3))
        for i, line in enumerate(st.session_state.lines):
            ax.plot(
                np.arange(len(line["intensity"])),
                line["intensity"],
                label=f"Line {i+1}",
            )
        ax.set_xlabel("Distance (px)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.legend()
        st.pyplot(fig2)

    else:
        st.info("No lines yet – enter coordinates and click **Add / update line-scan**.")

else:
    st.info("👆 Upload an image to start.")
