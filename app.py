import os
import cv2
import time
import tempfile
import glob

import streamlit as st
from streamlit_drawable_canvas import st_canvas

from settings import (
    BASE_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_VIDEO_PATH,
    DEFAULT_CONFIDENCE,
    CANVAS_WIDTH,
    CANVAS_HEIGHT,
    CANVAS_FILL_COLOR,
    CANVAS_STROKE_WIDTH,
    CANVAS_BG_COLOR
)
from helper import extract_line_coords, run_counter

# — Pastikan cwd = folder ini agar DEFAULT_VIDEO_PATH valid
os.chdir(BASE_DIR)

st.set_page_config(page_title="Vehicle Counter", layout="wide")
st.title("🚗 Vehicle Counter & Speed Estimation")

# — Sidebar —————————————————————————————————————————————
st.sidebar.header("⚙️ Settings")
model_path = st.sidebar.text_input("Model Path", DEFAULT_MODEL_PATH)

video_mode = st.sidebar.radio("Video Source", ["Sample Video", "Upload Video"])
if video_mode == "Sample Video":
    # coba pakai DEFAULT_VIDEO_PATH; jika tidak ada, cari .mp4 di subfolder
    if os.path.exists(DEFAULT_VIDEO_PATH):
        video_path = DEFAULT_VIDEO_PATH
    else:
        matches = glob.glob("**/*.mp4", recursive=True)
        if matches:
            video_path = matches[0]
        else:
            st.sidebar.error(f"No sample .mp4 found under {BASE_DIR}")
            st.stop()
else:
    uploaded = st.sidebar.file_uploader("Upload a video", type=["mp4","avi"])
    if uploaded:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(uploaded.read())
        video_path = tmp.name
    else:
        st.sidebar.warning("Please upload a video.")
        st.stop()

conf = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, DEFAULT_CONFIDENCE, 0.05)

# — Drawable Canvas ———————————————————————————————————————
st.sidebar.markdown("Draw entry & exit lines below:")
canvas_result = st_canvas(
    fill_color=CANVAS_FILL_COLOR,
    stroke_width=CANVAS_STROKE_WIDTH,
    stroke_color="red",
    background_color=CANVAS_BG_COLOR,  # kosong → putih
    width=CANVAS_WIDTH,
    height=CANVAS_HEIGHT,
    drawing_mode="line",
    key="canvas",
)

objects   = canvas_result.json_data["objects"] if canvas_result.json_data else []
entry_ln, exit_ln = extract_line_coords(objects)
if not entry_ln or not exit_ln:
    st.warning("Draw at least two lines (entry & exit).")
    st.stop()

# — Main Loop —————————————————————————————————————————————
if st.button("▶️ Play Video"):
    frame_pl  = st.empty()
    col_vid, col_stat = st.columns((0.7, 0.3))
    fps_m    = col_stat.empty()
    ent_m    = col_stat.empty()
    ext_m    = col_stat.empty()
    spd_m    = col_stat.empty()
    pie_m    = col_stat.empty()

    idx = 0
    all_e = set()
    all_x = set()
    all_s = []
    t0    = time.time()

    while True:
        out = run_counter(video_path, model_path, conf, entry_ln, exit_ln, idx)
        if out[1] is None:
            break  # no more frames

        metrics, frame_bgr, pie_fig, ids_e, ids_x, speeds = out
        all_e.update(ids_e)
        all_x.update(ids_x)
        all_s += speeds

        # real‐time FPS
        fps = idx / (time.time() - t0 + 1e-6)

        # BGR → RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # tampil video + statistik
        with col_vid:
            frame_pl.image(frame_rgb, use_column_width=True)
        with col_stat:
            fps_m.metric("FPS", f"{fps:.2f}")
            ent_m.metric("Vehicle Enter", len(all_e))
            ext_m.metric("Vehicle Exit",  len(all_x))
            avg_sp = sum(all_s)/len(all_s) if all_s else 0.0
            spd_m.metric("Avg Speed (km/h)", f"{avg_sp:.2f}")
            pie_m.plotly_chart(pie_fig, use_container_width=True)

        idx += 1
        time.sleep(1 / max(fps, 1))

    st.success("✅ Processing completed!")
else:
    st.info("Press ▶️ Play Video to start counting.")