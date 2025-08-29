import cv2
from ultralytics import YOLO
import numpy as np
import time
from absl import app, flags
import sys
import os
import tempfile

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import plotly.graph_objects as go

# Argument
FLAGS = flags.FLAGS
flags.DEFINE_string("model", "yolo11m.pt", "YOLO11 Model")
flags.DEFINE_string("video", "highway.mp4", "Video")
flags.DEFINE_string("conf", "0.2", "Confidence Threshold")

### Configuration
st.set_page_config(
    page_title="UrbanVision",
    layout="wide",
)

#######################
# CSS styling
st.markdown("""
<style>

[data-testid="stMetric"] {
    background-color: #393939;
    text-align: center;
    padding: 15px 0;
    border-radius: 10px;
    color: #FFFFFF;
}

[data-testid="stMetricLabel"] {
    display: flex;
    justify-content: center;
    align-items: center;
    color: #FFFFFF;
}

section[data-testid="stSidebar"] { position: relative; }
.sidebar-footer {
    position: absolute;
    left: 12px;
    bottom: 12px;
    font-size: 12px;
    color: #9aa0a6;
    opacity: 0.9;
}

</style>
""", unsafe_allow_html=True)

def point_side_of_line(x, y, x1, y1, x2, y2):
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)

def scale_coords_from_canvas_to_frame(line, frame_width, frame_height, canvas_width=800, canvas_height=450):
    x1_scaled = int(line['x1'] * frame_width / canvas_width)
    y1_scaled = int(line['y1'] * frame_height / canvas_height)
    x2_scaled = int(line['x2'] * frame_width / canvas_width)
    y2_scaled = int(line['y2'] * frame_height / canvas_height)
    return {'x1': x1_scaled, 'y1': y1_scaled, 'x2': x2_scaled, 'y2': y2_scaled}

def show_counter(frame, title, class_names, vehicle_count, x_init):
    overlay = frame.copy()

    # Show Counters
    y_init = 100
    gap = 30

    alpha = 0.5

    cv2.rectangle(overlay, (x_init - 5, y_init - 35), (x_init + 200, 265), (0, 255, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    cv2.putText(frame, title, (x_init, y_init - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    for vehicle_id, count in vehicle_count.items():
        y_init += gap

        vehicle_name = class_names[vehicle_id]
        vehicle_count = "%.3i" % (count)
        cv2.putText(frame, vehicle_name, (x_init, y_init), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)            
        cv2.putText(frame, vehicle_count, (x_init + 145, y_init), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

def scale_line_from_default_resolution(line, frame_width, frame_height, base_width=1920, base_height=1080):
    x1_scaled = int(line['x1'] * frame_width / base_width)
    y1_scaled = int(line['y1'] * frame_height / base_height)
    x2_scaled = int(line['x2'] * frame_width / base_width)
    y2_scaled = int(line['y2'] * frame_height / base_height)
    return {'x1': x1_scaled, 'y1': y1_scaled, 'x2': x2_scaled, 'y2': y2_scaled}

prev_centers = {}

def main(_argv):
    # Avoid conflicts with Streamlit
    _argv = [sys.argv[0]] 

    st.markdown("<h1 style='margin-bottom:0'>UrbanVision</h1>", unsafe_allow_html=True)

    # Sidebar: logo on top of credit text
    sidebar_logo_candidates = [
        os.path.join(os.path.dirname(__file__), "logo.png"),
        "vehicle-counter/logo.png",
    ]
    sidebar_logo_path = next((p for p in sidebar_logo_candidates if os.path.exists(p)), None)
    if sidebar_logo_path:
        st.sidebar.image(sidebar_logo_path, use_container_width=False)
    st.sidebar.markdown("<div style='font-size:12px;color:#9aa0a6;margin-bottom:8px'>developed by Rivan — URBANESHA</div>", unsafe_allow_html=True)

    # ==== Source selection (Webcam / Upload / Path) ====
    st.sidebar.header("Input Source")
    src_mode = st.sidebar.radio("Source", ["Webcam", "Upload", "Path", "URL Stream"], index=1)

    webcam_idx = None
    uploaded_bytes = None
    video_path = None
    stream_url = None

    if src_mode == "Webcam":
        webcam_idx = st.sidebar.number_input("Webcam index", 0, 10, 0, step=1)
    elif src_mode == "Upload":
        up = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
        if up is not None:
            uploaded_bytes = up.read()
    elif src_mode == "Path":
        video_path = st.sidebar.text_input("Local video path", value=FLAGS.video)
    elif src_mode == "URL Stream":
        stream_url = st.sidebar.text_input("Stream URL (RTSP/HTTP, e.g. m3u8/rtsp)", "")

    # --- Options ---
    enable_people = st.sidebar.checkbox("Count people (enter/exit)", value=True, help="Include 'person' in crossing counts")
    
  # Helper to open a cv2.VideoCapture from selected source
    def _open_capture(webcam_index, bytes_buf, path_str, stream_url):
        if webcam_index is not None:
            return cv2.VideoCapture(int(webcam_index)), None
        if bytes_buf is not None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(bytes_buf)
            tmp.flush()
            tmp.close()
            return cv2.VideoCapture(tmp.name), tmp.name
        if path_str:
            return cv2.VideoCapture(path_str), None
        if stream_url:
            return cv2.VideoCapture(stream_url), None
        return cv2.VideoCapture(), None

    # Ambil satu frame dari sumber terpilih untuk background canvas
    cap_preview, _tmp_preview = _open_capture(webcam_idx, uploaded_bytes, video_path, stream_url)
    ret, frame_preview = cap_preview.read()
    cap_preview.release()
    if not ret:
        frame_preview = np.zeros((450, 800, 3), dtype=np.uint8)
        st.info("Preview is blank because the selected source has not provided a frame yet. Choose/Upload a video or provide a valid stream URL.")

    # Konversi ke PIL image (canvas butuh PIL)
    from PIL import Image
    img_pil = Image.fromarray(cv2.cvtColor(cv2.resize(frame_preview, (800, 450)), cv2.COLOR_BGR2RGB))
    # --- PATCH: Ambil ukuran frame asli video dari sumber terpilih ---
    cap_size, _tmp_size = _open_capture(webcam_idx, uploaded_bytes, video_path, stream_url)
    ret_vid, frame_video = cap_size.read()
    if not ret_vid:
        frame_video = np.zeros((450, 800, 3), dtype=np.uint8)
    frame_height, frame_width = frame_video.shape[:2]
    cap_size.release()

    st.subheader("Adjust Entry and Exit Lines")
    try:
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=3,
            stroke_color="red",
            background_image=img_pil,
            height=450,
            width=800,
            drawing_mode="line",
            key="canvas",
        )
    except Exception:
        # Fallback for Streamlit versions where drawable-canvas cannot convert background image
        st.warning("Canvas background image is not supported on this Streamlit version. You can still draw the counting line on a blank canvas.")
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=3,
            stroke_color="red",
            background_color="#000000",
            height=450,
            width=800,
            drawing_mode="line",
            key="canvas_nobg",
        )
    # Ambil garis dari canvas untuk entry_line dan exit_line
    lines = canvas_result.json_data["objects"] if (canvas_result and canvas_result.json_data) else []

    def extract_line_coords(line_object):
        # PATCH: Handle x1/y1/x2/y2 from streamlit-drawable-canvas 'line' object
        if all(k in line_object for k in ["x1", "y1", "x2", "y2", "left", "top"]):
            # x1/y1/x2/y2 are RELATIVE to center; need to convert to ABSOLUTE coordinates
            cx, cy = line_object["left"], line_object["top"]
            x1 = cx + line_object["x1"]
            y1 = cy + line_object["y1"]
            x2 = cx + line_object["x2"]
            y2 = cy + line_object["y2"]
            return {'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)}
        elif "path" in line_object:
            path = line_object["path"]
            x1, y1 = path[0]
            x2, y2 = path[1]
            return {'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)}
        elif "points" in line_object:
            path = line_object["points"]
            x1, y1 = path[0]
            x2, y2 = path[1]
            return {'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)}
        elif "data" in line_object and "start" in line_object["data"] and "end" in line_object["data"]:
            path = [line_object["data"]["start"], line_object["data"]["end"]]
            x1, y1 = path[0]
            x2, y2 = path[1]
            return {'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)}
        else:
            raise KeyError("Cannot find line coordinates in canvas object:", line_object)

    # Default fallback jika user belum gambar sama sekali
    default_entry = {
        'x1' : 160, 'y1' : 558, 'x2' : 708, 'y2' : 558,
    }
    default_exit = {
        'x1' : 1155, 'y1' : 558, 'x2' : 1718, 'y2' : 558,
    }

    if len(lines) == 0:
        # PATCH: scaling default line dari 1920x1080 ke frame video ---
        entry_line = scale_line_from_default_resolution(default_entry, frame_width, frame_height)
        exit_line  = scale_line_from_default_resolution(default_exit, frame_width, frame_height)
    else:
        entry_line = extract_line_coords(lines[0])
        exit_line  = extract_line_coords(lines[0])
        entry_line = scale_coords_from_canvas_to_frame(entry_line, frame_width, frame_height)
        exit_line  = scale_coords_from_canvas_to_frame(exit_line,  frame_width, frame_height)

    if len(lines) == 0:
        st.warning("Please draw a line on the canvas above.")
    elif len(lines) == 1:
        st.info("The line is used for both entry & exit (one line, two-way detection).")
    else:
        st.info("Only the first line is used.")

    result_elem = st.empty()
    
    # Initialize the video capture from selected source
    cap, _tmp_run = _open_capture(webcam_idx, uploaded_bytes, video_path, stream_url)

    if not cap.isOpened():
        print('Error: Unable to open video source.')
        return
    if src_mode == "URL Stream" and not stream_url:
        st.warning("Please provide a valid stream URL (e.g., RTSP or HLS m3u8).")
        return

    if src_mode == "Upload" and uploaded_bytes is None:
        st.warning("Please upload a video file first.")
        return

    # Load YOLO model    
    model = YOLO(FLAGS.model)  # Load the YOLO11 model

    batch_size = 2  # Batch size for parallel processing
    frames = []       

    # Class names     
    classes_path = "coco.names"
    with open(classes_path, "r") as f:
        class_names = f.read().strip().split("\n")    
    
    # Create a list of random colors to represent each class
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(class_names), 3)) 

    ## Vehicle Counter
    # Helper Variable
    entered_vehicle_ids = []
    exited_vehicle_ids = []

    vehicle_class_ids = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck
    people_class_ids  = [0] if enable_people else []  # person
    tracked_class_ids = set(vehicle_class_ids + people_class_ids)

    vehicle_entry_count = {
        1: 0,  # bicycle
        2: 0,  # car
        3: 0,  # motorcycle
        5: 0,  # bus
        7: 0   # truck
    }
    vehicle_exit_count = {
        1: 0,  # bicycle
        2: 0,  # car
        3: 0,  # motorcycle
        5: 0,  # bus
        7: 0   # truck
    }
    people_entry_count = {0: 0} if enable_people else {}
    people_exit_count  = {0: 0} if enable_people else {}

    offset = 25

    start_time = time.time()    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break        
                
        frames.append(frame)

        # Counting Line
        cv2.line(frame, (entry_line['x1'], entry_line['y1']), (exit_line['x2'], exit_line['y2']), (0, 127, 255), 3)

        # Once the batch size has been reached, run tracking.
        if len(frames) == batch_size:            
            # Perform Object Tracking using YOLO
            results = model.track(frames, persist=True, tracker="bytetrack.yaml", conf=float(FLAGS.conf), verbose=False)    
            
            frames = []  # Empty the batch after processing               

            # FPS Calculation
            end_time = time.time()
            fps = 2 / (end_time - start_time)
            fps = float("{:.2f}".format(fps))           

            start_time = end_time

            if results[0].boxes.id is not None:                                   
                boxes = results[0].boxes.xyxy.int().cpu().tolist()
                class_ids = results[0].boxes.cls.cpu().tolist()
                track_ids = results[0].boxes.id.int().cpu().tolist()                                
                
                for box, track_id, class_id in zip(boxes, track_ids, class_ids):                
                    x1, y1, x2, y2 = box                

                    color = colors[int(class_id)]
                    B, G, R = map(int, color)                

                    text = f"{track_id} - {class_names[int(class_id)]}"

                    center_x = int((x1 + x2) / 2 )
                    center_y = int((y1 + y2) / 2 )

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
                    cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(text) * 12, y1), (B, G, R), -1)
                    cv2.putText(frame, text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    # Simpan previous center (jika belum ada, set ke current)
                    if track_id not in prev_centers:
                        prev_centers[track_id] = (center_x, center_y)
                    prev_x, prev_y = prev_centers[track_id]

                    side_prev = point_side_of_line(prev_x, prev_y, entry_line['x1'], entry_line['y1'], entry_line['x2'], entry_line['y2'])
                    side_curr = point_side_of_line(center_x, center_y, entry_line['x1'], entry_line['y1'], entry_line['x2'], entry_line['y2'])

                    if side_prev * side_curr < 0:
                        # crossing happened
                        moving_enter = side_prev < side_curr
                        tid = int(track_id)

                        # vehicles
                        if class_id in vehicle_class_ids:
                            if moving_enter:
                                if tid not in entered_vehicle_ids:
                                    vehicle_entry_count[class_id] += 1
                                    entered_vehicle_ids.append(tid)
                            else:
                                if tid not in exited_vehicle_ids:
                                    vehicle_exit_count[class_id] += 1
                                    exited_vehicle_ids.append(tid)

                        # people
                        if class_id in people_class_ids:
                            if moving_enter:
                                people_entry_count[0] += 1
                            else:
                                people_exit_count[0] += 1

                    # Update posisi sebelumnya
                    prev_centers[track_id] = (center_x, center_y)                     
            
            # Show Counters
            show_counter(frame, "Vehicle Enter", class_names, vehicle_entry_count, 10)
            show_counter(frame, "Vehicle Exit", class_names, vehicle_exit_count, 1710)                         

            resized = cv2.resize(frame, (800, 450))

            # Total Entered and Exited Vehicles
            all_vehicle_entry_count = sum(vehicle_entry_count.values())
            all_vehicle_exit_count = sum(vehicle_exit_count.values())            
            
            # Combine the counts
            vehicle_count = {}
            for key in vehicle_entry_count:
                vehicle_count[key] = vehicle_entry_count[key] + vehicle_exit_count[key]                  

            with result_elem.container():
                # create 2 columns
                col = st.columns((0.65, 0.35), gap='medium')

                with col[0]:
                    # create 3 columns
                    col_fps, col_enter, col_exit = st.columns(3)

                    col_fps.metric(label="FPS", value=fps)
                    col_enter.metric(label="Vehicle Enter", value=all_vehicle_entry_count)
                    col_exit.metric(label="Vehicle Exit", value=all_vehicle_exit_count)

                    # Show Result Video
                    st.image(resized, channels="BGR", use_container_width=True)

                    # People metrics row (immediately after vehicle metrics)
                    if enable_people:
                        p1, p2 = st.columns(2)
                        all_people_entry = sum(people_entry_count.values())
                        all_people_exit  = sum(people_exit_count.values())
                        p1.metric(label="People Enter", value=all_people_entry)
                        p2.metric(label="People Exit", value=all_people_exit)

                with col[1]:
                    # # Pie Chart
                    pie_colors = ['rgb(33, 75, 99)', 'rgb(79, 129, 102)', 'rgb(151, 179, 100)',
                        'rgb(175, 49, 35)', 'rgb(255, 174, 66)']

                    labels = ["bicycle", "car", "motorcycle", "bus", "truck"]
                    values = [vehicle_count.get(k, 0) for k in vehicle_class_ids]
                    if enable_people:
                        labels = ["person"] + labels
                        values = [people_entry_count.get(0, 0) + people_exit_count.get(0, 0)] + values
                    fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='text',
                                                marker_colors=pie_colors)])
                    fig.update_layout(title_text='Vehicle Statistics', height=400, margin=dict(b=0))
                    frame_id = int(time.time() * 1000)  # atau bisa pakai variabel frame_count jika ada
                    st.plotly_chart(fig, use_container_width=True, key=f"vehicle_stats_chart_{frame_id}")

                    grand_total = sum(values)
                    for lab, val in zip(labels, values):
                        number_str = '{:,.0f}'.format(val)
                        pct = (val / grand_total * 100) if grand_total > 0 else 0
                        pct_str = "{:.2f}".format(pct).rstrip("0").rstrip(".")
                        st.text(f'{lab} ( {number_str} / {pct_str}% )')

    # Release video capture
    cap.release()        

    # Cleanup temporary files created for uploaded video (if any)
    try:
        if '_tmp_preview' in locals() and _tmp_preview:
            os.unlink(_tmp_preview)
    except Exception:
        pass
    try:
        if '_tmp_size' in locals() and _tmp_size:
            os.unlink(_tmp_size)
    except Exception:
        pass
    try:
        if '_tmp_run' in locals() and _tmp_run:
            os.unlink(_tmp_run)
    except Exception:
        pass


if __name__ == '__main__':
    app.run(main)