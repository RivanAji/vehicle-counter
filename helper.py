"""
helper.py – Utility & core logic untuk Vehicle Counter
"""

import cv2
import numpy as np
import time
from ultralytics import YOLO
import plotly.graph_objects as go

from settings import (
    VEHICLE_CLASS_IDS,
    CANVAS_WIDTH, CANVAS_HEIGHT,
    DEFAULT_MODEL_PATH
)

# Simpan history posisi per track id
GLOBAL_TRACK_DATA = {}

def point_side_of_line(x, y, x1, y1, x2, y2):
    return (x2 - x1)*(y - y1) - (y2 - y1)*(x - x1)

def scale_coords_from_canvas_to_frame(line, fw, fh):
    """Skala coords dari canvas ke koordinat frame video."""
    return {
        'x1': int(line['x1'] * fw / CANVAS_WIDTH),
        'y1': int(line['y1'] * fh / CANVAS_HEIGHT),
        'x2': int(line['x2'] * fw / CANVAS_WIDTH),
        'y2': int(line['y2'] * fh / CANVAS_HEIGHT),
    }

def show_region(frame, entry_line, exit_line):
    """Gambar entry & exit line di frame OpenCV."""
    # merah & jingga dalam BGR
    red   = (0, 0, 255)
    orange= (0,165,255)
    cv2.line(frame, (entry_line['x1'], entry_line['y1']),
                  (entry_line['x2'], entry_line['y2']), red, 2)
    cv2.line(frame, (exit_line ['x1'], exit_line ['y1']),
                  (exit_line ['x2'], exit_line ['y2']), orange, 2)
    return frame

def add_position_time(tid, ypos, track_data):
    rec = track_data.setdefault(tid, {'pos': [], 't': []})
    rec['pos'].append(ypos)
    rec['t'].append(time.time())

def calculate_speed(y1, y2, times, pixel_to_meter=1.0):
    if len(times) < 2 or times[-1] == times[0]:
        return 0.0
    dy = abs(y2 - y1) * pixel_to_meter
    dt = times[-1] - times[0]
    return (dy / dt) * 3.6  # m/s → km/h

def extract_line_coords(objects):
    """
    Ambil dua lines dari canvas JSON:
    objek[0] → entry, objek[1] → exit
    """
    if not objects or len(objects) < 2:
        return None, None
    def _c(o):
        return {'x1': int(o['x1']), 'y1': int(o['y1']),
                'x2': int(o['x2']), 'y2': int(o['y2'])}
    return _c(objects[0]), _c(objects[1])

def run_counter(
    video_path, model_path, conf,
    entry_line, exit_line, frame_idx
):
    # load model YOLO sekali  
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        # kembalikan empty outputs
        empty = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
        return {"enter":0,"exit":0,"avg_speed":0}, empty, go.Figure(), [], [], []

    fh, fw = frame.shape[:2]
    entry_f = scale_coords_from_canvas_to_frame(entry_line, fw, fh)
    exit_f  = scale_coords_from_canvas_to_frame(exit_line,  fw, fh)

    frame = show_region(frame, entry_f, exit_f)

    # deteksi (tanpa tracking → no .boxes.id)
    results = model(frame, conf=conf)[0]

    # ambil arrays
    bboxes = results.boxes.xyxy.cpu().numpy().astype(int)
    classes= results.boxes.cls.cpu().numpy().astype(int)
    # guard untuk id
    if hasattr(results.boxes, "id") and results.boxes.id is not None:
        ids = results.boxes.id.cpu().numpy().astype(int)
    else:
        ids = np.arange(len(classes))

    vehicle_enter = {c:0 for c in VEHICLE_CLASS_IDS}
    vehicle_exit  = {c:0 for c in VEHICLE_CLASS_IDS}
    entry_ids = []
    exit_ids  = []
    speeds    = []

    for (box, cls, tid) in zip(bboxes, classes, ids):
        if cls not in VEHICLE_CLASS_IDS:
            continue
        x1,y1,x2,y2 = box
        cx, cy = (x1+x2)//2, (y1+y2)//2

        prev = GLOBAL_TRACK_DATA.get(tid, {'pos':[(cx,cy)], 't':[time.time()]})
        side_prev = point_side_of_line(
            prev['pos'][-1][0], prev['pos'][-1][1], **entry_f
        )
        side_cur  = point_side_of_line(cx, cy, **entry_f)
        if side_prev * side_cur < 0:
            if side_prev < side_cur and tid not in entry_ids:
                vehicle_enter[cls] += 1; entry_ids.append(tid)
            elif side_prev > side_cur and tid not in exit_ids:
                vehicle_exit[cls]  += 1; exit_ids.append(tid)

        # speed
        add_position_time(tid, cy, GLOBAL_TRACK_DATA)
        hist = GLOBAL_TRACK_DATA[tid]
        sp = calculate_speed(hist['pos'][0], hist['pos'][-1], hist['t'])
        speeds.append(sp)

        # draw box & label
        cv2.rectangle(frame, (x1,y1),(x2,y2),(255,0,0),2)
        cv2.putText(
            frame, f"{cls}-{tid} {sp:.1f}km/h",
            (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2
        )
        # reset history to only last
        GLOBAL_TRACK_DATA[tid] = {'pos': [cy], 't': [hist['t'][-1]]}

    # metrics
    total_enter = sum(vehicle_enter.values())
    total_exit  = sum(vehicle_exit.values())
    avg_speed   = sum(speeds)/len(speeds) if speeds else 0.0

    # pie chart
    labels = ["bicycle","car","motorcycle","bus","truck"]
    values = [vehicle_enter[c]+vehicle_exit[c] for c in VEHICLE_CLASS_IDS]
    pie = go.Figure(data=[go.Pie(labels=labels, values=values)])

    cap.release()
    return (
        {"enter":total_enter,"exit":total_exit,"avg_speed":avg_speed},
        frame, pie, entry_ids, exit_ids, speeds
    )