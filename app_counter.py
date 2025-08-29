import cv2
from ultralytics import YOLO
import numpy as np
import time
from absl import app, flags
import sys

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
    page_title="Vehicle Counter and Speed Estimation",        
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

def show_region(frame, points):
    points = points.reshape((-1, 1, 2))
    cv2.polylines(frame, np.int32([points]), isClosed=True, color=(0, 0, 255), thickness=3)  

def transform_points(perspective, points):
    if points.size == 0:
        return points

    reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
    transformed_points = cv2.perspectiveTransform(
            reshaped_points, perspective)
    
    return transformed_points.reshape(-1, 2)     

def add_position_time(track_id, current_position, track_data):
    track_time = time.time()

    if(track_id in track_data):
        track_data[track_id]['position'].append(current_position)
    else:
        track_data[track_id] = {'position' : [current_position], 'time': track_time}

def calculate_speed(start, end, start_time, fps):
    now = time.time()

    move_time = now - start_time    
    distance = abs(end - start)    
    distance = distance / 10

    fps_ratio = 30 / fps

    # m/s
    speed = (distance / move_time) * fps_ratio
    # Convert m/s to km/h
    speed = speed * 3.6 

    return speed

def speed_estimation(vehicle_position, speed_region, perspective_region, track_data, track_id, text, fps):   
    min_x = int(np.amin(speed_region[:, 0]))
    max_x = int(np.amax(speed_region[:, 0]))

    min_y = int(np.amin(speed_region[:, 1]))
    max_y = int(np.amax(speed_region[:, 1]))

    speed = 0

    if((vehicle_position[0] in range(min_x, max_x)) and (vehicle_position[1] in range(min_y, max_y))):
        points = np.array([[vehicle_position[0], vehicle_position[1]]], 
                        dtype=np.float32)                                

        point_transform = transform_points(perspective_region, points)                
        
        add_position_time(track_id, int(point_transform[0][1]), track_data)                

        if(len(track_data[track_id]['position']) > 5):
            start_position = track_data[track_id]['position'][0]
            end_position = track_data[track_id]['position'][-1]
            start_estimate = track_data[track_id]['time']

            speed = calculate_speed(start_position, end_position, start_estimate, fps)

            speed_string = "{:.2f}".format(speed)
            text = text + " - " + speed_string + " km/h"
    
    return text, speed
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

    ## Dashboard    
    st.title("Vehicle Counter and Speed Estimation")    
        # Tambahkan canvas
        # Ambil satu frame acak dari video untuk background canvas
    cap_preview = cv2.VideoCapture(FLAGS.video)
    ret, frame_preview = cap_preview.read()
    cap_preview.release()
    if not ret:
        frame_preview = np.zeros((450, 800, 3), dtype=np.uint8)

    # Konversi ke PIL image (canvas butuh PIL)
    from PIL import Image
    img_pil = Image.fromarray(cv2.cvtColor(cv2.resize(frame_preview, (800, 450)), cv2.COLOR_BGR2RGB))
        # --- PATCH: Ambil ukuran frame asli video upload ---
    cap_size = cv2.VideoCapture(FLAGS.video)
    ret_vid, frame_video = cap_size.read()
    if not ret_vid:
        frame_video = np.zeros((450, 800, 3), dtype=np.uint8)
    frame_height, frame_width = frame_video.shape[:2]
    cap_size.release()

    st.subheader("Adjust Entry and Exit Lines")
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
    
    # Initialize the video capture
    video_input = FLAGS.video
    # Check if the video input is an integer (webcam index)
    if video_input.isdigit():
        video_input = int(video_input)
        cap = cv2.VideoCapture(video_input)
    else:
        cap = cv2.VideoCapture(video_input)

    if not cap.isOpened():
        print('Error: Unable to open video source.')
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

    vehicle_class_ids = [1, 2, 3, 5, 7]

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
    
    # entry_line = {
    #     'x1' : 160, 
    #     'y1' : 558,  
    #     'x2' : 708,  
    #     'y2' : 558,          
    # }
    # exit_line = {
    #     'x1' : 1155, 
    #     'y1' : 558,  
    #     'x2' : 1718,  
    #     'y2' : 558,          
    # }
    offset = 25
    ##

    ## Speed Estimation
    # Region 1 (Left)
    speed_region_1 = np.float32([[393, 478], 
					[760, 482],
					[611, 838], 
					[-135, 777]]) 
    width = 150
    height = 270
    target_1 = np.float32([[0, 0], 
					[width, 0],
					[width, height], 
					[0, height]])
    
    # Region 2 (Right)
    speed_region_2 = np.float32([[1074, 500], 
					[1422, 490],
					[2021, 812], 
					[1377, 932]])     
    width = 120
    height = 270
    target_2 = np.float32([[0, 0], 
					[width, 0],
					[width, height], 
					[0, height]])
    
    # Transform Perspective
    perspective_region_1 = cv2.getPerspectiveTransform(speed_region_1, target_1)    
    perspective_region_2 = cv2.getPerspectiveTransform(speed_region_2, target_2)

    track_data = {}

    start_time = time.time()    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break        
                
        frames.append(frame)

        # Counting Line
        cv2.line(frame, (entry_line['x1'], entry_line['y1']), (exit_line['x2'], exit_line['y2']), (0, 127, 255), 3)

        # Speed Region
        show_region(frame, speed_region_1)
        show_region(frame, speed_region_2)

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

            # Average Speed
            speeds = []  

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

                    ## Speed Estimation
                    # Region 1                  
                    vehicle_position = (center_x, y2)
                    text, vehicle_speed = speed_estimation(vehicle_position, speed_region_1, perspective_region_1, track_data, track_id, text, fps)   

                    if(vehicle_speed > 0):
                        speeds.append(vehicle_speed)
                    
                    # Region 2  
                    vehicle_position = (center_x, y1)
                    text, vehicle_speed = speed_estimation(vehicle_position, speed_region_2, perspective_region_2, track_data, track_id, text, fps)            

                    if(vehicle_speed > 0):
                        speeds.append(vehicle_speed)
                    ##
                    
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
                        # Ada crossing!
                        if side_prev < side_curr:
                            # Bergerak searah vektor entry_line (masuk)
                            if (int(track_id) not in entered_vehicle_ids and class_id in vehicle_class_ids):
                                vehicle_entry_count[class_id] += 1
                                entered_vehicle_ids.append(int(track_id))
                        else:
                            # Bergerak berlawanan vektor entry_line (keluar)
                            if (int(track_id) not in exited_vehicle_ids and class_id in vehicle_class_ids):
                                vehicle_exit_count[class_id] += 1
                                exited_vehicle_ids.append(int(track_id))

                    # Update posisi sebelumnya
                    prev_centers[track_id] = (center_x, center_y)                     
            
            # Show Counters
            show_counter(frame, "Vehicle Enter", class_names, vehicle_entry_count, 10)
            show_counter(frame, "Vehicle Exit", class_names, vehicle_exit_count, 1710)                         

            resized = cv2.resize(frame, (800, 450))

            # Average Speed        
            total_speed = sum(speeds)
            num_speeds = len(speeds)
            average_speed = 0
            if(num_speeds > 0):
                average_speed = total_speed / num_speeds

            average_speed = "{:.2f} km/h".format(average_speed)  

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
                    # create 4 columns
                    col_fps, col_enter, col_exit, col_speed = st.columns(4)     

                    col_fps.metric(label="FPS", value=fps)
                    col_enter.metric(label="Vehicle Enter", value=all_vehicle_entry_count)
                    col_exit.metric(label="Vehicle Exit", value=all_vehicle_exit_count)
                    col_speed.metric(label="Average Speed", value=average_speed)

                    # Show Result Video                
                    st.image(resized, channels="BGR", use_column_width=True)           
                
                with col[1]:                
                    # # Pie Chart                
                    pie_colors = ['rgb(33, 75, 99)', 'rgb(79, 129, 102)', 'rgb(151, 179, 100)',
                        'rgb(175, 49, 35)', 'rgb(36, 73, 147)']
                    
                    labels = ["bicycle", "car", "motorcycle", "bus", "truck"]

                    total_vehicle_counts = list(vehicle_count.values())
                    fig = go.Figure(data=[go.Pie(labels=labels, values=total_vehicle_counts, textinfo='text', 
                                                marker_colors=pie_colors)])
                    fig.update_layout(title_text='Vehicle Statistics', height=400, margin=dict(b=0))
                    frame_id = int(time.time() * 1000)  # atau bisa pakai variabel frame_count jika ada
                    st.plotly_chart(fig, use_container_width=True, key=f"vehicle_stats_chart_{frame_id}")            

                    # 
                    all_vehicle_count = all_vehicle_entry_count + all_vehicle_exit_count                    
                    for index, (key, value) in enumerate(vehicle_count.items()):
                        number_of_vehicle = '{:,.0f}'.format(value)
                        vehicle_percentage = 0
                        if(all_vehicle_count > 0):
                            vehicle_percentage = (value / all_vehicle_count) * 100 

                        formatted_percentage = "{:.2f}".format(vehicle_percentage).rstrip("0").rstrip(".")                    
                        st.text(f'{labels[index]} ( {number_of_vehicle} / {formatted_percentage}% )')                                                        

    # Release video capture
    cap.release()        

if __name__ == '__main__':
    app.run(main)