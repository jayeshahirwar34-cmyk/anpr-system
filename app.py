import streamlit as st
import cv2, os, re, tempfile
import numpy as np
import pandas as pd
import yt_dlp
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import easyocr

from database import setup_tables, insert_vehicle, insert_sighting, insert_challan, get_conn
from speed import estimate_speed
from challan import generate_pdf

# --- 1. Premium UI Configuration ---
st.set_page_config(page_title="NexTrack ANPR", page_icon="🚓", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    div[data-testid="metric-container"] {
        background-color: #1E1E2E;
        border: 1px solid #303046;
        padding: 5% 5% 5% 10%;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        border-left: 6px solid #4CAF50;
    }
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: -webkit-linear-gradient(#4CAF50, #00FFCC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: -5px;
    }
    .sub-title { color: #A0A0B8; font-size: 1.1rem; margin-bottom: 25px; }
    </style>
""", unsafe_allow_html=True)

# --- Check for Critical Files Before Starting ---
if not os.path.exists("best.pt"):
    st.error("🚨 CRITICAL ERROR: 'best.pt' is missing. Please upload your trained YOLO model to the repository.")
    st.stop()

# --- 2. Initialization ---
@st.cache_resource
def load_models():
    model = YOLO("best.pt")
    reader = easyocr.Reader(['en'], gpu=False)
    setup_tables()
    return model, reader

model, reader = load_models()

def clean_plate(text):
    text = text.upper().replace(" ", "").replace("-", "")
    match = re.search(r'[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}', text)
    return match.group() if match else text[:10]

def download_youtube_video(url):
    ydl_opts = {'format': 'best[ext=mp4]/best', 'outtmpl': tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name, 'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        return ydl.prepare_filename(ydl.extract_info(url, download=True))

# --- 3. Sidebar Configuration ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2083/2083161.png", width=70)
    st.markdown("### System Control Panel")
    
    input_method = st.radio("Select Ingestion Stream:", ["Upload Local File", "YouTube Live Link"])
    
    video_path = None
    if input_method == "Upload Local File":
        uploaded_file = st.file_uploader("Upload Target Asset (MP4, JPG, PNG)", type=['mp4', 'jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            # Clear previous session state when a new file is uploaded
            if 'last_uploaded' not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
                st.session_state.last_uploaded = uploaded_file.name
                st.session_state.pop('process_complete', None)
                
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4' if uploaded_file.name.endswith('.mp4') else '.jpg')
            tfile.write(uploaded_file.read())
            tfile.close() # Critical for Windows/Linux read access
            video_path = tfile.name
            is_video = uploaded_file.name.endswith('.mp4')
            
    else:
        yt_url = st.text_input("Target YouTube URL:")
        is_video = True
        if yt_url and st.button("🔗 Connect Stream"):
            with st.spinner("Downloading stream securely..."):
                video_path = download_youtube_video(yt_url)
                st.session_state.pop('process_complete', None) # Reset state

    st.divider()
    SPEED_LIMIT = st.slider("Enforcement Speed Boundary (km/h)", 20, 120, 60)

# --- 4. Main Application UI ---
st.markdown('<p class="main-title">NexTrack Intelligence</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Automated Number Plate Recognition & Speed Enforcement Terminal</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📹 Terminal Operations", "📊 Analytical Ledger"])

with tab1:
    if video_path:
        os.makedirs("challans", exist_ok=True)
        
        # --- IMAGE LOGIC (Auto-runs without button) ---
        if not is_video:
            st.info("Analyzing static perspective...")
            frame = cv2.imread(video_path)
            
            if frame is None:
                st.error("⚠️ OpenCV failed to read this image. It may be corrupted.")
            else:
                detections = model(frame, verbose=False)[0].boxes
                for box in detections:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    if float(box.conf[0]) > 0.4:
                        crop = frame[max(0,y1):y2, max(0,x1):x2]
                        if crop.size > 0:
                            result = reader.readtext(crop)
                            plate = clean_plate(result[0][1]) if result else "UNKNOWN"
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, plate, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)

        # --- VIDEO LOGIC ---
        else:
            if st.button("🚀 Boot Vision Core Engine", use_container_width=True):
                cap = cv2.VideoCapture(video_path)
                
                # SAFETY NET 1: Catch OpenCV Codec Failures
                if not cap.isOpened():
                    st.error("⚠️ OpenCV failed to open the video. The file might be corrupted or in an unsupported codec format.")
                else:
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                    out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(cap.get(3)), int(cap.get(4))))

                    tracker = DeepSort(max_age=40)
                    seen, track_entry, challaned = {}, {}, set()
                    master_log = []
                    
                    col1, col2 = st.columns([5, 3])
                    with col1:
                        video_placeholder = st.empty()
                        progress_bar = st.progress(0)
                    with col2:
                        log_placeholder = st.empty()

                    frame_num = 0
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: break
                        frame_num += 1
                        
                        if frame_num % 5 == 0:
                            progress_bar.progress(min(frame_num / max(total_frames, 1), 1.0))

                        detections = [([int(b[0]), int(b[1]), int(b[2]-b[0]), int(b[3]-b[1])], float(c), "plate") 
                                      for b, c in zip(model(frame, verbose=False)[0].boxes.xyxy.tolist(), 
                                                      model(frame, verbose=False)[0].boxes.conf.tolist()) if c > 0.4]

                        for track in tracker.update_tracks(detections, frame=frame):
                            if not track.is_confirmed(): continue
                            tid = track.track_id
                            x1, y1, x2, y2 = map(int, track.to_ltrb())
                            cx, cy = (x1+x2)//2, (y1+y2)//2

                            insert_sighting(tid, frame_num, cx, cy)

                            if tid not in seen:
                                crop = frame[max(0,y1):y2, max(0,x1):x2]
                                if crop.size > 0:
                                    result = reader.readtext(crop)
                                    plate = clean_plate(result[0][1]) if result else "UNKNOWN"
                                    seen[tid] = plate
                                    insert_vehicle(tid, plate)
                                    master_log.insert(0, {"Time (Frame)": frame_num, "Track ID": tid, "Plate": plate, "Speed": "...", "Status": "Logged"})

                            plate, speed = seen.get(tid, "UNKNOWN"), estimate_speed(track, fps, track_entry)
                            color = (0, 0, 255) if speed and speed > SPEED_LIMIT and tid not in challaned else (0, 255, 0)

                            if speed:
                                for log_entry in master_log:
                                    if log_entry["Track ID"] == tid and log_entry["Speed"] == "...":
                                        log_entry["Speed"] = f"{speed} km/h"
                                if color == (0, 0, 255):
                                    pdf = generate_pdf(tid, plate, speed)
                                    insert_challan(tid, plate, speed, pdf)
                                    challaned.add(tid)
                                    master_log.insert(0, {"Time (Frame)": frame_num, "Track ID": tid, "Plate": plate, "Speed": f"{speed} km/h", "Status": "🚨 VIOLATION"})

                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, f"{plate} {speed if speed else ''}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        if frame_num % 2 == 0: video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                        if frame_num % 10 == 0 and master_log: log_placeholder.dataframe(pd.DataFrame(master_log[:15]), use_container_width=True, hide_index=True)

                        out.write(frame)

                    cap.release()
                    out.release()
                    progress_bar.empty()
                    
                    # SAFETY NET 2: Save to Session State so downloads don't disappear
                    st.session_state['process_complete'] = True
                    st.session_state['out_path'] = out_path
                    st.session_state['master_log'] = master_log
                    st.session_state['summary'] = f"Complete: {len(seen)} mapped, {len(challaned)} infractions."

            # --- Render Downloads Outside the Button Block ---
            if st.session_state.get('process_complete'):
                st.success(st.session_state['summary'])
                d_col1, d_col2 = st.columns(2)
                with d_col1:
                    with open(st.session_state['out_path'], "rb") as file:
                        st.download_button(label="📥 Download Annotated Video", data=file, file_name="anpr_output.mp4", mime="video/mp4", use_container_width=True)
                with d_col2:
                    if st.session_state['master_log']:
                        csv_log = pd.DataFrame(st.session_state['master_log']).to_csv(index=False).encode('utf-8')
                        st.download_button(label="📄 Export Traffic Ledger (CSV)", data=csv_log, file_name="traffic_ledger.csv", mime="text/csv", use_container_width=True)
    else:
        st.info("👈 System waiting for feed configuration inside the Control Panel.")


with tab2:
    if st.button("🔄 Poll Database", use_container_width=True): st.rerun()
    try:
        conn = get_conn()
        total_v = pd.read_sql("SELECT COUNT(*) n FROM vehicles", conn).iloc[0]['n']
        total_c = pd.read_sql("SELECT COUNT(*) n FROM challans", conn).iloc[0]['n']
        avg_spd = pd.read_sql("SELECT ROUND(AVG(speed_kmph),1) s FROM challans", conn).iloc[0]['s'] or 0

        m1, m2, m3 = st.columns(3)
        m1.metric("🚗 Tracked Vehicles", total_v)
        m2.metric("🚨 Total Violations", total_c)
        m3.metric("⚡ Mean Velocity", f"{avg_spd} km/h")
        
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("System Traffic History")
            st.dataframe(pd.read_sql("SELECT plate_text as 'Identified Plate', first_seen as 'Timestamp' FROM vehicles ORDER BY first_seen DESC LIMIT 50", conn), use_container_width=True, height=450)
        with col2:
            st.subheader("Violation Archive")
            st.dataframe(pd.read_sql("SELECT plate_text as 'Offending Plate', speed_kmph as 'Velocity', issued_at as 'Time' FROM challans ORDER BY issued_at DESC LIMIT 50", conn), use_container_width=True, height=450)
    except Exception as e:
        st.error("Connect an active database core to populate metrics.")
