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

# --- 2. Safe Model Initialization ---
@st.cache_resource
def load_models():
    # Fallback to YOLO base model if 'best.pt' isn't uploaded yet
    model_path = "best.pt" if os.path.exists("best.pt") else "yolov8n.pt"
    model = YOLO(model_path)
    reader = easyocr.Reader(['en'], gpu=False)
    try:
        setup_tables()
    except Exception as e:
        st.warning(f"Database schema auto-setup delayed. Please check Aiven connection. Error: {e}")
    return model, reader, model_path

model, reader, active_model = load_models()

def clean_plate(text):
    text = text.upper().replace(" ", "").replace("-", "")
    match = re.search(r'[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}', text)
    return match.group() if match else text[:10]

# High-Performance YouTube Downloader
def download_youtube_video(url):
    ydl_opts = {
        'format': 'best[ext=mp4]/best', 
        'outtmpl': tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name, 
        'quiet': True,
        'no_warnings': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)

# --- 3. Sidebar Control Panel ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2083/2083161.png", width=70)
    st.markdown("### System Control Panel")
    st.info(f"Active Vision Core: {active_model}")
    
    input_method = st.radio("Select Ingestion Stream:", ["Upload Local File", "YouTube Live Link"])
    
    video_path = None
    if input_method == "Upload Local File":
        uploaded_file = st.file_uploader("Upload Target Asset (MP4, JPG, PNG)", type=['mp4', 'jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            # Clear cache if it's a new file
            if 'last_uploaded' not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
                st.session_state.last_uploaded = uploaded_file.name
                st.session_state.pop('process_complete', None)
                
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4' if uploaded_file.name.endswith('.mp4') else '.jpg')
            tfile.write(uploaded_file.read())
            tfile.close() 
            video_path = tfile.name
            is_video = uploaded_file.name.endswith('.mp4')
    else:
        yt_url = st.text_input("Target YouTube URL:")
        is_video = True
        if yt_url and st.button("🔗 Connect Stream"):
            with st.spinner("Downloading YouTube stream securely..."):
                try:
                    video_path = download_youtube_video(yt_url)
                    st.session_state.pop('process_complete', None) 
                    st.success("YouTube video loaded and ready!")
                except Exception as e:
                    st.error(f"YouTube Download Error. Ensure the video is public. Error: {e}")

    st.divider()
    SPEED_LIMIT = st.slider("Enforcement Speed Boundary (km/h)", 20, 120, 60)

# --- 4. Main UI Layout ---
st.markdown('<p class="main-title">NexTrack Intelligence</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Automated Number Plate Recognition & Speed Enforcement Terminal</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📹 Terminal Operations", "📊 Analytical Ledger"])

with tab1:
    if video_path:
        os.makedirs("challans", exist_ok=True)
        
        # --- STATIC IMAGE ANALYSIS ---
        if not is_video:
            st.info("Analyzing static perspective...")
            frame = cv2.imread(video_path)
            if frame is None:
                st.error("⚠️ OpenCV failed to read this image file.")
            else:
                results = model(frame, verbose=False)[0]
                if results.boxes is not None:
                    for box in results.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        if float(box.conf[0]) > 0.4:
                            crop = frame[max(0,y1):y2, max(0,x1):x2]
                            if crop.size > 0:
                                result = reader.readtext(crop)
                                plate = clean_plate(result[0][1]) if result else "UNKNOWN"
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, plate, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)

        # --- DYNAMIC VIDEO ANALYSIS ---
        else:
            if st.button("🚀 Boot Vision Core Engine", use_container_width=True):
                cap = cv2.VideoCapture(video_path)
                
                if not cap.isOpened():
                    st.error("⚠️ OpenCV Core failed to initialize stream reader. Codec may be unsupported.")
                else:
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    st.write(f"📊 **Diagnostic Log:** Stream initialized. Total frames: `{total_frames}` @ `{fps:.2f}` FPS")

                    out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(cap.get(3)), int(cap.get(4))))

                    tracker = DeepSort(max_age=40)
                    seen, track_entry, challaned = {}, {}, set()
                    master_log = []
                    
                    col1, col2 = st.columns([5, 3])
                    with col1:
                        video_placeholder = st.empty()
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                    with col2:
                        log_placeholder = st.empty()

                    frame_num = 0
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: 
                            break
                        frame_num += 1
                        
                        if frame_num % 5 == 0:
                            progress_bar.progress(min(frame_num / max(total_frames, 1), 1.0))
                            status_text.text(f"Processing Frame: {frame_num} / {total_frames}")

                        # --- OPTIMIZED SINGLE-PASS INFERENCE ---
                        detections = []
                        results = model(frame, verbose=False)[0]
                        if results.boxes is not None:
                            for box in results.boxes:
                                conf = float(box.conf[0].item())
                                if conf > 0.4:
                                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "plate"))

                        tracks = tracker.update_tracks(detections, frame=frame)

                        # --- TRACKING & DATABASE LOGIC ---
                        for track in tracks:
                            if not track.is_confirmed(): 
                                continue
                            
                            tid = track.track_id
                            x1, y1, x2, y2 = map(int, track.to_ltrb())
                            cx, cy = (x1+x2)//2, (y1+y2)//2

                            # 1. ALWAYS CREATE THE VEHICLE RECORD FIRST (Fixes IntegrityError)
                            if tid not in seen:
                                crop = frame[max(0,y1):y2, max(0,x1):x2]
                                plate = "UNKNOWN"
                                
                                if crop.size > 0:
                                    result = reader.readtext(crop)
                                    if result:
                                        plate = clean_plate(result[0][1])
                                
                                seen[tid] = plate
                                
                                # Safe database insertion
                                try:
                                    insert_vehicle(tid, plate)
                                except Exception:
                                    pass 
                                
                                master_log.insert(0, {
                                    "Time (Frame)": frame_num, 
                                    "Track ID": tid, 
                                    "Plate": plate, 
                                    "Speed": "Calculating...", 
                                    "Status": "Logged"
                                })

                            # 2. NOW IT IS SAFE TO LOG THE SIGHTING
                            try:
                                insert_sighting(tid, frame_num, cx, cy)
                            except Exception:
                                pass 

                            # 3. SPEED ESTIMATION & CHALLAN GENERATION
                            plate = seen.get(tid, "UNKNOWN")
                            speed = estimate_speed(track, fps, track_entry)
                            color = (0, 255, 0)

                            if speed:
                                for log_entry in master_log:
                                    if log_entry["Track ID"] == tid and log_entry["Speed"] == "Calculating...":
                                        log_entry["Speed"] = f"{speed} km/h"
                                
                                if speed > SPEED_LIMIT and tid not in challaned:
                                    color = (0, 0, 255)
                                    challaned.add(tid)
                                    try:
                                        pdf = generate_pdf(tid, plate, speed)
                                        insert_challan(tid, plate, speed, pdf)
                                    except Exception:
                                        pass
                                    master_log.insert(0, {
                                        "Time (Frame)": frame_num, 
                                        "Track ID": tid, 
                                        "Plate": plate, 
                                        "Speed": f"{speed} km/h", 
                                        "Status": "🚨 VIOLATION"
                                    })

                            # 4. DRAW BOXES & LABELS
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, f"{plate} {f'{speed}km/h' if speed else ''}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        # UI Refreshes (Throttle to prevent crashes)
                        if frame_num % 3 == 0: 
                            video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                        if frame_num % 15 == 0 and master_log: 
                            log_placeholder.dataframe(pd.DataFrame(master_log[:15]), use_container_width=True, hide_index=True)

                        out.write(frame)

                    cap.release()
                    out.release()
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Lock data into session state so it doesn't disappear when downloading
                    st.session_state['process_complete'] = True
                    st.session_state['out_path'] = out_path
                    st.session_state['master_log'] = master_log
                    st.session_state['summary'] = f"Success: {len(seen)} tracks mapped, {len(challaned)} speed violations recorded."

            # --- RENDER DOWNLOAD BUTTONS SAFELY ---
            if st.session_state.get('process_complete'):
                st.success(st.session_state['summary'])
                d_col1, d_col2 = st.columns(2)
                with d_col1:
                    with open(st.session_state['out_path'], "rb") as file:
                        st.download_button(label="📥 Download Processed Tracking MP4", data=file, file_name="anpr_output.mp4", mime="video/mp4", use_container_width=True)
                with d_col2:
                    if st.session_state['master_log']:
                        csv_log = pd.DataFrame(st.session_state['master_log']).to_csv(index=False).encode('utf-8')
                        st.download_button(label="📄 Export Full System Log (CSV)", data=csv_log, file_name="traffic_ledger.csv", mime="text/csv", use_container_width=True)
    else:
        st.info("👈 System waiting for data stream inside the control panel.")

# --- TAB 2: ANALYTICS DASHBOARD ---
with tab2:
    if st.button("🔄 Poll Database Metrics", use_container_width=True): 
        st.rerun()
    try:
        conn = get_conn()
        total_v = pd.read_sql("SELECT COUNT(*) n FROM vehicles", conn).iloc[0]['n']
        total_c = pd.read_sql("SELECT COUNT(*) n FROM challans", conn).iloc[0]['n']
        avg_spd = pd.read_sql("SELECT ROUND(AVG(speed_kmph),1) s FROM challans", conn).iloc[0]['s'] or 0

        m1, m2, m3 = st.columns(3)
        m1.metric("🚗 Tracked Vehicles", total_v)
        m2.metric("🚨 Total Violations", total_c)
        m3.metric("⚡ Mean Violation Velocity", f"{avg_spd} km/h")
        
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("System Traffic History")
            st.dataframe(pd.read_sql("SELECT plate_text as 'Identified Plate', first_seen as 'Timestamp' FROM vehicles ORDER BY first_seen DESC LIMIT 50", conn), use_container_width=True, height=450)
        with col2:
            st.subheader("Violation Archive")
            st.dataframe(pd.read_sql("SELECT plate_text as 'Offending Plate', speed_kmph as 'Velocity', issued_at as 'Time' FROM challans ORDER BY issued_at DESC LIMIT 50", conn), use_container_width=True, height=450)
    except Exception as e:
        st.error(f"Analytical database connection sleeping or unavailable. Check Aiven secrets. Error: {e}")
