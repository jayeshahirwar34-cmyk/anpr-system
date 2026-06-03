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

# --- 1. Premium UI Configuration & Custom CSS ---
st.set_page_config(
    page_title="NexTrack ANPR Intelligence", 
    page_icon="🚓", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Custom premium styling via CSS injection
st.markdown("""
    <style>
    /* Premium Metric Cards styling */
    div[data-testid="metric-container"] {
        background-color: #1E1E2E;
        border: 1px solid #303046;
        padding: 5% 5% 5% 10%;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        border-left: 6px solid #4CAF50;
    }
    /* Style the main application header */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: -webkit-linear-gradient(#4CAF50, #00FFCC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: -5px;
    }
    .sub-title {
        color: #A0A0B8;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. Model Initialization & Resource Caching ---
@st.cache_resource
def load_models():
    model = YOLO("best.pt")
    reader = easyocr.Reader(['en'], gpu=False)
    setup_tables()
    return model, reader

model, reader = load_models()

# Clean regular expressions for typical number plates
def clean_plate(text):
    text = text.upper().replace(" ", "").replace("-", "")
    match = re.search(r'[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}', text)
    return match.group() if match else text[:10]

# Secure stream processing downloader for YouTube URLs
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

# --- 3. Sidebar Control Infrastructure ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2083/2083161.png", width=70)
    st.markdown("### System Control Panel")
    st.caption("Configure ingestion feeds and deployment rules.")
    
    input_method = st.radio("Select Ingestion Stream:", ["Upload Local File", "YouTube Live Link"])
    
    video_path = None
    if input_method == "Upload Local File":
        uploaded_file = st.file_uploader("Upload Target Asset (MP4, JPG, PNG)", type=['mp4', 'jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4' if uploaded_file.name.endswith('.mp4') else '.jpg')
            tfile.write(file_bytes)
            tfile.close()
            video_path = tfile.name
            is_video = uploaded_file.name.endswith('.mp4')
            
    else:
        yt_url = st.text_input("Target YouTube URL:")
        is_video = True
        if yt_url and st.button("🔗 Connect Stream"):
            with st.spinner("Downloading stream metadata safely..."):
                try:
                    video_path = download_youtube_video(yt_url)
                    st.success("Remote stream cached locally!")
                except Exception as e:
                    st.error(f"Stream ingestion failed: {e}")

    st.divider()
    st.markdown("⚙️ **Enforcement Thresholds**")
    SPEED_LIMIT = st.slider("Enforcement Speed Boundary (km/h)", 20, 120, 60)

# --- 4. Main Core UI Presentation ---
st.markdown('<p class="main-title">NexTrack Intelligence</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Automated Number Plate Recognition & Speed Enforcement Terminal</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📹 Terminal Operations", "📊 Analytical Ledger & Telemetry"])

# --- TAB 1: OPERATIONAL WORKFLOW ---
with tab1:
    if video_path:
        if st.button("🚀 Boot Vision Core Engine", use_container_width=True):
            os.makedirs("challans", exist_ok=True)
            
            # --- STATIC OBJECT ANALYSIS ---
            if not is_video:
                st.info("Analyzing static perspective...")
                frame = cv2.imread(video_path)
                detections = model(frame, verbose=False)[0].boxes
                for box in detections:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    if conf > 0.4:
                        crop = frame[max(0,y1):y2, max(0,x1):x2]
                        if crop.size > 0:
                            result = reader.readtext(crop)
                            plate = clean_plate(result[0][1]) if result else "UNKNOWN"
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, plate, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)

            # --- DYNAMIC STREAM MULTI-OBJECT TRACKING ---
            else:
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(cap.get(3)), int(cap.get(4))))

                tracker = DeepSort(max_age=40)
                seen, track_entry, challaned = {}, {}, set()
                master_log = []  # Master data object containing all sequential vehicle sightings
                
                # Layout distribution for operations window
                col1, col2 = st.columns([5, 3])
                
                with col1:
                    st.markdown("#### Primary Camera Stream")
                    video_placeholder = st.empty()
                    progress_bar = st.progress(0)
                
                with col2:
                    st.markdown("#### Real-time Traffic Ledger")
                    log_placeholder = st.empty()

                frame_num = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    frame_num += 1
                    
                    if frame_num % 5 == 0:
                        progress_bar.progress(min(frame_num / total_frames, 1.0))

                    detections = []
                    for box in model(frame, verbose=False)[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        if conf > 0.4:
                            detections.append(([x1, y1, x2-x1, y2-y1], conf, "plate"))

                    tracks = tracker.update_tracks(detections, frame=frame)

                    for track in tracks:
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
                                master_log.insert(0, {
                                    "Time (Frame)": frame_num, 
                                    "Track ID": tid, 
                                    "Plate": plate, 
                                    "Speed": "Calculating...", 
                                    "Status": "Passed UI Log"
                                })

                        plate = seen.get(tid, "UNKNOWN")
                        speed = estimate_speed(track, fps, track_entry)
                        color = (0, 255, 0)

                        if speed:
                            for log_entry in master_log:
                                if log_entry["Track ID"] == tid and log_entry["Speed"] == "Calculating...":
                                    log_entry["Speed"] = f"{speed} km/h"

                        if speed and speed > SPEED_LIMIT and tid not in challaned:
                            color = (0, 0, 255)
                            pdf = generate_pdf(tid, plate, speed)
                            insert_challan(tid, plate, speed, pdf)
                            challaned.add(tid)
                            master_log.insert(0, {
                                "Time (Frame)": frame_num, 
                                "Track ID": tid, 
                                "Plate": plate, 
                                "Speed": f"{speed} km/h", 
                                "Status": "🚨 VIOLATION"
                            })

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{plate} " + (f"{speed}km/h" if speed else "")
                        cv2.putText(frame, label, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # Update live tracking display framework
                    if frame_num % 2 == 0:
                        video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                    
                    # Safe UI Frame refreshing without generating Duplicate Element IDs
                    if frame_num % 10 == 0 and master_log:
                        df_logs = pd.DataFrame(master_log[:15])
                        log_placeholder.dataframe(df_logs, use_container_width=True, hide_index=True)

                    out.write(frame)

                cap.release()
                out.release()
                progress_bar.empty()
                
                st.success(f"Execution Terminal Concluded: {len(seen)} tracks mapped, {len(challaned)} offenses cataloged.")
                
                # System Download Frameworks
                d_col1, d_col2 = st.columns(2)
                with d_col1:
                    with open(out_path, "rb") as file:
                        st.download_button(label="📥 Download Processed Tracking Video", data=file, file_name="anpr_output.mp4", mime="video/mp4", use_container_width=True)
                with d_col2:
                    if master_log:
                        csv_log = pd.DataFrame(master_log).to_csv(index=False).encode('utf-8')
                        st.download_button(label="📄 Export Full System Log (CSV)", data=csv_log, file_name="traffic_ledger.csv", mime="text/csv", use_container_width=True)
    else:
        st.info("👈 System waiting for feed configuration inside the Control Panel.")


# --- TAB 2: METRICS & TELEMETRY DASHBOARD ---
with tab2:
    if st.button("🔄 Poll Analytics Database Layer", use_container_width=True):
        st.rerun()

    try:
        conn = get_conn()
        
        # Pull telemetry variables safely
        total_v = pd.read_sql("SELECT COUNT(*) n FROM vehicles", conn).iloc[0]['n']
        total_c = pd.read_sql("SELECT COUNT(*) n FROM challans", conn).iloc[0]['n']
        avg_spd = pd.read_sql("SELECT ROUND(AVG(speed_kmph),1) s FROM challans", conn).iloc[0]['s'] or 0

        # Metrics Layout
        m1, m2, m3 = st.columns(3)
        m1.metric("🚗 Cumulative Traffic Tracked", total_v)
        m2.metric("🚨 Total System Violations", total_c)
        m3.metric("⚡ Mean Enforcement Velocity", f"{avg_spd} km/h")
        
        st.divider()

        # Database Presentation Panels
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("System Traffic History")
            st.caption("Active permanent record of all vehicle interactions mapped by the system.")
            df_all = pd.read_sql("SELECT id as 'ID Reference', plate_text as 'Identified Plate', first_seen as 'Timestamp' FROM vehicles ORDER BY first_seen DESC LIMIT 50", conn)
            st.dataframe(df_all, use_container_width=True, height=450)

        with col2:
            st.subheader("Violation Archive")
            st.caption("Cataloged instances of boundary acceleration infractions.")
            df_chal = pd.read_sql("SELECT vehicle_id as 'ID Reference', plate_text as 'Offending Plate', speed_kmph as 'Velocity', issued_at as 'Infraction Time' FROM challans ORDER BY issued_at DESC LIMIT 50", conn)
            st.dataframe(df_chal, use_container_width=True, height=450)

    except Exception as e:
        st.error("Connect an active analytical database core to populate metrics.")
