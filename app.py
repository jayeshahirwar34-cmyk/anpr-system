import streamlit as st
import cv2, os, re, tempfile
import numpy as np
import pandas as pd
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import easyocr

from database import setup_tables, insert_vehicle, insert_sighting, insert_challan, get_conn
from speed import estimate_speed
from challan import generate_pdf

st.set_page_config(page_title="ANPR System", layout="wide")

# --- Initialize Models (Cached so they only load once) ---
@st.cache_resource
def load_models():
    model = YOLO("best.pt")
    reader = easyocr.Reader(['en'], gpu=False)
    setup_tables() # Ensure DB is ready
    return model, reader

model, reader = load_models()
SPEED_LIMIT = 60

def clean_plate(text):
    text = text.upper().replace(" ", "").replace("-", "")
    match = re.search(r'[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}', text)
    return match.group() if match else text[:10]

# --- UI Setup ---
st.title("🚦 Unified ANPR & Dashboard System")
tab1, tab2 = st.tabs(["📹 Process Media", "📊 Live Dashboard"])

with tab1:
    st.header("Upload Traffic Media")
    uploaded_file = st.file_uploader("Upload an Image (JPG/PNG) or Video (MP4)", type=['mp4', 'jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        is_video = uploaded_file.name.endswith('.mp4')

        if st.button("Start Processing"):
            # Ensure challan directory exists
            os.makedirs("challans", exist_ok=True)

            if not is_video:
                # --- IMAGE PROCESSING ---
                st.info("Processing single image. Note: Speed estimation requires video.")
                nparr = np.frombuffer(file_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                detections = model(frame, verbose=False)[0].boxes
                for box in detections:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    
                    if conf > 0.4:
                        crop = frame[max(0,y1):y2, max(0,x1):x2]
                        if crop.size > 0:
                            result = reader.readtext(crop)
                            raw = result[0][1] if result else "UNKNOWN"
                            plate = clean_plate(raw)
                            
                            # Draw on image
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, plate, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            st.success(f"Detected Plate: {plate}")

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, caption="Processed Image", use_column_width=True)

            else:
                # --- VIDEO PROCESSING ---
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(file_bytes)
                tfile.close()

                cap = cv2.VideoCapture(tfile.name)
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

                tracker = DeepSort(max_age=40)
                seen = {}
                challaned = set()
                track_entry = {}
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                log_container = st.empty()
                logs = []

                frame_num = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    frame_num += 1
                    
                    # Update progress UI
                    progress_bar.progress(min(frame_num / total_frames, 1.0))
                    status_text.text(f"Processing frame {frame_num}/{total_frames}")

                    # YOLO Detect
                    detections = []
                    for box in model(frame, verbose=False)[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        if conf > 0.4:
                            detections.append(([x1, y1, x2-x1, y2-y1], conf, "plate"))

                    # DeepSORT Track
                    tracks = tracker.update_tracks(detections, frame=frame)

                    for track in tracks:
                        if not track.is_confirmed(): continue
                        tid = track.track_id
                        x1, y1, x2, y2 = map(int, track.to_ltrb())
                        cx, cy = (x1+x2)//2, (y1+y2)//2

                        insert_sighting(tid, frame_num, cx, cy)

                        # OCR (Once per track)
                        if tid not in seen:
                            crop = frame[max(0,y1):y2, max(0,x1):x2]
                            if crop.size > 0:
                                result = reader.readtext(crop)
                                raw = result[0][1] if result else ""
                                plate = clean_plate(raw)
                                seen[tid] = plate
                                insert_vehicle(tid, plate)
                                logs.insert(0, f"🚗 Track {tid} Detected — Plate: {plate}")

                        plate = seen.get(tid, "???")
                        speed = estimate_speed(track, fps, track_entry)

                        color = (0, 255, 0)
                        if speed and speed > SPEED_LIMIT and tid not in challaned:
                            color = (0, 0, 255)
                            pdf = generate_pdf(tid, plate, speed)
                            insert_challan(tid, plate, speed, pdf)
                            challaned.add(tid)
                            logs.insert(0, f"🚨 CHALLAN ISSUED: {plate} @ {speed} km/h")

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{plate}" + (f" {speed}km/h" if speed else "")
                        cv2.putText(frame, label, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    out.write(frame)
                    log_container.text_area("Live Logs", "\n".join(logs[:10]), height=200)

                cap.release()
                out.release()
                status_text.text("Processing Complete!")
                st.success(f"Done! {len(seen)} vehicles detected. {len(challaned)} challans issued.")

                # Provide processed video download (Streamlit Cloud often lacks browser H264 support for cv2 generated mp4s)
                with open(out_path, "rb") as file:
                    st.download_button(label="📥 Download Annotated Video", data=file, file_name="anpr_output.mp4", mime="video/mp4")

with tab2:
    st.header("Real-Time Analytics")
    
    if st.button("🔄 Refresh Data"):
        st.rerun()

    try:
        conn = get_conn()
        
        # Metrics
        total_v = pd.read_sql("SELECT COUNT(*) n FROM vehicles", conn).iloc[0]['n']
        total_c = pd.read_sql("SELECT COUNT(*) n FROM challans", conn).iloc[0]['n']
        avg_spd = pd.read_sql("SELECT ROUND(AVG(speed_kmph),1) s FROM challans", conn).iloc[0]['s'] or 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Vehicles", total_v)
        c2.metric("Challans Issued", total_c)
        c3.metric("Avg Violation Speed", f"{avg_spd} km/h")

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Recent Violations")
            df = pd.read_sql("""
                SELECT plate_text as 'Plate', speed_kmph as 'Speed', issued_at as 'Time'
                FROM challans ORDER BY issued_at DESC LIMIT 15
            """, conn)
            st.dataframe(df, use_container_width=True)

        with col2:
            st.subheader("Violations by Hour")
            df2 = pd.read_sql("""
                SELECT HOUR(issued_at) as hour, COUNT(*) as count
                FROM challans GROUP BY hour ORDER BY hour
            """, conn)
            if not df2.empty:
                st.bar_chart(df2.set_index("hour"))

    except Exception as e:
        st.error("Could not connect to database. Please check your credentials.")
        st.code(str(e))