REAL_DISTANCE_M = 5.0
LINE_Y1 = 200   # top line pixel y
LINE_Y2 = 400   # bottom line pixel y

def estimate_speed(track, fps, track_entry):
    tid = track.track_id
    x1, y1, x2, y2 = track.to_ltrb()
    cy = (y1 + y2) / 2

    # Check if crossing top line
    if cy > LINE_Y1 and tid not in track_entry:
        track_entry[tid] = {"frame": None, "started": True}

    # Record exact frame of crossing top line
    if track_entry.get(tid, {}).get("started") and track_entry[tid]["frame"] is None:
        if cy >= LINE_Y1:
            track_entry[tid]["frame"] = track.age   

    # Check if crossing bottom line
    if track_entry.get(tid, {}).get("frame") and cy >= LINE_Y2:
        frames_taken = track.age - track_entry[tid]["frame"]
        if frames_taken > 0:
            time_sec = frames_taken / fps
            speed_mps = REAL_DISTANCE_M / time_sec
            speed_kmph = speed_mps * 3.6
            track_entry.pop(tid, None)   # reset to prevent memory leaks
            return round(speed_kmph, 1)
    return None