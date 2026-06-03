from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import os, datetime

os.makedirs("challans", exist_ok=True)

def generate_pdf(track_id, plate_text, speed_kmph, frame_img_path=None):
    filename = f"challans/challan_{track_id}_{plate_text}.pdf"
    c = canvas.Canvas(filename, pagesize=A4)
    w, h = A4

    c.setFillColor(colors.HexColor("#1a1a2e"))
    c.rect(0, h - 100, w, 100, fill=1, stroke=0)

    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(w / 2, h - 45, "TRAFFIC VIOLATION NOTICE")
    c.setFont("Helvetica", 11)
    c.drawCentredString(w / 2, h - 68, "Issued by Automated ANPR System")

    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(60, h - 140, "Challan Details")

    c.setFont("Helvetica", 12)
    details = [
        ("Vehicle Number",  plate_text),
        ("Violation",       "Overspeeding"),
        ("Recorded Speed",  f"{speed_kmph} km/h"),
        ("Speed Limit",     "60 km/h"),
        ("Date & Time",     datetime.datetime.now().strftime("%d %b %Y, %I:%M %p")),
        ("Camera ID",       "CAM_01"),
        ("Track ID",        str(track_id)),
    ]
    y = h - 175
    for label, value in details:
        c.setFont("Helvetica-Bold", 11)
        c.drawString(60, y, f"{label}:")
        c.setFont("Helvetica", 11)
        c.drawString(220, y, value)
        y -= 28

    if frame_img_path and os.path.exists(frame_img_path):
        c.drawImage(frame_img_path, 60, y - 160, width=200, height=130)

    c.setFont("Helvetica", 9)
    c.setFillColor(colors.grey)
    c.drawCentredString(w / 2, 40, "This is a system-generated challan. Pay within 30 days.")
    c.save()
    return filename