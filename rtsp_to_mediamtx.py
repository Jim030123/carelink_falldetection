import cv2
import subprocess
import time

# =========================
# CCTV RTSP URL
# =========================
RTSP_URL = "rtsp://JJFAMILY:JJAN31237252@192.168.1.151:554/stream1"

# =========================
# MediaMTX RTMP path
# =========================
RTMP_URL = "rtmp://127.0.0.1/cam1"

# =========================
# Open RTSP via OpenCV
# =========================
cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print("❌ Cannot open RTSP stream")
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS) or 25
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

print(f"RTSP opened: {width}x{height} @ {fps}fps")

# =========================
# Start FFmpeg (stdin → RTMP)
# =========================
ffmpeg = subprocess.Popen(
    [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", "30",
        "-i", "-",

        # 🔽 关键：降分辨率
        "-vf", "scale=1280:720",

        # HLS / Flutter 友好编码
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-profile:v", "baseline",
        "-level", "3.1",
        "-preset", "veryfast",
        "-tune", "zerolatency",

        "-f", "flv",
        "rtmp://127.0.0.1/cam1",
    ],
    stdin=subprocess.PIPE
)

# =========================
# Main loop
# =========================
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ RTSP frame read failed, retrying...")
            time.sleep(0.5)
            continue

        try:
            ffmpeg.stdin.write(frame.tobytes())
        except BrokenPipeError:
            print("❌ FFmpeg pipe broken")
            break

except KeyboardInterrupt:
    print("Stopping...")

cap.release()
ffmpeg.stdin.close()
ffmpeg.wait()
