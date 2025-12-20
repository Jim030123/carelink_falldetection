import cv2
import subprocess
import time
from collections import deque

from ultralytics import YOLO


# =====================================================
# CONFIG
# =====================================================
CAMERA_INDEX = 1                 # 0 = default camera
RTMP_URL = "rtmp://127.0.0.1/cam1"

OUT_W, OUT_H = 1280, 720
OUT_FPS = 25

YOLO_MODEL = "yolov8n.onnx"
YOLO_EVERY_N_FRAMES = 1         # 越小越灵敏，越大越稳

FALL_RATIO_THRESHOLD = 1.2
FALL_FRAMES_REQUIRED = 5
FALL_CLEAR_SEC = 3.0


# =====================================================
def main():
    print("🚀 Starting Fall Detection (FINAL STABLE VERSION)")

    # ---------------- YOLO (ONNX / GPU) ----------------
    model = YOLO(YOLO_MODEL, task="detect")

    # ---------------- Open Camera ----------------
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, OUT_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, OUT_H)
    cap.set(cv2.CAP_PROP_FPS, OUT_FPS)

    print(f"Camera opened: {OUT_W}x{OUT_H} @ {OUT_FPS}fps")

    # ---------------- FFmpeg (REALTIME + NVENC) ----------------
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel", "warning",

            "-re",                      # ⭐ 实时节奏（防快放）
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{OUT_W}x{OUT_H}",
            "-r", str(OUT_FPS),
            "-i", "-",

            "-c:v", "h264_nvenc",
            "-preset", "fast",          # 最稳 NVENC
            "-profile:v", "baseline",   # RTMP / Flutter 最兼容
            "-pix_fmt", "yuv420p",

            "-g", str(OUT_FPS),
            "-keyint_min", str(OUT_FPS),

            "-f", "flv",
            RTMP_URL,
        ],
        stdin=subprocess.PIPE
    )

    # ---------------- Fall buffers ----------------
    fall_ratio_buf = deque(maxlen=FALL_FRAMES_REQUIRED)
    fall_detected = False
    fall_time = None

    frame_idx = 0
    last_results = []

    frame_interval = 1.0 / OUT_FPS

    # =====================================================
    try:
        while True:
            loop_start = time.time()

            ret, frame = cap.read()
            if not ret:
                print("⚠️ Camera frame read failed")
                time.sleep(0.1)
                continue

            frame_idx += 1
            now = time.time()

            frame = cv2.resize(frame, (OUT_W, OUT_H))

            # ---- clear fall state ----
            if fall_detected and now - fall_time > FALL_CLEAR_SEC:
                fall_detected = False
                fall_ratio_buf.clear()

            # ---- YOLO (throttled) ----
            if frame_idx % YOLO_EVERY_N_FRAMES == 0:
                last_results = model.predict(
                    frame,
                    device=0,
                    imgsz=640,
                    conf=0.25,
                    verbose=False
                )

            # ---- Person + fall detection ----
            for r in last_results:
                for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
                    if int(cls) != 0:
                        continue

                    x1, y1, x2, y2 = map(int, box)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(OUT_W, x2), min(OUT_H, y2)

                    w = x2 - x1
                    h = y2 - y1
                    if h <= 0:
                        continue

                    ratio = w / h
                    fall_ratio_buf.append(ratio)

                    if len(fall_ratio_buf) == fall_ratio_buf.maxlen:
                        if sum(fall_ratio_buf) / len(fall_ratio_buf) > FALL_RATIO_THRESHOLD:
                            fall_detected = True
                            fall_time = now

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            if fall_detected:
                cv2.putText(
                    frame,
                    "FALL DETECTED",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3
                )

            # ---- Push to RTMP ----
            try:
                ffmpeg.stdin.write(frame.tobytes())
            except BrokenPipeError:
                print("❌ FFmpeg pipe broken (RTMP server down)")
                break

            # ---- FPS sync (防忽快忽慢) ----
            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        cap.release()
        if ffmpeg.stdin:
            ffmpeg.stdin.close()
        ffmpeg.wait()


# =====================================================
if __name__ == "__main__":
    main()
