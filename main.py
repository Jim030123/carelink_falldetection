import cv2
import subprocess
import time
from collections import deque

import numpy as np
from ultralytics import YOLO


# =====================================================
# CONFIG
# =====================================================
RTSP_URL = "rtsp://JJFAMILY:JJAN31237252@192.168.1.151:554/stream1"
RTMP_URL = "rtmp://127.0.0.1/cam1"

# 固定输出尺寸（4K 源必须先降）
OUT_W, OUT_H = 1280, 720
OUT_FPS = 25

# YOLO ONNX
YOLO_MODEL = "yolov8n.onnx"   # ⚠️ ONNX，不是 .pt
YOLO_EVERY_N_FRAMES = 5       # 再快一点（你可以 3~6 调）

# Fall detection（稳定优先）
FALL_RATIO_THRESHOLD = 1.2    # box_w / box_h
FALL_FRAMES_REQUIRED = 5
FALL_CLEAR_SEC = 3.0


# =====================================================
def main():
    print("🚀 Starting Fall Detection (ONNX FINAL STABLE)")

    # ---------------- YOLO (ONNX) ----------------
    # ⚠️ 注意：ONNX 模型【不要】 .to("cuda")
    model = YOLO(YOLO_MODEL, task="detect")

    # ---------------- Open RTSP ----------------
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("❌ Cannot open RTSP stream")
        return

    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    in_fps = cap.get(cv2.CAP_PROP_FPS) or 25

    print(f"RTSP opened: {in_w}x{in_h} @ {in_fps}fps")

    # ---------------- FFmpeg (stdin → RTMP) ----------------
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel", "error",

            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{OUT_W}x{OUT_H}",
            "-r", str(OUT_FPS),
            "-i", "-",

            "-c:v", "libx264",
            "-profile:v", "main",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-pix_fmt", "yuv420p",

            # 固定 GOP，防 Flutter / HLS 花屏
            "-g", str(OUT_FPS),
            "-keyint_min", str(OUT_FPS),
            "-sc_threshold", "0",

            "-f", "flv",
            RTMP_URL,
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    # ---------------- Fall buffers ----------------
    fall_ratio_buf = deque(maxlen=FALL_FRAMES_REQUIRED)
    fall_detected = False
    fall_time = None

    frame_idx = 0
    last_results = []

    # =====================================================
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ RTSP frame read failed, retrying...")
                time.sleep(0.2)
                continue

            frame_idx += 1
            now = time.time()

            # 🔑 4K 源统一降分辨率（防花屏 & 加速）
            frame = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)

            # 清除 fall 状态
            if fall_detected and now - fall_time > FALL_CLEAR_SEC:
                fall_detected = False
                fall_ratio_buf.clear()

            # ---------------- YOLO (ONNX + GPU, throttled) ----------------
            if frame_idx % YOLO_EVERY_N_FRAMES == 0:
                last_results = model.predict(
                    frame,
                    device=0,        # 👈 ONNX 用 GPU 在这里指定
                    imgsz=640,       # 再快一点，fall detection 足够
                    conf=0.25,
                    verbose=False
                )

            # ---------------- Person detection ----------------
            for r in last_results:
                for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
                    if int(cls) != 0:
                        continue  # 只看 person

                    x1, y1, x2, y2 = map(int, box)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(OUT_W, x2), min(OUT_H, y2)

                    box_w = x2 - x1
                    box_h = y2 - y1
                    if box_h <= 0:
                        continue

                    # ========= 稳定版 Fall Detection =========
                    ratio = box_w / box_h
                    fall_ratio_buf.append(ratio)

                    if len(fall_ratio_buf) == fall_ratio_buf.maxlen:
                        avg_ratio = sum(fall_ratio_buf) / len(fall_ratio_buf)
                        if avg_ratio > FALL_RATIO_THRESHOLD:
                            fall_detected = True
                            fall_time = now

                    # draw person box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # ---------------- Draw status ----------------
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

            # ---------------- Push to RTMP ----------------
            ffmpeg.stdin.write(frame.tobytes())

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        cap.release()
        ffmpeg.stdin.close()
        ffmpeg.wait()


# =====================================================
if __name__ == "__main__":
    main()
