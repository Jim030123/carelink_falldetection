"""
Real-time Fall Detection (RTSP Only)
-----------------------------------
RTSP CCTV
→ YOLOv8 (person detection)
→ MediaPipe Pose (fall detection)
→ FFmpeg (RTMP)
→ MediaMTX
→ HLS (Flutter)
→ FastAPI (/fall)

Windows-ready, production-safe.
"""

import math
import time
import threading
import subprocess
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

from fastapi import FastAPI
import uvicorn

# ======================================================
# CONFIG
# ======================================================

# 🔴 CCTV RTSP (唯一视频来源)
RTSP_URL = "rtsp://JJFAMILY:JJAN31237252@192.168.1.151:554/stream1"

# 推到 MediaMTX（同一台电脑）
RTMP_URL = "rtmp://127.0.0.1/cam1"

# 处理 / 推流分辨率（稳定）
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 20

# ======================================================
# FastAPI - Fall Event API
# ======================================================
app = FastAPI()

fall_state = {
    "fall_detected": False,
    "timestamp": None
}

@app.get("/fall")
def get_fall():
    return fall_state


def start_api():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")


# ======================================================
# Helper Functions
# ======================================================
def calculate_angle(hip, shoulder):
    dy = shoulder[1] - hip[1]
    dx = shoulder[0] - hip[0]
    angle = math.atan2(dy, dx)
    return abs(90 - np.degrees(angle))


# ======================================================
# Main
# ======================================================
def main():
    # --------------------------------------------------
    # Start HTTP API
    # --------------------------------------------------
    threading.Thread(target=start_api, daemon=True).start()

    # --------------------------------------------------
    # Load Models
    # --------------------------------------------------
    print("Loading YOLO model...")
    model = YOLO("yolov8l.pt")

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils

    # --------------------------------------------------
    # Open RTSP (Force TCP, Low Buffer)
    # --------------------------------------------------
    rtsp = RTSP_URL + "?rtsp_transport=tcp&stimeout=5000000"

    cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    if not cap.isOpened():
        print("❌ ERROR: Cannot open RTSP stream")
        return

    print("RTSP connected")

    # --------------------------------------------------
    # FFmpeg → MediaMTX (RTMP)
    # --------------------------------------------------
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg",
            "-re",
            "-fflags", "nobuffer",
            "-flags", "low_delay",

            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
            "-r", str(FPS),
            "-i", "-",

            "-vf", "format=yuv420p",
            "-c:v", "libx264",
            "-profile:v", "baseline",
            "-level", "3.1",
            "-preset", "veryfast",
            "-tune", "zerolatency",
            "-g", "40",

            "-f", "flv",
            RTMP_URL,
        ],
        stdin=subprocess.PIPE
    )

    # --------------------------------------------------
    # Fall Detection State
    # --------------------------------------------------
    angle_buf = deque(maxlen=10)
    hip_y_buf = deque(maxlen=10)
    t_buf = deque(maxlen=10)

    fall_detected = False
    fall_start_time = None
    CLEAR_AFTER = 3.0

    # Thresholds
    ANG_VEL_DIRECT = 400.0
    HIP_VEL_DIRECT = 1.2

    # --------------------------------------------------
    # Pose Pipeline
    # --------------------------------------------------
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ RTSP frame lost, retrying...")
                    time.sleep(0.2)
                    continue

                now = time.time()

                # Auto clear fall
                if fall_detected and fall_start_time:
                    if now - fall_start_time >= CLEAR_AFTER:
                        fall_detected = False
                        fall_state["fall_detected"] = False
                        fall_state["timestamp"] = None
                        fall_start_time = None

                # ------------------------------
                # YOLO: detect person
                # ------------------------------
                results = model(frame, verbose=False)

                for result in results:
                    for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                        if int(cls) != 0:
                            continue  # only person

                        x1, y1, x2, y2 = map(int, box)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                        person = frame[y1:y2, x1:x2]
                        if person.size == 0:
                            continue

                        rgb = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
                        res = pose.process(rgb)
                        if not res.pose_landmarks:
                            continue

                        mp_draw.draw_landmarks(
                            person,
                            res.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS
                        )

                        lm = res.pose_landmarks.landmark

                        sh_l = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                        sh_r = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                        hp_l = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
                        hp_r = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]

                        shoulder = (
                            (sh_l.x + sh_r.x) * person.shape[1] / 2,
                            (sh_l.y + sh_r.y) * person.shape[0] / 2
                        )
                        hip = (
                            (hp_l.x + hp_r.x) * person.shape[1] / 2,
                            (hp_l.y + hp_r.y) * person.shape[0] / 2
                        )

                        angle = calculate_angle(hip, shoulder)
                        hip_y = y1 + hip[1]

                        angle_buf.append(angle)
                        hip_y_buf.append(hip_y)
                        t_buf.append(now)

                        ang_vel = hip_vel = 0.0
                        if len(angle_buf) >= 2:
                            dt = t_buf[-1] - t_buf[-2]
                            if dt > 1e-6:
                                ang_vel = (angle_buf[-1] - angle_buf[-2]) / dt
                                hip_vel = (hip_y_buf[-1] - hip_y_buf[-2]) / dt

                        hip_vel_norm = hip_vel / FRAME_HEIGHT

                        # ------------------------------
                        # Direct Fall Detection
                        # ------------------------------
                        if (
                            abs(ang_vel) >= ANG_VEL_DIRECT
                            or hip_vel_norm >= HIP_VEL_DIRECT
                        ):
                            if not fall_detected:
                                fall_detected = True
                                fall_start_time = now
                                fall_state["fall_detected"] = True
                                fall_state["timestamp"] = now

                        frame[y1:y2, x1:x2] = person
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

                # ------------------------------
                # Push frame to FFmpeg
                # ------------------------------
                try:
                    ffmpeg.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    print("❌ FFmpeg disconnected")
                    break

        except KeyboardInterrupt:
            print("Stopping...")

    cap.release()
    ffmpeg.stdin.close()
    ffmpeg.wait()


if __name__ == "__main__":
    main()
