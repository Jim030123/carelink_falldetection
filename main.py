"""
Real-time fall detection (camera/live) using YOLO + MediaPipe Pose.

This script replaces the previous file-based pipeline and runs real-time detection from a camera by default.
Usage:
  python main.py --source 0            # use camera 0
  python main.py --source 0            # use camera 0
  python main.py --source in_video.mp4 # use a video file
  python main.py --source 0 --output out.mp4  # optionally save processed output
Press 'q' in the OpenCV window to quit.
"""

import math
import time
import argparse
import numpy as np
import cv2
from ultralytics import YOLO
import mediapipe as mp


def calculate_angle(shoulder_center, hip_center):
    dy = shoulder_center[1] - hip_center[1]
    dx = shoulder_center[0] - hip_center[0]
    angle = math.atan2(dy, dx)
    return abs(90 - np.degrees(angle))


def classify_posture(torso_angle, standing_threshold=10, lying_threshold=60):
    if torso_angle < standing_threshold:
        return "Standing"
    elif torso_angle > lying_threshold:
        return "Lying Down"
    else:
        return "Falling"


def to_source(val):
    try:
        return int(val)
    except Exception:
        return val


def main():
    parser = argparse.ArgumentParser(description='Real-time fall detection (YOLO + MediaPipe)')
    parser.add_argument('--source', default='0', help='Camera index (0,1,...) or path to video file')
    parser.add_argument('--model', default='yolov8l.pt', help='YOLO model path')
    parser.add_argument('--output', default=None, help='Optional output file to save processed video')
    # fall-detection tuning parameters
    parser.add_argument('--lying-threshold', type=float, default=60.0, help='Torso angle (deg) above which is considered lying')
    parser.add_argument('--ang-vel-threshold', type=float, default=150.0, help='Angular velocity threshold (deg/sec) to consider sudden fall')
    parser.add_argument('--hip-vel-threshold', type=float, default=0.6, help='Hip vertical velocity threshold (fraction of frame height per second)')
    parser.add_argument('--sustain-sec', type=float, default=0.2, help='Seconds the fall condition must sustain to trigger alert')
    args = parser.parse_args()

    source = to_source(args.source)

    print('Loading model...', args.model)
    model = YOLO(args.model)

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print('ERROR: cannot open source', source)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        if not writer.isOpened():
            print('WARNING: could not open video writer for', args.output)
            writer = None

    window = 'Fall Detection - press q to quit'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    falling_count = 0
    fall_detected = False
    # temporal buffers for smoothing and velocity estimates
    from collections import deque
    angle_buf = deque(maxlen=10)
    hip_y_buf = deque(maxlen=10)
    t_buf = deque(maxlen=10)
    frame_counter = 0
    t0 = time.time()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_counter += 1

                results = model(frame)

                for result in results:
                    for bbox, cls in zip(result.boxes.xyxy, result.boxes.cls):
                        if int(cls) != 0:
                            continue
                        x1, y1, x2, y2 = map(int, bbox)
                        x1 = max(0, x1); y1 = max(0, y1)
                        x2 = min(frame.shape[1]-1, x2); y2 = min(frame.shape[0]-1, y2)

                        person_bbox = frame[y1:y2, x1:x2]
                        if person_bbox.size == 0:
                            continue

                        person_rgb = cv2.cvtColor(person_bbox, cv2.COLOR_BGR2RGB)
                        person_results = pose.process(person_rgb)

                        if person_results.pose_landmarks:
                            mp_drawing.draw_landmarks(person_bbox, person_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                            # compute simple torso angle using shoulders and hips if available
                            try:
                                landmarks = person_results.pose_landmarks.landmark
                                shoulders = [
                                    (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * person_bbox.shape[1], landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * person_bbox.shape[0]),
                                    (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * person_bbox.shape[1], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * person_bbox.shape[0])
                                ]
                                hips = [
                                    (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * person_bbox.shape[1], landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * person_bbox.shape[0]),
                                    (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * person_bbox.shape[1], landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * person_bbox.shape[0])
                                ]

                                shoulder_center = ((shoulders[0][0] + shoulders[1][0]) / 2, (shoulders[0][1] + shoulders[1][1]) / 2)
                                hip_center = ((hips[0][0] + hips[1][0]) / 2, (hips[0][1] + hips[1][1]) / 2)

                                torso_angle = calculate_angle(hip_center, shoulder_center)
                                posture = classify_posture(torso_angle, standing_threshold=10, lying_threshold=args.lying_threshold)

                                # compute global hip y (frame coordinates)
                                hip_y_global = y1 + hip_center[1]
                                now = time.time()

                                # push to buffers
                                angle_buf.append(torso_angle)
                                hip_y_buf.append(hip_y_global)
                                t_buf.append(now)

                                # compute angular velocity and hip vertical velocity
                                ang_vel = 0.0
                                hip_vel = 0.0
                                if len(angle_buf) >= 2:
                                    dt = t_buf[-1] - t_buf[-2]
                                    if dt > 1e-6:
                                        ang_vel = (angle_buf[-1] - angle_buf[-2]) / dt
                                        hip_vel = (hip_y_buf[-1] - hip_y_buf[-2]) / dt

                                hip_vel_norm = hip_vel / float(height)

                                # display metrics
                                cv2.putText(frame, f'Posture: {posture}', (x1, max(0, y1-30)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                                cv2.putText(frame, f'Angle: {torso_angle:.1f}deg', (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
                                cv2.putText(frame, f'AngVel: {ang_vel:.1f}d/s', (x1, max(0, y1-50)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)

                                ang_fall = abs(ang_vel) >= args.ang_vel_threshold
                                hip_fall = hip_vel_norm >= args.hip_vel_threshold

                                is_fall_frame = (posture == 'Falling' or torso_angle >= args.lying_threshold) and (ang_fall or hip_fall)

                                # update falling_count using sustain frames logic
                                if is_fall_frame:
                                    falling_count += 1
                                else:
                                    falling_count = max(0, falling_count - 1)

                                min_frames = max(1, int(args.sustain_sec * fps))
                                if falling_count >= min_frames:
                                    fall_detected = True
                                elif posture == 'Standing' and falling_count == 0:
                                    fall_detected = False
                            except Exception:
                                pass

                        frame[y1:y2, x1:x2] = person_bbox
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # alert text
                if fall_detected:
                    cv2.putText(frame, 'FALL DETECTED', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

                # FPS display (simple)
                if frame_counter % 10 == 0:
                    t1 = time.time()
                    fps_disp = 10.0 / max(1e-6, (t1 - t0))
                    t0 = t1
                try:
                    fps_disp
                except NameError:
                    fps_disp = fps
                cv2.putText(frame, f'FPS: {fps_disp:.1f}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                cv2.imshow(window, frame)

                if writer:
                    writer.write(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print('Interrupted by user')

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


