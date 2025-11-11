"""run_camera.py
Real-time YOLOv8 + MediaPipe Pose demo using OpenCV windows.

Usage examples:
  python run_camera.py --source 0
  python run_camera.py --source 0 --output recorded.mp4
  python run_camera.py --source in_video.mp4 --output out.mp4

Press 'q' to quit.
"""
import argparse
import time
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp


def to_source(src):
    try:
        return int(src)
    except Exception:
        return src


def main():
    parser = argparse.ArgumentParser(description="Real-time YOLO + MediaPipe pose demo")
    parser.add_argument('--source', default='0', help='Camera index (0,1,...) or path to video file')
    parser.add_argument('--model', default='yolov8l.pt', help='YOLO model path (default: yolov8l.pt)')
    parser.add_argument('--output', default=None, help='Optional output video file to save processed frames')
    parser.add_argument('--display-width', type=int, default=960, help='Width to display window (keeps aspect ratio)')
    args = parser.parse_args()

    source = to_source(args.source)

    # Load models
    print(f'Loading YOLO model: {args.model} ...')
    model = YOLO(args.model)

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print('ERROR: Could not open video source:', source)
        return

    # Read basic properties and set defaults for cameras
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = None
    if args.output:
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        if not writer.isOpened():
            print('WARNING: Could not open video writer for', args.output)
            writer = None

    window_name = 'Fall Detection (press q to quit)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    t0 = time.time()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    # end of file or camera disconnected
                    break

                frame_count += 1

                # Run YOLO model on frame (returns iterable of results)
                results = model(frame)

                # Process detections
                for result in results:
                    # result.boxes.xyxy: list of boxes, result.boxes.cls: classes
                    for bbox, cls in zip(result.boxes.xyxy, result.boxes.cls):
                        if int(cls) != 0:
                            continue  # only person class (0)
                        x1, y1, x2, y2 = map(int, bbox)
                        x1 = max(0, x1); y1 = max(0, y1);
                        x2 = min(frame.shape[1]-1, x2); y2 = min(frame.shape[0]-1, y2)

                        # draw bbox
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                        # crop person region for pose (skip tiny boxes)
                        if x2 - x1 < 20 or y2 - y1 < 20:
                            continue
                        person = frame[y1:y2, x1:x2]
                        person_rgb = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
                        pres = pose.process(person_rgb)
                        if pres.pose_landmarks:
                            mp_drawing.draw_landmarks(person, pres.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                            # place annotated person back
                            frame[y1:y2, x1:x2] = person

                # Compute and draw FPS
                if frame_count % 10 == 0:
                    t1 = time.time()
                    fps_disp = 10.0 / max(1e-6, (t1 - t0))
                    t0 = t1
                # Show FPS on frame
                cv2.putText(frame, f'FPS: {fps_disp:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

                # Resize display window while preserving aspect ratio
                h, w = frame.shape[:2]
                scale = args.display_width / float(w)
                disp = cv2.resize(frame, (int(w*scale), int(h*scale)))

                cv2.imshow(window_name, disp)

                if writer:
                    # write original-sized frame
                    writer.write(frame)

                # Exit on 'q'
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
