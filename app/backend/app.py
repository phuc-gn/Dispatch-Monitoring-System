from fastapi import FastAPI, status
from pydantic import BaseModel

import os

import cv2
from ultralytics import YOLO


app = FastAPI(title='Dispatch Monitoring API',
              description='API for processing video files to detect and classify dishes and trays.'
              )

# Shared directory for container communication
SHARED_DIR = '/shared_volume'
LOG_IMAGE_DIR = os.path.join(SHARED_DIR, 'logs', 'images')
LOG_LABEL_DIR = os.path.join(SHARED_DIR, 'logs', 'labels')
os.makedirs(LOG_IMAGE_DIR, exist_ok=True)
os.makedirs(LOG_LABEL_DIR, exist_ok=True)

# Load models once at startup
det_model = YOLO('dect.pt')
dish_model = YOLO('cls_dish.pt')
tray_model = YOLO('cls_tray.pt')
cls_classes = ['empty', 'kakigori', 'not_empty']

class VideoProcessingRequest(BaseModel):
    filename: str
    minutes: int

@app.post('/process_video')
def process_video(payload: VideoProcessingRequest):
    filename = payload.filename
    minutes = payload.minutes

    output_name = filename.split('.')[0] + '_processed.mp4'
    video_path = os.path.join(SHARED_DIR, filename)

    if not os.path.exists(video_path):
        return {'error': 'File not found.'}
 
    # Get video properties
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    confidence_threshold = 0.5

    # Calculate total frames to process
    if minutes < 0:
        total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        total_frame = fps * minutes * 60

    output_path = os.path.join(SHARED_DIR, output_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_id = 0
    seen_ids = set()

    # Detect and classify objects in the video
    while cap.isOpened() and frame_id < total_frame:
        ret, frame = cap.read()
        if not ret:
            break

        og_frame = frame.copy()
        frame_id += 1

        det_results = det_model.track(frame, persist=True, verbose=False)[0]

        new_objects = []
        label_lines = []

        for box in det_results.boxes:
            conf = box.conf.item()

            # Skip boxes with low confidence
            if conf < confidence_threshold:
                continue

            cls_id = int(box.cls.item())
            track_id = int(box.id.item()) if box.id is not None else -1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]

            # Classify the detected object
            if cls_id == 0:
                cls_result = dish_model.predict(crop, verbose=False)[0]
                cls_label = cls_classes[int(cls_result.probs.top1)]
                label_text = f'dish_{cls_label}'

            elif cls_id == 1:
                cls_result = tray_model.predict(crop, verbose=False)[0]
                cls_label = cls_classes[int(cls_result.probs.top1)]
                label_text = f'tray_{cls_label}'

            else:
                cls_label = 'unknown'
                label_text = f'unknown'

            # Prepare label for display
            label = f'[#{track_id}] {label_text} ({conf:.2f})'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Detect new object and log
            if track_id not in seen_ids:
                new_objects.append(track_id)
                seen_ids.add(track_id)
            
            # Prepare label line for saving
            label_lines.append(f'{track_id},{label_text},{conf:.2f},{x1},{y1},{x2},{y2}')

        # Save image and label if new object(s) detected
        if new_objects:
            img_path = os.path.join(LOG_IMAGE_DIR, f'{frame_id}.jpg')
            lbl_path = os.path.join(LOG_LABEL_DIR, f'{frame_id}.txt')
            cv2.imwrite(img_path, og_frame)
            
            with open(lbl_path, 'w') as f:
                f.write('\n'.join(label_lines))

        out.write(frame)

    cap.release()
    out.release()

    return {'output': output_name}

@app.get('/health')
def health():
    return status.HTTP_200_OK, {'status': 'ok'}