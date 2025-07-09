# Dispatch Monitoring System

A computer vision system for monitoring dish and tray dispatch using YOLO-based object detection and classification models. The system processes video files to detect, track, and classify dishes and trays with states like empty, kakigori, or not_empty.

My current setup is CPU-only, so the system is designed to run without GPU support. The training was conducted using the Kaggle GPU environment. I’ve also packaged the training scripts using Docker, allowing you to run them on your own machine. If you have a compatible GPU, you can modify the Dockerfile and training scripts to utilize GPU acceleration for faster processing.

## System Architecture

The system consists of three main components:
- **Backend API**: FastAPI-based service for video processing and ML inference
- **Frontend**: Streamlit web interface for user interaction
- **Models**: Pre-trained YOLO models for detection and classification

## Project Structure

```
Dispatch-Monitoring-System/
├── app/
│   ├── backend/              # FastAPI backend service
│   ├── frontend/             # Streamlit frontend
│   └── shared_volume/        # Shared data directory
├── models/                   # Pre-trained model files
├── train/                    # Training scripts and data
├── docker-compose.yaml       # Docker composition
├── Dockerfile.backend        # Backend container
├── Dockerfile.frontend       # Frontend container
└── README.md
```

## Features

- Offline video processing, can be minimally modified for real-time processing
- Object detection for dishes and trays
- Classification of dish/tray states (empty, kakigori, not_empty)
- Web-based user interface for video selection and processing
- Shared volume for input/output video files
- Docker containerization for easy deployment

## Prediction Pipeline

For each frame in the video:
1. **Object Detection**: Use YOLO model to detect dishes and trays.
2. **State Classification**: Classify detected objects into states (empty, kakigori, not_empty).
3. **Tracking**: Maintain object identities across frames.

## Prerequisites

- Docker and Docker Compose
- GPU support recommended for faster processing, requires changing Dockerfile to use a GPU-enabled base image

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/phuc-gn/Dispatch-Monitoring-System.git
cd Dispatch-Monitoring-System
```

### 2. Verify Required Files

The models are included in the `models/` directory. If you need to train your own models, follow the training instructions below.

Ensure the following model files are present:
- `dect.pt` - Object detection model
- `cls_dish.pt` - Dish classification model  
- `cls_tray.pt` - Tray classification model

### 3. Prepare Input Videos

Place your `.mp4` video files in the `app/shared_volume/` directory for processing.

## Usage

### Quick Start with Docker Compose

1. **Start the system**:
   
   ```bash
   docker compose up
   ```

2. **Access the web interface**:

   Open your browser and navigate to: http://localhost:8501

3. **Process videos**:
   
   - Select a video file from the dropdown menu
   - Set the processing duration (in minutes). Negative value means processing the entire video.
   - Click "Start processing"
   - View results in the `app/shared_volume/` directory

4. **Stop the system**:
   
   ```bash
   docker compose down
   ```

## API Documentation

The backend API provides the following endpoints:

### POST /process_video

Process a video file for dish/tray detection and classification.

**Request Body:**

```json
{
    "filename": "test_video.mp4",
    "minutes": 1
}
```

**Response:**

```json
{
    "output": "test_video_processed.mp4",
}
```

### GET /health

Health check endpoint for service monitoring.

## Model Training

To train your own models, follow these steps:

### 1. Prepare Training Data

Place your datasets in the data/ directory following this structure:

```
data/
├── Classification/
│   ├── dish/
│   └── tray/
└── Detection/
    ├── train/
    ├── val/
    └── dataset.yaml
```

And run the following commands to navigate to the training directory:

```bash
cd train/
```

### 2. Data Preprocessing

Run the script below to split the dataset into training and testing sets.
The script will create a new directory `Classification_split` with the splited data.

```bash
chmod +x data_preprocessing.sh
./data_preprocessing.sh
```

### 3. Train Models

```bash
# Train detection model
chmod +x train_detector.sh
./train_detector.sh

# Train classification models
chmod +x train_classifier.sh
./train_classifier.sh
```

## Output Files

Processed videos and logs are saved in the `app/shared_volume/` directory:
- Processed videos: `*_processed.mp4`
- Detection logs: `logs/images/` and `logs/labels/`

Logs contain images with bounding boxes and labels. It can be used for further analysis and model improvement. We only log the images that contain newly detected objects, which helps in reducing the size of the logs and focusing on relevant data.

## Configuration

### Docker Configuration

The system uses the following ports:
- Frontend (Streamlit): 8501
- Backend (FastAPI): 8000 (internal)

### Model Configuration

Classification classes supported:
- `empty`: Empty dish/tray
- `kakigori`: Dish/tray with kakigori
- `not_empty`: Non-empty dish/tray

## Troubleshooting

### Performance Optimization

- Use a higher parameters variant of YOLO models for better accuracy
- Adjust the `confidence` thresholds in the backend API for optimal detection performance
- Use GPU-enabled Docker images for faster processing