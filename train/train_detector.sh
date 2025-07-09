docker pull ultralytics/ultralytics:8.3.162-python

docker run -it --rm \
  --volume $(pwd)/data/Detection:/train_data \
  ultralytics/ultralytics:8.3.162-python \
  sh -c "cd /train_data && yolo detect train data=dataset_edit.yaml model=yolo11s.pt epochs=100 batch=64 patience=10 save=True amp=False && cp /ultralytics/runs/detect/train/weights/best.pt /train_data/dect.pt"

rm -f data/Detection/yolo11s.pt
mv -f data/Detection/dect.pt ../models/dect.pt