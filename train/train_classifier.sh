docker pull ultralytics/ultralytics:8.3.162-python

# Dish classification training
docker run -it --rm \
  --volume $(pwd)/data/Classification_split/dish:/train_data \
  ultralytics/ultralytics:8.3.162-python \
  sh -c "yolo classify train data=/train_data model=yolo11s-cls.pt epochs=100 batch=256 patience=10 save=True amp=False lr0=1e-3 lrf=1e-3 && cp /ultralytics/runs/classify/train/weights/best.pt /train_data/cls_dish.pt"

mv -f data/Classification_split/dish/cls_dish.pt ../models/cls_dish.pt

# Tray classification training
docker run -it --rm \
  --volume $(pwd)/data/Classification_split/tray:/train_data \
  ultralytics/ultralytics:8.3.162-python \
  sh -c "yolo classify train data=/train_data model=yolo11s-cls.pt epochs=100 batch=256 patience=10 save=True amp=False lr0=1e-3 lrf=1e-3 && cp /ultralytics/runs/classify/train/weights/best.pt /train_data/cls_tray.pt"

mv -f data/Classification_split/tray/cls_tray.pt ../models/cls_tray.pt
