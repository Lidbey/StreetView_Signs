from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#model = YOLO('runs/detect/train2/weights/last.pt')
model = YOLO('../models/detect_models/yolov8n.pt')
results = model.train(data='../datasets/datasetChris.yaml', epochs=200, batch=32, fraction=0.9, resume=False)
