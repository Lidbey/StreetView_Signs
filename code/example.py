import cv2
from ultralytics import YOLO
import numpy as np
import classification
from data import classes
#https://www.kaggle.com/datasets/kasia12345/polish-traffic-signs-dataset
#https://www.kaggle.com/datasets/chriskjm/polish-traffic-signs-dataset
file = '../datasets/RumiaGraph350Main/gsv_41.jpg'

model_c = classification.get_model('../models/class_models/min_loss')
model_d = YOLO('../models/detect_models/best.pt')
model_ds = YOLO('../models/detect_models/yolov8n.pt')

img = classification.get_pre_image(file)

results = model_d.predict(source=file)
results_s = model_ds.predict(source=file, classes=[9])
boxes = results[0].boxes.xyxy.numpy()

cut_imgs = classification.cut_by_bbox(img, boxes)

r = model_c.predict(cut_imgs)
Y = np.argmax(r, axis=1)

# some magic to show :)
v = r[np.arange(len(Y)), Y]
res = results[0].numpy()
for id, i in enumerate(classes):
    res.names[id] = i
for id, i in enumerate(Y):
    res.boxes.cls[id] = i
    res.boxes.conf[id] *= v[id]
print(Y)
print(classes[Y])
res_plotted = res.plot()
cv2.imshow("result", res_plotted)
cv2.waitKey(0)

