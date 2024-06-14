import cv2
from ultralytics import YOLO

file = '../datasets/RumiaGraph250Main/gsv_10.jpg'

model = YOLO('../models/detect_models/best.pt')
#model = YOLO('models/detect_models/yolov8n.pt')
#model = YOLO('yolov8n-seg.pt')

print(model.names)
results = model.predict(source=file)
results = results[0].cpu()
results.names[0] = 'Sign'
res_plotted = results.plot()
cv2.imshow("result", res_plotted)
cv2.waitKey(0)

