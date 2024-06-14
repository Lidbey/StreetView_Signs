from pathlib import Path
from ultralytics import YOLO
import numpy as np
import classification
import roads
import pickle

#dodac tu jakies sprawdzanie po ktorej stronie zdjecia to jest moze!

GRAPH_NAME = 'GdanskCentrum1000'

model_c = classification.get_model('../models/class_models/max_acc2')
model_d = YOLO('../models/detect_models/best.pt')
G = roads.get_graph(GRAPH_NAME)
rdf = roads.get_graph_data(G, 3)
df = roads.get_image_data(rdf, GRAPH_NAME)

markers = []
for item in df:
    b = item[2]
    x, y = roads.get_sign_pos(item[0], item[1], b, 1, 1)
    file = f'../datasets/{GRAPH_NAME}/{item[3]}'
    if not Path(file).exists():
        continue
    results = model_d.predict(source=file, conf=0.7)
    boxes = results[0].boxes.xyxy.numpy()
    img = classification.get_pre_image(file)
    cut_imgs = classification.cut_by_bbox(img, boxes)
    classes = []
    if len(cut_imgs) != 0:
        r = model_c.predict(cut_imgs)
        Y = np.argmax(r, axis=1)
        for i in Y:
            classes.append(i)
    if len(classes) != 0:
        markers.append((x, y, b, file, classes, boxes))
    print(file, len(df))

file_name = GRAPH_NAME
file = Path(f'../data/markers/{file_name}')
with file.open("wb") as f:
    print(f'Saving markers to {file_name}')
    pickle.dump(markers, f)