import base64
import math
import os
from io import BytesIO
from pathlib import Path

import folium
import pickle

from PIL import Image, ImageDraw, ImageFont
from folium import IFrame
from folium.plugins import MarkerCluster

from data import classes

#GRAPHS = ['RumiaGraph250Main']
GRAPHS = ['Rumia2500', 'GdyniaSkwer1000', 'GdanskCentrum1000', 'GdanskPG1000']
markers_file = [Path(f'../data/markers/{GRAPH}') for GRAPH in GRAPHS]
markers = []
for i in markers_file:
    with i.open("rb") as f:
        markers.append(pickle.load(f))

signs_file = Path('../data/signs.pkl')
signs = None
with signs_file.open("rb") as f:
    signs = pickle.load(f)
signs['D-tablica'] = 'https://upload.wikimedia.org/wikipedia/commons/6/6c/PL_road_sign_D-34b.svg'
signs['B-33'] = 'https://upload.wikimedia.org/wikipedia/commons/5/52/PL_road_sign_B-33-50.svg'
signs['C-13-C-16'] = 'https://upload.wikimedia.org/wikipedia/commons/e/ee/PL_road_sign_C-13-16.svg'
signs['B-18'] = 'https://upload.wikimedia.org/wikipedia/commons/1/16/PL_road_sign_B-18-7t.svg'
signs['B-34'] = 'https://upload.wikimedia.org/wikipedia/commons/e/ea/PL_road_sign_B-34-50.svg'
signs['B-44'] = 'https://upload.wikimedia.org/wikipedia/commons/2/28/PL_road_sign_B-44-30.svg'
signs['B-43'] = 'https://upload.wikimedia.org/wikipedia/commons/c/c6/PL_road_sign_B-43-30.svg'
signs['B-6-B-8-B-9'] = 'https://upload.wikimedia.org/wikipedia/commons/d/dc/PL_road_sign_B-6-8-9.svg'


m = folium.Map(location=[54.460178, 18.501940], zoom_start=11)
cluster = MarkerCluster().add_to(m)

for id2, marker in enumerate(markers):
    GRAPH = GRAPHS[id2]
    for i in marker:
        x = i[0]
        y = i[1]
        b = i[2]
        file = i[3]
        c = i[4]
        box = i[5]
        for id, label in enumerate(c):
            if label == 92:
                continue
            curr_box = box[id]
            # jesli jest bardziej prostokatne niz x na 2.5x to wyrzucam
            if (curr_box[0]-curr_box[2])/(curr_box[1]-curr_box[3]) < 0.4 or \
                (curr_box[0] - curr_box[2]) / (curr_box[1] - curr_box[3]) > 2.5 or \
                    abs((curr_box[0]-curr_box[2]) * (curr_box[1]-curr_box[3])) > 640*640*0.05:
                continue
            c_str = classes[c[id]]
            if not os.path.exists(f'../datasets/{GRAPH}Transformed/{id}_{os.path.basename(file)}'):
                img = Image.open(file)
                img_draw = ImageDraw.Draw(img)
                img_draw.fill = False
                img_draw.rectangle(curr_box, outline = 'red', width = 5)
                img_draw.text((curr_box[0]+1, curr_box[1]-20), c_str, fill='white',
                              font = ImageFont.truetype("arial.ttf", 20), stroke_width=2, stroke_fill='black')
                img.save(f'../datasets/{GRAPH}Transformed/{id}_{os.path.basename(file)}')
            img_html = f'<img src="http://lidbey.pl/imgs/{GRAPH}Transformed/{id}_{os.path.basename(file)}">'
            html = f'{x} {y} {file} {img_html}'
            iframe = IFrame(html, width=670, height=680)
            popup = folium.Popup(iframe, max_width=670, lazy=True)
            if math.isnan(y) or math.isnan(x):
                print(f'File {file} is wrong! location is {y} {x}')
                print(y, x)
                print(i[0], i[1])
                continue
            marker_icon = folium.Marker(location=[y, x], tooltip=img_html, popup=popup,
                                        icon=folium.CustomIcon(signs[c_str], icon_size = (32, 32)))
            marker_icon.add_to(cluster)

#m.show_in_browser()
map_title = 'Mapowanie znaków drogowych<br/>Wojciech Wicki'
title_html = f'<h1 align="center" style="position:absolute;z-index:100000;left:35vw" >{map_title}</h1>'
m.get_root().html.add_child(folium.Element(title_html))
m.save('map.html')