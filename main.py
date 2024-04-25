import base64
from pathlib import Path

import folium
import pickle
from folium import IFrame

from data import classes

markers_file = Path('datasets/markers/2024-02-22-00-37')
markers = None
with markers_file.open("rb") as f:
    markers = pickle.load(f)


m = folium.Map(location=[54.571005, 18.387882], zoom_start=15)

for i in markers:
    x = i[0]
    y = i[1]
    b = i[2]
    file = i[3]
    c = i[4]
    if c == [92]:
        continue
    c_str = [classes[i] for i in c]
    encoded = base64.b64encode(open(file, 'rb').read())
    html = (str(x) + ' ' + str(y) + ' ' + file+' ' + ' '.join(c_str)+' '+'<figcaption>Name</figcaption><img src="data:image/png;base64,{}">').format
    iframe = IFrame(html(encoded.decode('UTF-8')), width= 670, height = 680)
    popup = folium.Popup(iframe, max_width=670)
    folium.Marker(location=[y, x], tooltip=html, popup=popup, icon = folium.Icon(angle=round(b), color='gray', icon='arrow-up')).add_to(m)

m.show_in_browser()