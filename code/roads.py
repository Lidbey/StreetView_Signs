import osmnx as os
from pathlib import Path
import pickle
import geopy.distance
import math
import json


def get_graph(name, location=(54.571005, 18.387882), distance=350):
    file = Path(f'../data/graphs/{name}')
    if file.exists():
        with file.open("rb") as f:
            print(f'Found file with name {name}')
            return pickle.load(f)
    else:
        with file.open("wb") as f:
            print(f'Didnt find file with name {name}, downloading')
            graph = os.graph_from_point(location, dist=distance)
            pickle.dump(graph, f)
            return graph


def append_to_dict(d, x, y, b):
    d['x'].append(x)
    d['y'].append(y)
    d['b'].append(b)


def is_closer_than(x1, y1, x2, y2, dist):
    if geopy.distance.great_circle((y1, x1), (y2, x2)).meters < dist:
        return True
    else:
        return False


def is_any_closer_than(x1, y1, df, dist):
    for item in df:
        x2 = item[0]
        y2 = item[1]
        if is_closer_than(x1, y1, x2, y2, dist):
            return True
    return False


def get_graph_data(graph, max_dist):
    graph = os.add_edge_bearings(graph)
    edges = os.graph_to_gdfs(graph, nodes=False)
    d = []
    for row in edges.itertuples():
        x = row.geometry.centroid.x
        y = row.geometry.centroid.y
        if is_any_closer_than(x, y, d, max_dist):
            continue
        d.append((x, y, row.bearing))
        d.append((x, y, row.bearing + 180 if row.bearing < 180 else row.bearing - 180))
    return d


def get_image_data(gdf, graph):
    f = open(f'../datasets/{graph}/meta.json')
    metadata = json.load(f)
    d = []
    for i in range(len(metadata)):
        if 'location' not in metadata[i]:
            continue
        line = (metadata[i]['location']['lng'], metadata[i]['location']['lat'], gdf[i][2], metadata[i]['_file'])
        if line not in d:
            d.append(line)
        else:
            print('found duplicate')
            print(line)
    return d


# perpendicular is always to the right side in PL
def get_sign_pos(start_lon, start_lat, heading, d_parallel = 2, d_perpendicular = 2, to_right = True):
    earth_radius = 6371000 # average
    alfa = math.pi*heading/180
    beta = math.pi*(heading + 90 if to_right else heading - 90)/180
    dx = d_parallel*math.sin(alfa)+d_perpendicular*math.sin(beta) # distance in X - heading is from top - top = 0 degrees
    dy = d_parallel*math.cos(alfa)+d_perpendicular*math.cos(beta) # distance in Y
    lon_circumference = 2*math.pi*earth_radius # obwod poludnika
    d_lat = 360*dy/lon_circumference # odleglosc w osi Y (gora/dol - rownolezniki)
    lat_circumference = 2*math.pi*earth_radius*math.cos(math.pi/180*(start_lat+d_lat)) # obwod rownoleznika
    d_lon = 360*dx/lat_circumference # odleglosc w osi X (prawo/lewo - poludniki)
    return start_lon + d_lon, start_lat + d_lat

