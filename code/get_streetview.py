import roads
import osmnx as os
import google_streetview.api as gsapi

from key import KEY


# SELECT PLACE - CHOOSE YOUR OWN
GRAPH_NAME = 'GdanskCentrum1000'
POINT = (54.350661, 18.656753)
RADIUS = 1000
REDUCE_METERS = 3

# PARAMETERS
SIZE = '640x640'
PITCH = '0'

G = roads.get_graph(GRAPH_NAME, POINT, RADIUS)
os.plot_graph(G)
df = roads.get_graph_data(G, REDUCE_METERS)
print(len(df))


param = []
for i in df:
	param.append({
		'size': SIZE,
		'location': str(i[1])+','+str(i[0]),
		'heading': str(i[2]),
		'pitch': PITCH,
		'key': KEY,
		'source': 'outdoor'
	})
results = gsapi.results(param)
results.download_links('../datasets/'+GRAPH_NAME, 'meta.json')