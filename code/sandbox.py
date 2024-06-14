import pickle

from ultralytics import YOLO

import roads
import osmnx as ox
import folium
import networkx as nx
from folium.features import DivIcon
import cv2
import matplotlib.pyplot as plt
import numpy as np
import csv
import data


def plotRoads():
    d = [['RumiaGraph250Main', 54.571005, 18.387882, 250]]#,
    #['Rumia2500', 54.571005, 18.387882, 2500],
    #['GdanskPG1000', 54.371891, 18.615316, 1000],
    #['GdyniaSkwer1000', 54.519366, 18.542529, 1000],
    #['GdanskCentrum1000', 54.350661, 18.656753, 1000]]

    graphs = []
    for i in d:
        graphs.append(roads.get_graph(i[0], (i[1], i[2]), i[3]))
    G = nx.compose_all(graphs)
    m = ox.folium.plot_graph_folium(G,tiles='openstreetmap',popup_attribute='name',opacity = 1,color = 'red',weight= 1)

    for i in d:
        radius = i[3]
        lat = i[1]
        lon = i[2]
        col = 'green'
        txt = i[0]
        if txt == 'RumiaGraph250Main':
            col = 'cornflowerblue'
        folium.Circle(
            location = [lat, lon],
            radius=radius,
            color='black',
            weight=1,
            fill_opacity=0.3,
            opacity=1,
            fill_color=col
        ).add_to(m)
        folium.map.Marker(
            [lat if i[0] != 'RumiaGraph250Main' else 54.573, lon],
            icon = DivIcon(
            icon_size=(150,36),
            icon_anchor=(0,0),
            html='<div style="font-size: 24pt">%s</div>' % txt,
            )
        ).add_to(m)


    m.show_in_browser()

def plotAt(x, y, img, axes, title):
    axes[x, y].imshow(img)
    axes[x, y].axis('off')
    axes[x, y].set_title(title)

def printImage():
    imgPath='../datasets/signClassificationData/test/A-1/2019_0726_185626_003 284_0.jpg'
    f, axes = plt.subplots(2,3)

    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plotAt(0, 0, img, axes, 'Oryginalne zdjęcie')

    center = (img.shape[1] / 2, img.shape[0] / 2)
    M = cv2.getRotationMatrix2D(center, -10, 1)
    img_r = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    plotAt(0, 1, img_r, axes, 'Obrót o 10 stopni')

    M = np.float32([[1, 0, 20], [0, 1, 0]])
    img_sx = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    plotAt(0, 2, img_sx, axes, 'Przesunięcie o 20p w poziomie')

    M = np.float32([[1, 0, 0], [0, 1, 15]])
    img_sy = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    plotAt(1, 0, img_sy, axes, 'Przesuniecie o 15p w pionie')

    M = np.float32([[1, -0.2, 0], [0, 1, 0], [0, 0, 1]])
    img_s = cv2.warpPerspective(img, M,
                                     (int(img.shape[1] ),
                                      int(img.shape[0] )))
    plotAt(1, 1, img_s, axes, 'Ścięcie o 20%')

    cy, cx = [i / 2 for i in img.shape[:-1]]
    rot_mat = cv2.getRotationMatrix2D((cx, cy), 0, 1.2)
    img_z = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    plotAt(1, 2, img_z, axes, 'Przybliżenie o 20%')

    plt.show(block=True)

#printImage()

def plot_decay():
    ILR = 0.001
    DR = 0.9
    DS = 66
    x_l = list(range(0,5000))
    y = list(map(lambda x: ILR*pow(DR, x/DS), x_l))
    plt.plot(x_l, y)
    #plt.gca().set_yscale('log')
    plt.gca().set_xlabel('Numer kroku')
    plt.gca().set_ylabel('Wspolczynnik uczenia sie')
    plt.show()
#plot_decay()

#plotRoads()

def testModel():
    model = YOLO('../runs/detect/train2/weights/best.pt')
    model.val(conf=0.7)

#testModel()

def testSystem():
    l = []
    with open('../datasets/Custom_dataset.csv') as f:
        reader = csv.reader(f, delimiter=';')
        map = {
            'C-8': 'C-9',
        }
        remove = ['F-10', 'T-27', 'B-12','C-11', 'B-23', 'D-36', 'F-21', #not in classification
                  'D-23', 'C-16', 'C-13', 'B-5', 'B-18', 'A-29'] #not in detection
        for id, row in enumerate(reader):
            if id <= 100 or id >=501:
                continue
            l2 = []
            for id2, item in enumerate(row):
                if id2 != 0 and item != '':
                    #if item in data.classes:
                    #    l2.append(data.classes.tolist().index(item))
                    if item in data.detection_trained_on:
                        l2.append(data.classes.tolist().index(item))
                    elif item in map:
                        l2.append(data.classes.tolist().index(map[item]))
                    elif item in remove:
                        continue
                    else:
                        print(f'item {item} not in analyzed classes!')
            l.append([id-101, l2, []])
        print(l)
    sum_photos_with_sign = 0
    with open('../data/markers/RumiaGraph250Main', 'rb') as f:
        markers = pickle.load(f)
        for i in markers:
            file_id = int(i[3].split('_')[1][:-4])
            lc = []
            for item in i[4]:
                if data.classes[item] in data.detection_trained_on:
                    lc.append(item)
            if file_id <= 400:
                l[file_id][2] = lc
                if len(lc) > 0:
                    sum_photos_with_sign = sum_photos_with_sign + 1
            #r.append([file_id, l[file_id], i[4]])
    print(l)

    sum_real = 0
    sum_pred = 0
    sum_correct = 0
    for i in l:
        l1 = i[1]
        l2 = i[2]
        sum_real = sum_real + len(set(l1))
        sum_pred = sum_pred + len(set(l2))
        for item in l2:
            if item in l1:
                sum_correct = sum_correct + 1
        if len(set(l1)) > len(set(l2)):
            print(l1, l2)
    print(sum_real, sum_pred, sum_correct, sum_photos_with_sign)
testSystem()