import glob
import os.path
import pickle
from pathlib import Path

from bs4 import BeautifulSoup
import requests
import time

dict = {}
for filename in glob.iglob('../datasets/signClassificationData/train/*'):
    url = 'https://commons.wikimedia.org/wiki/File:PL_road_sign_{}.svg'.format(os.path.basename(filename))
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    file = soup.find(id='file')
    if(not file):
        continue
    url = file.find('a').get('href')
    dict[os.path.basename(filename)] = url

file = Path(f'datasets/signs.pkl')
with file.open("wb") as f:
    pickle.dump(dict, f)