import json
import numpy as np

with open('raw-archives/1861_8.json', 'rb') as data, \
                open('indices4', 'rb') as indices:
    data = json.load(data)
    indices = np.array([int(line) for line in indices])
    popular = np.array(data['response']['docs'])[indices]
    for pop in popular:
        print pop['headline']['main']
