from importlib.util import module_for_loader
import numpy as np
from sklearn.cluster import DBSCAN
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#%matplotlib inline
from copy import deepcopy


def preprocess_img(fimg_name):
    img = Image.open(fimg_name)
    img_data = np.asarray(img)
    data = []
    for x in range(img.width):
        for y in range(img.height):
            z = int(3 * img_data[x][y][2] + 2 * img_data[x][y][1] + img_data[x][y][0])
            data.append((x, y, z))
    return data
    
def cluster(data):
    model = DBSCAN(eps=2.5, min_samples=2, algorithm='ball_tree', n_jobs=4)
    model.fit_predict(data)
    print(f'# clusters: {len(set(model.labels_))}')
    print(f'labels {model.labels_}')
    return dict(filter(lambda e: e[1] >= 0, zip(data, model.labels_)))

def draw_cluster(clusters):
    coords = np.array([list(t) for t in clusters.keys()])
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(coords.T[0], coords.T[1], coords.T[2], c=list(float(f) for f in clusters.values()), s=30)
    ax.view_init(azim=200)
    plt.show()