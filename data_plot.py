import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

def tsne_plot2d(x, y):
    colors=['blue', 'red']
    tsne = TSNE(n_components=2).fit_transform(x)
    plt.figure(figsize=(15, 15))
    plt.scatter(*zip(*tsne[:,:2]), c=y, cmap=ListedColormap(colors))
    plt.title("t-SNE plot 2D")
    plt.show()
    
def tsne_plot3d(x, y):
    colors=['blue', 'red']
    tsne = TSNE(n_components=3).fit_transform(x)
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(*zip(*tsne), c=y, cmap=ListedColormap(colors))
    plt.title("t-SNE plot 3D")
    plt.show()    
    
def pca_plot3d(x, y):
    colors=['blue', 'red']
    pca = PCA(n_components=3).fit_transform(x)
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*zip(*pca), c=y, cmap=ListedColormap(colors))
    plt.title("PCA plot 3D")
    plt.show()    
    
def histo_drawer(x, title, i):
    plt.figure(figsize=(15, 10))
    n, bins, patches = plt.hist(x, len(x), facecolor='blue', alpha=0.5)
    plt.title('Histogram of ' + title + ' column №' + str(i))
    plt.show()
       
    
    
def pca_explained_variance_ratio(data):
    pca = PCA(n_components=3).fit(data)
    print(np.sum(pca.explained_variance_ratio_))
    print('Common explained variance ratio: ' +  str(round(np.sum(pca.explained_variance_ratio_) * 100, 2)) + '%')
    pca = pca.transform(data)
    print('=> Number of pc is 3')
    return pca    