import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import os

class Node: 
    def __init__(self,elem):  # used for storing and merging clusters
        self.left = None
        self.right = None
        self.parent = None 
        self.elem = elem # list of data points in clusterS

def dist(arr1,arr2):
    cos_sim=np.sum(arr1*arr2)/(np.sqrt(np.sum(arr1*arr1))*np.sqrt(np.sum(arr2*arr2))) # cosine similarity using dot product of unit vector
    return np.exp(-cos_sim) # distance as e-similarity

def agglomerative_clustering(arr,reqdClusters,filename):
    np.seterr(invalid='ignore')
    N=len(arr)
    clusters=[Node({i}) for i in range(N)] # clusters containing one data point each
    simMat=arr.dot(arr.transpose()) # finding cosine similarity matrix by taking dot product of each pair of rows
    distMat=np.exp(-simMat) # finding distance matrix
    np.fill_diagonal(distMat, np.nan) # edges between same points removed
    numClusters=N
    while numClusters > reqdClusters: # till clusters reduced to reqd
        pair=np.unravel_index(np.nanargmin(distMat),distMat.shape) # pair of points in differebt clusters with minimum distance
        a=pair[0]
        b=pair[1]
        total_elem=clusters[a].elem.union(clusters[b].elem) # creating new cluster as union of child clusters
        root=Node(total_elem)
        root.left=clusters[a] #left child
        root.right=clusters[b] #right child
        clusters[a].parent=root
        clusters[b].parent=root
        clusters[a]=root # setting cluster of a to root
        clusters[b]=None # cluster only present once in clusters list
        distMat[a,:]=np.minimum(distMat[a,:],distMat[b,:]) # updating minimum distance of cluster of a, as it has merged with b
        distMat[:,a]=distMat[a,:]  # symmetric
        distMat[b,:]=np.nan # as cluster[b] is merged into cluster[a], it is removed
        distMat[:,b]=np.nan  # updating minimum distance of cluster of a, as it has merged with b
        numClusters-=1 # 2 clusters merged into one
    with open(filename, 'w') as f:
        for i in range(N):
            if clusters[i]!=None: # cluster found
                print(','.join(map(str, clusters[i].elem)),file=f) # writing all data points of cluster as comma separated list in file

def kmeans_clustering(arr,k,filename):
    N=len(arr)
    D=arr.shape[1] # numFeatures of each datapoint
    np.random.seed(2)
    centroids=np.random.rand(k,D) # randomly initialise k centroids, from 0 to 1
    distances=np.zeros((k)) # for calculating distance of point with each cluster
    clusters=np.zeros((N)) # centroid to which this point is closest
    change=True
    while change: # if assigned centroid not changed of each point, converged return
        change=False
        for i in range(N): # for each datapoint
            for j in range(k): # for each centroid
                distances[j]=dist(arr[i],centroids[j]) # distance of ith datapoint with jth centroid
            cluster_index=np.argmin(distances) # finding closest centroid
            if cluster_index!=clusters[i]:
                clusters[i]=cluster_index # update assigned centroid
                change=True # if assigned centroid changes
        for j in range(k):
            points=arr[clusters==j] # data points in cluster corresponding to centroid j
            centroids[j]=np.mean(points,axis=0) # setting new centroid to their mean
    with open(filename, 'w') as f:
        for j in range(k): # for each cluster
            indices=[i for i in range(N) if clusters[i]==j] # find indices in cluster
            print(','.join(map(str,indices)),file=f) # write indices of this cluster as comma list in given fiel

if __name__ == "__main__":
    df=pd.read_csv('../data/tf-idf.csv', header = 0)  # reading dataset from part A
    arr=df.iloc[:,1:].values # obtaining array removing labels
    pca = PCA(n_components=100,random_state=20) # pca with 100 components
    arr=pca.fit_transform(arr) # appling pca on the tf-idf matrix
    arr=arr/np.sqrt(np.sum(arr*arr,axis=1,keepdims=True)) # normalising the obtained dataset after pca
    agglomerative_clustering(arr,8,"../clusters/agglomerative_reduced.txt") # agglomerative_clustering on reduced dataset
    kmeans_clustering(arr,8,"../clusters/kmeans_reduced.txt") # kmeans_clustering on reduced dataset