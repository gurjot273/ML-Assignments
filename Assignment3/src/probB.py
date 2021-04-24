import pandas as pd
import numpy as np

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


if __name__ == "__main__":
    df=pd.read_csv('../data/tf-idf.csv', header = 0) # reading dataset from part A
    arr=df.iloc[:,1:].values # obtaining array removing labels
    agglomerative_clustering(arr,8,"../clusters/agglomerative.txt") # writes clusters to the provided file