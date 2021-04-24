import pandas as pd
import numpy as np

def dist(arr1,arr2):
    cos_sim=np.sum(arr1*arr2)/(np.sqrt(np.sum(arr1*arr1))*np.sqrt(np.sum(arr2*arr2)))  # cosine similarity using dot product of unit vector
    return np.exp(-cos_sim) # distance as e-similarity

def kmeans_clustering(arr,k,filename):
    N=len(arr)
    D=arr.shape[1] # numFeatures of each datapoint
    np.random.seed(18)
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
            print(','.join(map(str,indices)),file=f) # write indices of this cluster as comma list in given file


if __name__ == "__main__":
    df=pd.read_csv('../data/tf-idf.csv', header = 0) # reading dataset from part A
    arr=df.iloc[:,1:].values # obtaining array removing labels
    kmeans_clustering(arr,8,"../clusters/kmeans.txt")  # writes clusters to the provided files