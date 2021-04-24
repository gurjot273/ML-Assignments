import pandas as pd
import numpy as np

def NMI(class_labels,filename): # reads the class_labels, clusters from  file filename and calculates and returns the NMI
    N=len(class_labels)
    clusters=[]
    with open(filename,'r') as f: # reads clusters from file and stores in list of lists
        lines=f.read().splitlines()
        for l in lines:
            lst=l.split(',') # solitting comma list
            lst=[int(x) for x in lst]
            clusters.append(np.array(lst)) # list of np array corresponding to cluster indices
    h_cluster=0 # cluster_entropy
    clusters_counts=np.zeros((len(clusters)))
    for i in range(len(clusters)): # calculating cluster_entropy
        clusters_counts[i]=len(clusters[i])
        prob=clusters_counts[i]/N # finding probability
        h_cluster-=(prob*np.log2(prob)) # adding to cluster entropy
    h_class=0 # class entropy
    values,counts=np.unique(class_labels,return_counts=True) # unique labels with their counts
    class_counts=dict() # dictionary of labels with their counts
    for i in range(len(counts)):
        class_counts[values[i]]=counts[i] # adding to dict
        prob=counts[i]/N # finding probability
        h_class-=(prob*np.log2(prob)) # adding to class entropy
    mutual_information=0
    for i in range(len(clusters)): # for each cluster
        classes=class_labels[clusters[i]] # list of classes corresponding to cluster indices
        values,counts=np.unique(classes,return_counts=True) # different classes within the cluster and their count
        for j in range(len(values)):
            mutual_information+=((counts[j]/N)*np.log2((N*counts[j])/(clusters_counts[i]*class_counts[values[j]]))) # calculate mutual info using formula, for each class within the cluster
    nmi=(2*mutual_information)/(h_cluster+h_class) # find normalised mutual info
    return nmi


if __name__ == "__main__":
    df=pd.read_csv('../data/tf-idf.csv', header = 0) # reading tf-idf.csv from part A
    class_labels=df.iloc[:,0].values # getting the array of class labelss
    print("NMI for cluster corresponding to agglomerative.txt:",NMI(class_labels,"../clusters/agglomerative.txt")) # NMI for agglomerative cluster
    print("NMI for cluster corresponding to kmeans.txt:",NMI(class_labels,"../clusters/kmeans.txt")) # NMI for kmeans cluster
    print("NMI for cluster corresponding to agglomerative_reduced.txt:",NMI(class_labels,"../clusters/agglomerative_reduced.txt")) # NMI for agglomerative reduced cluster
    print("NMI for cluster corresponding to kmeans_reduced.txt:",NMI(class_labels,"../clusters/kmeans_reduced.txt")) # NMI for kmeans reduced cluster