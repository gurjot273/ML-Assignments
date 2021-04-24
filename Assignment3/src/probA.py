import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer

if __name__ == "__main__":
    df=pd.read_csv('../data/AllBooks_baseline_DTM_Labelled.csv', header = 0) # reading DTM
    df.iloc[:,0]=df.iloc[:,0].apply(lambda name: name.split('_')[0] ) # removing chap numbers from label
    df=df.drop([13],axis=0).reset_index(drop=True) # removing 14th row and adjust indices
    term_freq=df.iloc[:,1:].values # np array
    n=len(term_freq)
    doc_freq=np.sum(term_freq>0,axis=0) # getting document frequency
    idf=np.log((1+n)/(1+doc_freq)) # idf
    idf=idf.reshape(1,-1)
    idf=np.repeat(idf,n,axis=0)
    tfidf_mat=term_freq*idf # mutiplying tf with idf to get corresponding tf-idf
    tfidf_mat=tfidf_mat/np.sqrt(np.sum(tfidf_mat*tfidf_mat,axis=1,keepdims=True)) # l2 normalising all vectorss
    arr=np.append(df.iloc[:,0].values.reshape(-1,1),tfidf_mat,axis=1) # adding back the label column
    tf_idf = pd.DataFrame(arr, columns = df.columns) # coverting to dataframe with previous column names
    tf_idf.to_csv('../data/tf-idf.csv',index=False) # saving to tf-idf.csvs