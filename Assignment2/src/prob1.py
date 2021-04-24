import pandas as pd
import numpy as np

if __name__ == "__main__":
    # Part A
    df=pd.read_csv('../data/winequality-red.csv', sep = ';', header = 0)
    df.loc[df['quality']<=6,'quality']=0
    df.loc[df['quality']>6,'quality']=1
    for col in df.columns:
        if col!='quality': #columns other than quality
            arr=df[col].values[:] 
            minVal=np.min(arr) # min value of attribute
            maxVal=np.max(arr)  # max value of attribute
            df.loc[:,col]=(arr-minVal)/(maxVal-minVal) #min-max scaling
    df.to_csv('../data/datasetA.csv',index=False,sep=';') #saving dataset A

    # Part B
    df=pd.read_csv('../data/winequality-red.csv', sep = ';', header = 0)
    df.loc[df['quality']<5,'quality']=0 # converting quality attribute
    df.loc[df['quality']==5,'quality']=1
    df.loc[df['quality']==6,'quality']=1
    df.loc[df['quality']>6,'quality']=2
    for col in df.columns:
        if col!='quality':  #columns other than quality
            arr=df[col].values[:]
            mean=np.mean(arr) # mean value of attribute
            std=np.std(arr) # std value of attribute
            arr=(arr-mean)/(std*1.0) # Z score normalization
            minVal=np.min(arr)
            maxVal=np.max(arr)  
            step=(maxVal-minVal)/4.0
            # deciding equal spaced bucket masks
            mask1=(arr<(minVal+step))
            mask2=(arr>=(minVal+step))&(arr<(minVal+2*step))
            mask3=(arr>=(minVal+2*step))&(arr<(minVal+3*step))
            mask4=(arr>=(minVal+3*step))
            # replacing with values in [0 to 3]
            arr[mask1]=0
            arr[mask2]=1
            arr[mask3]=2
            arr[mask4]=3
            df.loc[:,col]=arr # assigning new val 
    df.to_csv('../data/datasetB.csv',index=False,sep=';')  # saving dataset B