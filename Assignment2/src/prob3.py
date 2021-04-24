import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

#Part 1
class Node():
    isLeaf=True
    label=0 # if isLeaf=True, label of node
    splitAttribute=None # if isLeaf=False, attribute to test
    children=[] # list of children if isLeaf=False
    def __init__(self):
        pass
    def makeInternal(self,splitAttribute,children): # set children and attribute of internal node
        self.isLeaf=False
        self.children=children
        self.splitAttribute=splitAttribute
    def makeLeaf(self,label): # set label of leaf
        self.isLeaf=True
        self.label=label

#Part 1
def getEntropy(df,targetAttribute): # calculating entropy of df wrt targetAttribute
    if len(df)==0:
        return 0
    labels=df[targetAttribute].values[:]
    (values,counts)=np.unique(labels,return_counts=True) #counts is array of counts of distinct values of attribute
    counts=np.array(counts) 
    prob=counts/(np.sum(counts)) # calculating probability of each class
    return np.sum(-(prob*np.log2(prob))) # calculating entropy using prob

#Part 1
def getInformationGain(df,attribute,targetAttribute): # calculating information gain if split by attribute
    numSamples=len(df)
    if numSamples==0:
        return 0.0
    parentEntropy=getEntropy(df,targetAttribute) # entropy of parent
    weightedChildEntropy=0.0 
    for val in [0,1,2,3]: # all the attributes except 'quality' have range [0,1,2,3]
        childDf=df.loc[df[attribute]==val] # subset of examples with attribute==val
        numSamplesChild=len(childDf)
        weightedChildEntropy=weightedChildEntropy + (numSamplesChild/(numSamples*1.0))*getEntropy(childDf,targetAttribute) # adding weighted entropy of child
    return parentEntropy-weightedChildEntropy # return information gain for the split

#Part 1
def buildTree(df,targetAttribute,attributes,minSamples):
    root=Node() # root of tree to be returned
    (values,counts)=np.unique(df[targetAttribute],return_counts=True)
    pos=np.argmax(counts) # index of largest count
    majorityLabel=values[pos] # majorityLabel wrt targetAttribute
    numSamples=len(df)
    if (len(values)==1) or (len(attributes)==0) or (numSamples<minSamples): # if all examples have same label or attribute set is empty or number of examples < minSize
        root.makeLeaf(majorityLabel) # leaf labelled by majorityLabel or only label present
        return root
    informationGains=np.array([getInformationGain(df,attr,targetAttribute) for attr in attributes]) # getting informationGain for each attribute in attribute set
    index=np.argmax(informationGains) # selecting index of attribute with maximum information gain
    splitAttribute=attributes[index] # getting the best attribute to split on
    childAttributes=[attr for attr in attributes if attr!=splitAttribute] # removing splitAttribute from childAttribute set
    children=[] # list of children nodes
    for val in [0,1,2,3]:
        childDf=df.loc[df[splitAttribute]==val] # subset of examples with attribute==val
        numSamplesChild=len(childDf)
        if numSamplesChild==0: # if subset is empty
            child=Node()
            child.makeLeaf(majorityLabel) # leaf labelled by majority label
            children.append(child) # append child node to list
        else:
            child=buildTree(childDf,targetAttribute,childAttributes,minSamples) # recursive building subtree using subset of examples and attributes
            children.append(child) # append child node to list
    root.makeInternal(splitAttribute,children) # internal node split by splitAttribute and children as list of children
    return root # return root of tree

#Part 1
def predictLabel(root,example):
    if root.isLeaf: # if we have reached a leaf node
        return root.label # return label
    return predictLabel(root.children[int(example[root.splitAttribute])],example) # go to child of root having value of splitAttribute as that of example

#Part 1
def predict(root,df):
    labels=[]
    for i in range(len(df)):
        labels.append(predictLabel(root,df.iloc[i])) # find the label for each example using predictLabel and append to list
    return np.array(labels) # return y_pred having predicted labels for the examples in df

if __name__ == "__main__":
    df=pd.read_csv('../data/datasetB.csv', sep = ';', header = 0) # loading dataset B for decision tree
    attributes=df.columns.tolist()
    attributes=[attr for attr in attributes if attr!='quality'] # attributes other than targetAttribute

    #Part 2
    model=DecisionTreeClassifier(criterion='entropy',min_samples_split=10,random_state=1) # sklearn Decisin tree split by entropy and min_samples_split=10

    # Part 3: Cross validating self-written decision tree classifier 
    kf = KFold(n_splits=3) # for KFoldCrossValidation
    train_accuracy=[]
    train_precision=[]
    train_recall=[]
    test_accuracy=[]
    test_precision=[]
    test_recall=[]
    for train_index, test_index in kf.split(df):  # generating splits for 3 folds
        df_train=df.iloc[train_index] # train_Set
        df_test=df.iloc[test_index] # test_set
        root=buildTree(df_train,'quality',attributes,10) # learning self-written decision tree on df_train and target_attribute 
        
        y_train=df_train['quality'].values[:] # predicting on train_set for current fold
        y_pred=predict(root,df_train.iloc[:,:-1])
        # Macro Training scores for current fold
        train_accuracy.append(accuracy_score(y_train,y_pred))
        train_precision.append(precision_score(y_train,y_pred,average='macro'))
        train_recall.append(recall_score(y_train,y_pred,average='macro'))
        
        y_test=df_test['quality'].values[:]
        y_pred=predict(root,df_test.iloc[:,:-1]) # predicting on test_set for current fold
        # Macro Test scores for current fold
        test_accuracy.append(accuracy_score(y_test,y_pred))
        test_precision.append(precision_score(y_test,y_pred,average='macro'))
        test_recall.append(recall_score(y_test,y_pred,average='macro'))
    # Mean Macro Training scores of all 3 folds for self-written
    train_accuracy=np.mean(train_accuracy)
    train_precision=np.mean(train_precision)
    train_recall=np.mean(train_recall)
    # Mean Macro Test scores of all 3 folds for self-written
    test_accuracy=np.mean(test_accuracy)
    test_precision=np.mean(test_precision)
    test_recall=np.mean(test_recall)
    print('Scores for self-written Decision Tree Classifier:')
    print('Mean macro train accuracy:',train_accuracy)
    print('Mean macro train precision:',train_precision)
    print('Mean macro train recall:',train_recall)
    print('Mean macro test accuracy:',test_accuracy)
    print('Mean macro test precision:',test_precision)
    print('Mean macro test recall:',test_recall)

    # Part 3:  Cross validating sklearn decision tree classifier
    model=DecisionTreeClassifier(criterion='entropy',min_samples_split=10,random_state=1)
    x_data=df.iloc[:,:-1].values[:]
    y_data=df['quality'].values[:]
    scoring=['accuracy','precision_macro','recall_macro']
    # cross validating using three folds and returning score object
    scores=cross_validate(model,x_data,y_data,cv=kf,scoring=scoring,return_train_score = True)
    print('Scores for sklearn Decision Tree Classifier:')
    # Mean macro Training scores for sklearn
    print('Mean macro train accuracy:',np.mean(scores['train_accuracy']))
    print('Mean macro train precision:',np.mean(scores['train_precision_macro']))
    print('Mean macro train recall:',np.mean(scores['train_recall_macro']))
    # Mean macro Test scores for sklearn
    print('Mean macro test accuracy:',np.mean(scores['test_accuracy']))
    print('Mean macro test precision:',np.mean(scores['test_precision_macro']))
    print('Mean macro test recall:',np.mean(scores['test_recall_macro']))