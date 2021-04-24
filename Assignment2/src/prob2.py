import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import LogisticRegression

#Part 1
def Train(x_train,y_train,lr,convergence,maxIter): # Trains self written logistic regression and returns weights learned, which can be passed to predict for training
    numSamples=x_train.shape[0]
    x0=np.ones((numSamples,1)) # for bias
    x_train=np.append(x0,x_train,axis=1) # appending x0 for bias
    numFeatures=x_train.shape[1]
    weights=np.ones((numFeatures,1)) # with bias, as x0=1 is appended
    weights[0]=0 # bias=0
    cost=0
    prev_cost=0
    for i in range(int(maxIter)):
        prod=x_train.dot(weights)
        h_theta=1.0/(1.0+np.exp(-prod))
        weights=weights-(lr/numSamples)*(x_train.transpose().dot(h_theta-y_train)) # update of weights
        cost=(-1.0/numSamples)*(y_train.transpose().dot(np.log(h_theta))+(1-y_train).transpose().dot(np.log(1-h_theta))) # cross-entropy loss
        if i!=0 and abs(cost-prev_cost)<convergence: # if converged
            break
        prev_cost=cost
    return weights # learned weights

# Part 1
def Predict(x_test,weights):    # Predicts the labels on x_test using weights passed
    numSamples=x_test.shape[0]
    x0=np.ones((numSamples,1)) # for bias
    x_test=np.append(x0,x_test,axis=1) # appending x0 for bias
    prod=x_test.dot(weights)
    h_theta=1.0/(1.0+np.exp(-prod)) # probability for class 1
    h_theta[h_theta<0.5]=0 # threshold = 0.5
    h_theta[h_theta>=0.5]=1
    return h_theta

if __name__ == "__main__":

    df=pd.read_csv('../data/datasetA.csv', sep = ';', header = 0) # loading dataset A for logistic regression
    data=df.values[:]
    x_data=data[:,:-1]
    y_data=data[:,-1:]
    kf = KFold(n_splits=3) # for KFoldCrossValidation

    #Part 2
    model= LogisticRegression(penalty='none',solver='saga',max_iter=1e5,tol=0.0001)  # best hyperparameter tolerance found to be 0.0001 using cross validation below
    

    # Part 3: Cross validating self-written logistic regression classifier 
    # for lr in [1,0.5,0.1,0.05]: used for hyperparameter selection
    #     for convergence in [1e-7,1e-8,1e-9,1e-10]: used for hyperparameter selection
    # best found lr = 0.5, convergence = 1e-9 using cross-validation
    lr=0.5
    convergence=1e-9
    train_accuracy=[]
    train_precision=[]
    train_recall=[]
    test_accuracy=[]
    test_precision=[]
    test_recall=[]
    for train_index, test_index in kf.split(x_data): # generating splits for 3 folds
        x_train=x_data[train_index]
        y_train=y_data[train_index]
        x_test=x_data[test_index]
        y_test=y_data[test_index]
        weights=Train(x_train,y_train,lr,convergence,1e6) # training custom classifier on train_set of current fold
        y_pred=Predict(x_train,weights) # predicting on train_set
        # Training scores for current fold
        train_accuracy.append(accuracy_score(y_train,y_pred))
        train_precision.append(precision_score(y_train,y_pred))
        train_recall.append(recall_score(y_train,y_pred))

        y_pred=Predict(x_test,weights) # Predicting on test_set for current fold
        # Test scores for current fold
        test_accuracy.append(accuracy_score(y_test,y_pred))
        test_precision.append(precision_score(y_test,y_pred))
        test_recall.append(recall_score(y_test,y_pred))
    # Mean Training scores for self-written
    train_accuracy=np.mean(train_accuracy)
    train_precision=np.mean(train_precision)
    train_recall=np.mean(train_recall)
    # Mean Test scores for self-written
    test_accuracy=np.mean(test_accuracy)
    test_precision=np.mean(test_precision)
    test_recall=np.mean(test_recall)
    print('Scores for self-written Logistic Regression Classifier:')
    print('Mean train accuracy:',train_accuracy)
    print('Mean train precision:',train_precision)
    print('Mean train recall:',train_recall)
    print('Mean test accuracy:',test_accuracy)
    print('Mean test precision:',test_precision)
    print('Mean test recall:',test_recall)


    # Part 3:  Cross validating sklearn logistic regression classifier
    model= LogisticRegression(penalty='none',solver='saga',max_iter=1e5,tol=0.0001) # best hyperparameter tolerance found to be 0.0001 using cross validation
    scoring=['accuracy','precision','recall']
    # cross validating using three folds and returning score object
    scores=cross_validate(model,x_data,y_data.reshape(-1),cv=kf,scoring=scoring,return_train_score = True)
    print('Scores for sklearn Logistic Regression Classifier:')
    # Mean Training scores for sklearn
    print('Mean train accuracy:',np.mean(scores['train_accuracy']))
    print('Mean train precision:',np.mean(scores['train_precision']))
    print('Mean train recall:',np.mean(scores['train_recall']))
    # Mean Test scores for sk-learn
    print('Mean test accuracy:',np.mean(scores['test_accuracy']))
    print('Mean test precision:',np.mean(scores['test_precision']))
    print('Mean test recall:',np.mean(scores['test_recall']))