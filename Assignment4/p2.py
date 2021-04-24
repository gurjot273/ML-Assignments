import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning

def dataloader(): # loads and returns training,test data
    x_train = pd.read_csv('data/train_features.csv',header=None).values
    y_train = pd.read_csv('data/train_labels.csv',header=None).values
    x_test = pd.read_csv('data/test_features.csv',header=None).values
    y_test = pd.read_csv('data/test_labels.csv',header=None).values
    return x_train,y_train,x_test,y_test

if __name__ == "__main__":
    x_train,y_train,x_test,y_test = dataloader() # same data used for Part 1
    warnings.filterwarnings(action='ignore', category=ConvergenceWarning) # turn off convergence warning as only 200 epochs
    print("Part 2 Specification 1A :")
    #same specification as 1A, so sgd solver used
    mlp = MLPClassifier(hidden_layer_sizes=(32,),activation='logistic',solver='sgd',batch_size=32,learning_rate_init=0.01,max_iter=200,alpha=0.0) 
    mlp.fit(x_train,y_train) # fit to training data
    pred_mlp = mlp.predict(x_train) # predicted training labels
    print("Final training accuracy is ",accuracy_score(y_train, pred_mlp)) # final training accuracy
    pred_mlp = mlp.predict(x_test) # predicted test labels
    print("Final test accuracy is ",accuracy_score(y_test, pred_mlp)) # final test accuracy
    print("Part 2 Specification 1B :") 
    #same specification as 1B, so sgd solver used
    mlp = MLPClassifier(hidden_layer_sizes=(64,32),activation='relu',solver='sgd',batch_size=32,learning_rate_init=0.01,max_iter=200,alpha=0.0)
    mlp.fit(x_train,y_train) # fit to training data
    pred_mlp = mlp.predict(x_train)  # predicted training labels
    print("Final training accuracy is ",accuracy_score(y_train, pred_mlp)) # final training accuracy
    pred_mlp = mlp.predict(x_test) # predicted test labels
    print("Final test accuracy is ",accuracy_score(y_test, pred_mlp)) # final test accuracy