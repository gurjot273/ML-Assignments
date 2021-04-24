import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def relu(input):
    return np.maximum(input,0) # relu calculation

def relu_derivative(input,output):
    return (input>0) # relu derivative

def sigmoid(input):
    return 1.0 / (1.0 + np.exp(-input)) # sigmoid calculation

def sigmoid_derivative(input,output):
    return output*(1-output) # sigmoid derivative

def softmax(x):
    x=x-np.max(x) # for numerical stability, does not change result
    exps=np.exp(x)
    out=exps/np.sum(exps,axis=1,keepdims=True) # softmax calculation
    return out

def categorical_cross_entropy_loss(pred,labels):
    epsilon = 1e-6
    log_prob=np.log(pred+epsilon)
    loss=np.sum(-labels*log_prob)/len(pred) # taking sum of log probabilities of correct labels for all samples
    return loss

def preprocess(): # preprocesses the data, divides into test and train and saves it to data folder
    # Read the lines
    features=[]
    labels=[]
    with open('data/seeds_dataset.txt','r') as f:
        for line in f:
            row=line.split() # get each column in row, last column is label
            row =[float(x) for x in row]
            feature=row[:-1] # features
            label=[0,0,0] # for one hot encoding
            label[int(row[-1])-1]=1 # converting label to one hot encoding
            features.append(feature) # append
            labels.append(label) # append
    features=np.array(features)
    features=(features-np.mean(features,axis=0))/np.std(features,axis=0) # z-normalising the feature matrix by column, each feature is normalised separatelly
    labels=np.array(labels)
    mask=np.random.permutation(len(features)) # mask for shuffing
    features=features[mask] # shuffling the dataset for train test split
    labels=labels[mask]
    train_size = int(0.8*len(features))
    train_features=features[:train_size] # get the train_features split
    train_labels=labels[:train_size] # get the train_labels split
    df = pd.DataFrame(train_features)
    df.to_csv('data/train_features.csv',index=False,header=None) # saving train dataset inside data folder
    df = pd.DataFrame(train_labels)
    df.to_csv('data/train_labels.csv',index=False,header=None)
    test_features=features[train_size:] # get the test_features split
    test_labels=labels[train_size:] # get the test_labels split
    df = pd.DataFrame(test_features)
    df.to_csv('data/test_features.csv',index=False,header=None) # saving test dataset inside data folder
    df = pd.DataFrame(test_labels)
    df.to_csv('data/test_labels.csv',index=False,header=None)

def dataloader(batch_size): # loads and returns training,test data and minibatches
    x_train = pd.read_csv('data/train_features.csv',header=None).values # load the training data
    y_train = pd.read_csv('data/train_labels.csv',header=None).values
    x_test = pd.read_csv('data/test_features.csv',header=None).values # load the test  data
    y_test = pd.read_csv('data/test_labels.csv',header=None).values
    batches=[] # minibatches
    j=0
    while j<len(x_train):
        batch_x=x_train[j:min(j+batch_size,len(x_train))] # minibatch of train_features of batch_size
        batch_y=y_train[j:min(j+batch_size,len(x_train))] # minibatch of train_labels of batch_size
        batches.append([batch_x, batch_y]) # append to batches
        j+=batch_size
    return x_train,y_train,x_test,y_test,batches 

def weight_initialiser(dims): # initialises the weights for training
    weights=[]
    for i in range(len(dims)-1):
        weights.append(np.random.uniform(-1.0,1.0,size=(dims[i],dims[i+1]))) # weight for first layer, from -1 to 1
    return weights

def forward(x_train,weights,activation): # computes a forward pass of the networks, returns the outputs of different layers, last layer output is y_pred
    layers=[x_train]
    for i in range(len(weights)-1): # forward pass of hidden layer
        h_in=np.dot(layers[i],weights[i]) 
        if activation=="sigmoid":
            layers.append(sigmoid(h_in)) # sigmoid
        else:
            layers.append(relu(h_in)) # relu
    h_in=np.dot(layers[-1],weights[-1]) # output layer
    layers.append(softmax(h_in)) # softmax for output layer
    return layers

def backward(x_train,y_train,layers,weights,activation): # computes the loss and returns the gradients of weights with respect to loss
    dweights=[0]*len(weights) #placeholder for gradients for weights in different layers 
    y_pred=layers[-1] # output of last layer
    loss=categorical_cross_entropy_loss(y_pred,y_train) # loss
    i=len(weights)-1
    h_in=np.dot(layers[i],weights[i])
    dh_in=(y_pred-y_train)/len(y_pred) # gradient for input to output layer softmax
    dweights[i]=layers[i].T.dot(dh_in) # gradient for weight of output layer
    dlayer=dh_in.dot(weights[i].T) # gradient for last second layer 
    for i in range(len(weights)-2,-1,-1): # backpropagation for hidden layers
        h_in=np.dot(layers[i],weights[i])
        if activation=="sigmoid":
            dh_in=dlayer*sigmoid_derivative(h_in,layers[i+1]) # gradient for input to sigmoid
        else:
            dh_in=dlayer*relu_derivative(h_in,layers[i+1]) # gradient for input to relu
        dweights[i]=layers[i].T.dot(dh_in) # gradient for current layer weights
        dlayer=dh_in.dot(weights[i].T) # gradient for output of prev layer
    return dweights # gradients of weights with respect to loss

def predict(x_test,y_test,weights,activation): # returns the accuracy using weights on x_test,y_test
    layers=forward(x_test,weights,activation)
    pred=layers[-1] # predicted probabilities for various labels for test data
    y_pred=np.argmax(pred,axis=1) # predicted labels
    correct=0
    for i in range(len(y_pred)):
        if y_test[i,y_pred[i]]==1: # if predicted and correct labels match
            correct+=1
    accuracy=correct/len(y_pred) # correct/total samples
    return accuracy

def train(x_train,y_train,weights,x_test,y_test,lr,epochs,batches,activation): # takes minibatches returned by dataloader as input,minibatch sgd loop, uses forward and backward to train the network
    train_accuracies = []
    test_accuracies = []
    for i in range(1,epochs+1):  # in each epoch
        for j in range(len(batches)): # for each batch
            xtrain_batch=batches[j][0] 
            ytrain_batch=batches[j][1]
            layers=forward(xtrain_batch,weights,activation) # forward pass for minibatch
            dweights=backward(xtrain_batch,ytrain_batch,layers,weights,activation) # backward pass for minibatch
            for k in range(len(weights)):
                weights[k]=weights[k]-lr*dweights[k] #weight update, weights of all layers updated
        if i%10==0: # store train and test accuracy after every 10 epochs
            train_accuracy=predict(x_train,y_train,weights,activation) # calculate train_accuracy using predict
            test_accuracy=predict(x_test,y_test,weights,activation) # calculate test_accuracy using predict
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
    epochs=[10*i for i in range(1,len(train_accuracies)+1)]
    plt.figure() # plot train and test accuracy after every 10 epochs
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    tr = plt.plot(epochs,train_accuracies) # plot train accuracy
    te = plt.plot(epochs,test_accuracies)  # plot test accuracy
    plt.legend([tr[0],te[0]],('Train accuracy','Test accuracy'),loc=0)
    if activation=="sigmoid":
        plt.title('Train and test accuracy vs epochs for specification 1A')
        plt.savefig('plot_1A.png')
        print("Plot 1A has been generated in a window. Please close the window to proceed further")
    else:
        plt.title('Train and test accuracy vs epochs for specification 1B')
        plt.savefig('plot_1B.png')
        print("Plot 1B has been generated in a window. Please close the window to proceed further")
    plt.show() # Displays plot in a window, Need to close the window so that code resumes working
    return weights # return learned weights



if __name__ == "__main__":
    preprocess() # preprocess data,creates train_features.csv,train_labels.csv,test_features.csv,test_labels.csv inside data folder
    x_train,y_train,x_test,y_test,batches = dataloader(batch_size=32) # loads train,test data and minibatches
    print("Part 1A :")
    dims=[x_train.shape[1],32,3]
    weights = weight_initialiser(dims) # weight initialisation
    weights =  train(x_train,y_train,weights,x_test,y_test,lr=0.01,epochs=200,batches=batches,activation="sigmoid") # returns learned weights training using batches for 1A
    print("Final training accuracy is ",predict(x_train,y_train,weights,"sigmoid")) # use predict to find train accuracy
    print("Final test accuracy is ",predict(x_test,y_test,weights,"sigmoid")) # use predict to find test accuracy
    print("Part 1B :")
    dims=[x_train.shape[1],64,32,3]
    weights = weight_initialiser(dims)  # weight initialisation
    weights =  train(x_train,y_train,weights,x_test,y_test,lr=0.01,epochs=200,batches=batches,activation="relu") # returns learned weights training using batches for 1A
    print("Final training accuracy is ",predict(x_train,y_train,weights,"relu")) # use predict to find train accuracy
    print("Final test accuracy is ",predict(x_test,y_test,weights,"relu")) # use predict to find test accuracy