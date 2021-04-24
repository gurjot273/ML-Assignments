import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    convergence_criteria = 1e-8 # Followed a difference of 10^-8 as convergence criteria, can change to smaller for better results
    learning_rate = 0.05
    training_steps = 5000000 # Maximum training steps, if convergence_criteria not reached till these steps will break
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    x_train = train_data.values[:,0]
    y_train = train_data.values[:,1]
    x_test = test_data.values[:,0]
    y_test = test_data.values[:,1]

    numTraining=x_train.shape[0]
    numTest=x_test.shape[0]
    weight_list=[]
    training_errors=[]
    test_errors=[]
    for n in range(1,10): 
        print('Fitting curve for training data using n = {0}'.format(n))
        weights = np.ones((n+1, 1)) # weights initialisation with weights set to 1
        weights[0,0]=0 # biases set to zero
        phi_train=[ [x**i for x in x_train] for i in range(0,n+1) ]
        phi_train=np.array(phi_train)
        squared_error = 100000
        y_correct=y_train.reshape(1,-1)
        num_steps = 0
        for i in range(training_steps):
            y_pred=weights.transpose().dot(phi_train) # y_pred = WTranspose*Phi_n
            squared_error=np.sum((y_pred-y_correct)**2, axis=1)[0] / (2.0*numTraining); # calculating mean squared error
            grad_weights=phi_train.dot((y_pred-y_correct).transpose())/numTraining # calculating gradients with respect to weights
            weights = weights - learning_rate * grad_weights # gradient descent
            num_steps=i+1
            if i!=0 and previous_error-squared_error<convergence_criteria:
                break # gradient descent has converged
            previous_error=squared_error

        print('Gradient descent has converged after {0} steps with training error={1}'.format(num_steps,squared_error))
        weight_list.append(weights)
        training_errors.append(squared_error) # appending errors for plot
    

    print('****************************************************************************')

    # 1) b) test error calculation
    y_correct=y_test.reshape(1,-1)
    for n in range(1,10): 
        phi_test=np.array([ [x**i for x in x_test] for i in range(0,n+1) ])   
        y_pred=weight_list[n-1].transpose().dot(phi_test) # y_pred = WTranspose*Phi_n
        test_error=np.sum((y_pred-y_correct)**2, axis=1)[0] / (2.0*numTest); # calculating mean squared error
        test_errors.append(test_error)
        print('Using n = {1} test error = {0}'.format(test_error,n)) 
    print('****************************************************************************')

    # 2) a) starts here

    for n in range(1,10): 
        x_input = x_train # input given is training data
        phi_n=[ [x**i for x in x_input] for i in range(0,n+1) ]
        phi_n=np.array(phi_n)
        y_output = (weight_list[n-1].transpose().dot(phi_n)).reshape(-1) # calculated y_predicted for training dataset
        plt.ylabel('Y_val')
        plt.xlabel('X_val')
        plt.title('Plot of curve for n = {0}'.format(n)) 
        plt.plot(x_input,y_output, 'ro') # plotting the curve of degree n
        plt.savefig('curve_2a_{0}.png'.format(n))
        plt.clf()

    # 2) b) starts here

    n_list=[i for i in range(1,10)]
    plt.ylabel('Training and test errors')
    plt.xlabel('n')
    plt.title('Training and test errors vs n') # plotting Training and test errors vs n
    plt.plot(n_list,training_errors, label='Training error')
    plt.plot(n_list,test_errors, label='Test error')
    plt.legend()
    plt.savefig('plot_2b.png')
    plt.clf()
    
    # Best value of n for the dataset is 9 as test and training errors both are minimum for n=9