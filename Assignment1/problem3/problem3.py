import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Best value of n for the dataset is 9 as test and training errors both are minimum for n=9
    # Worst value of n for the dataset is 1 as test and training errors both are minimum for n=9
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

    # 3) a)
    worst_n=1 # from part 2
    best_n=9 # from part 2
    lasso_training_errors_best=[]
    lasso_training_errors_worst=[]
    lasso_test_errors_best=[]
    lasso_test_errors_worst=[]

    for n in [worst_n,best_n]:
        for reg in [0.25,0.5,0.75,1.0]:
            print('Performing Lasso regression using n = {0} and lamda = {1}'.format(n,reg))
            weights = np.ones((n+1, 1)) # weights initialsation with weights set to 1
            weights[0,0]=0 # biases set to zero
            phi_train=[ [x**i for x in x_train] for i in range(0,n+1) ]
            phi_train=np.array(phi_train)
            lasso_error = 100000
            y_correct=y_train.reshape(1,-1)
            num_steps = 0
            for i in range(training_steps):
                y_pred=weights.transpose().dot(phi_train) # y_pred = WTranspose*Phi_n
                lasso_error=np.sum((y_pred-y_correct)**2, axis=1)[0] / (2.0*numTraining) # calculating mean squared error
                lasso_reg_error = lasso_error + (reg/(2.0*numTraining))*(np.sum(weights,axis=0)[0]-weights[0,0]) # adding regularization loss
                grad_weights=phi_train.dot((y_pred-y_correct).transpose())/numTraining # calculating gradient due to squared error
                reg_grad=(reg/(2.0*numTraining))*np.ones((n+1,1))
                reg_grad[0,0]=0 # since w0 is not included in regularisation loss as in slides
                grad_weights= grad_weights + reg_grad # adding gradient due to regularization term
                weights = weights - learning_rate * grad_weights # gradient descent
                num_steps=i+1
                if i!=0 and previous_error-lasso_error<convergence_criteria:
                    break # gradient descent has converged
                previous_error=lasso_error

            y_correct=y_test.reshape(1,-1)
            print('Gradient descent has converged after {0} steps with training error={1}'.format(num_steps,lasso_error))
            phi_test=np.array([ [x**i for x in x_test] for i in range(0,n+1) ])   
            y_pred=weights.transpose().dot(phi_test) # y_pred = WTranspose*Phi_n
            test_lasso_error= np.sum((y_pred-y_correct)**2, axis=1)[0] / (2.0*numTest); # calculating mean squared error
            test_reg_lasso_error = test_lasso_error + (reg/(2.0*numTest))*(np.sum(weights,axis=0)[0]-weights[0,0]) # adding regularization loss
            print('Test error : {0}'.format(test_lasso_error,n)) 
            if n==worst_n:
                lasso_training_errors_worst.append(lasso_error) # mean squared errors without reg terms
                lasso_test_errors_worst.append(test_lasso_error) # mean squared errors without reg terms
            else:
                lasso_training_errors_best.append(lasso_error)  # mean squared errors without reg terms
                lasso_test_errors_best.append(test_lasso_error) # mean squared errors without reg terms
        print('****************************************************************************')

    # 3) a) plotting for n=9
    reg_list=[0.25,0.5,0.75,1.0]

    plt.ylabel('Lasso training and test errors')
    plt.xlabel('Regularization')
    plt.title('Lasso training and test errors vs regularisation for n = {0}'.format(best_n))
    plt.plot(reg_list,lasso_training_errors_best, label='Training error') # plot lasso training error
    plt.plot(reg_list,lasso_test_errors_best, label='Test error') # plot lasso test error
    plt.legend()
    plt.savefig('plot_3b_lasso_{0}.png'.format(best_n))
    plt.clf()

    # 3) a) plotting for n=1
    plt.ylabel('Lasso training and test errors')
    plt.xlabel('Regularization')
    plt.title('Lasso training and test errors vs regularisation for n = {0}'.format(worst_n))
    plt.plot(reg_list,lasso_training_errors_worst, label='Training error') # plot lasso training error
    plt.plot(reg_list,lasso_test_errors_worst, label='Test error')# plot lasso test error
    plt.legend()
    plt.savefig('plot_3b_lasso_{0}.png'.format(worst_n))
    plt.clf()

    # 3) b)
    ridge_training_errors_best=[]
    ridge_training_errors_worst=[]
    ridge_test_errors_best=[]
    ridge_test_errors_worst=[]

    for n in [worst_n,best_n]:
        for reg in [0.25,0.5,0.75,1.0]:
            print('Performing Ridge regression using n = {0} and lamda = {1}'.format(n,reg))
            weights = np.ones((n+1, 1)) # weights initialsation with weights set to 1
            weights[0,0]=0 # biases set to zero
            phi_train=[ [x**i for x in x_train] for i in range(0,n+1) ]
            phi_train=np.array(phi_train)
            ridge_error = 100000
            y_correct=y_train.reshape(1,-1)
            num_steps = 0
            for i in range(training_steps):
                y_pred=weights.transpose().dot(phi_train) # y_pred = WTranspose*Phi_n
                ridge_error=np.sum((y_pred-y_correct)**2, axis=1)[0] / (2.0*numTraining) # calculating mean squared error
                ridge_reg_error = ridge_error + (reg/(2.0*numTraining))*(np.sum(weights**2,axis=0)[0]-(weights[0,0])**2) # adding regularization loss
                grad_weights=phi_train.dot((y_pred-y_correct).transpose())/numTraining # calculating gradient due to squared error
                reg_grad=(reg/numTraining)*weights
                reg_grad[0,0]=0 # since w0 is not included in regularisation loss as in slides
                grad_weights= grad_weights + reg_grad # adding gradient due to regularization term
                weights = weights - learning_rate * grad_weights # gradient descent
                num_steps=i+1
                if i!=0 and previous_error-ridge_error<convergence_criteria:
                    break # gradient descent has converged
                previous_error=ridge_error

            y_correct=y_test.reshape(1,-1)
            print('Gradient descent has converged after {0} steps with training error={1}'.format(num_steps,ridge_error))
            phi_test=np.array([ [x**i for x in x_test] for i in range(0,n+1) ])   
            y_pred=weights.transpose().dot(phi_test) # y_pred = WTranspose*Phi_n
            test_ridge_error= np.sum((y_pred-y_correct)**2, axis=1)[0] / (2.0*numTest); # calculating mean squared error
            test_reg_ridge_error = test_ridge_error + (reg/(2.0*numTest))*(np.sum(weights**2,axis=0)[0]-(weights[0,0])**2) # adding regularization loss
            print('Test error : {0}'.format(test_ridge_error,n)) 
            if n==worst_n:
                ridge_training_errors_worst.append(ridge_error) # mean squared errors without reg terms
                ridge_test_errors_worst.append(test_ridge_error) # mean squared errors without reg terms
            else:
                ridge_training_errors_best.append(ridge_error) # mean squared errors without reg terms
                ridge_test_errors_best.append(test_ridge_error) # mean squared errors without reg terms
        print('****************************************************************************')
                

    # 3) b) plotting for n=9
    plt.ylabel('Ridge training and test errors')
    plt.xlabel('Regularization')
    plt.title('Ridge training and test errors vs regularisation for n={0}'.format(best_n))
    plt.plot(reg_list,ridge_training_errors_best, label='Training error') # plot training error
    plt.plot(reg_list,ridge_test_errors_best, label='Test error') # plot test error
    plt.legend()
    plt.savefig('plot_3b_ridge_{0}.png'.format(best_n))
    plt.clf()

    # 3) b) plotting for n=1
    plt.ylabel('Ridge training and test errors')
    plt.xlabel('Regularization')
    plt.title('Ridge training and test errors vs regularisation for n={0}'.format(worst_n))
    plt.plot(reg_list,ridge_training_errors_worst, label='Training error') # plot ridge training error
    plt.plot(reg_list,ridge_test_errors_worst, label='Test error') # plot ridge test error
    plt.legend()
    plt.savefig('plot_3b_ridge_{0}.png'.format(worst_n))
    plt.clf()