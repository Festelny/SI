import numpy as np
import matplotlib.pyplot as plt

from data import get_data, inspect_data, split_data

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

# TODO: calculate closed-form solution
theta_best = [0, 0]

matrix_obs = np.ones((x_train.shape[0],2))
matrix_obs[:,1] = x_train
theta_opt =np.dot(np.dot((np.linalg.inv(np.dot(matrix_obs.T,matrix_obs))),matrix_obs.T),y_train)

# TODO: calculate error
ms_error=np.mean(np.square(np.dot(matrix_obs,theta_opt)-y_train))

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
y1 = float(theta_opt[0]) + float(theta_opt[1]) * x
#plt.plot(x, y)
plt.plot(x,y1)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

# TODO: standardization

x_train_std = ((x_train - np.mean(x_train)) / np.std(x_train))
y_train_std = ((y_train - np.mean(y_train)) / np.std(y_train))
x_test_std = ((x_test - np.mean(x_test)) / np.std(x_test))

# TODO: calculate theta using Batch Gradient Descent

def gradient_des(learnrate,matrix_obs2,matrix):
    size=matrix_obs2.shape[0]
    theta = np.random.randn(2,1)
    test=np.full_like(theta, 2)
    while((theta-test).all()):
        test=theta  
        tmp = ((matrix_obs2.dot(theta)) - matrix.reshape(-1,1))
        gradient = 2 / size * (matrix_obs2.T.dot(tmp))       
        theta = theta - learnrate * gradient
    return theta

matrix_obs2 = np.ones((x_train_std.shape[0],2))
matrix_obs2[:,1] = x_train_std

learnrate=0.001

theta_grad = gradient_des(learnrate,matrix_obs2,y_train_std)
theta_unstandarized = theta_grad.copy()
theta_unstandarized[1] = theta_unstandarized[1] * np.std(y_train) / np.std(x_train)
theta_unstandarized[0] = np.mean(y_train) - theta_unstandarized[1] * np.mean(x_train)
theta_unstandarized = theta_unstandarized.reshape(-1)
#theta_unstandardized = np.zeros(2)
#theta_unstandardized[0] = theta_grad.item(0) - theta_grad.item(1) * (np.mean(x_train) / np.std(x_train))
#theta_unstandardized[1] = theta_grad.item(1) / np.std(x_train)

# TODO: calculate error

ms_error2 = np.mean(np.square(np.dot(matrix_obs2,theta_grad)-y_train))

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_unstandarized[0]) + float(theta_unstandarized[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()
