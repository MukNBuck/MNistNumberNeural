#numpy fuckaround
import numpy as np


def sigmoid(x):
    return 1 / (1+np.exp(-x))

def deriv_sigmoid(x):
      return sigmoid(x) * (1 - sigmoid(x))

running = 1
n_in = 2 #input feaTURES
n_out = 4 # hidden neurons
limit = np.sqrt(6 / (n_in + n_out))
learning_rate = .01
Bx = np.zeros((1, n_out))
Wx = np.random.uniform(-limit, limit, size=(n_in, n_out))

n_in = 4 # 4 in each matrix of hidden layer
n_out = 1 # 1 outputs - dog or not
limit = np.sqrt(6 / (n_in + n_out))
Ba = np.zeros((1, n_out)) # bias
Wa = np.random.uniform(-limit, limit, size=(n_in, n_out)) #weights
for x in range(10000):




    X = np.array([
        [0, 0],   # not dog
        [0, 1],   # not dog
        [1, 0],   # A dog
        [1, 1]    # A dog
    ])


    print("Inputs (X):")
    print(X)
    print("Wx:")
    print(Wx)
    print("X . Wx")
    print(np.dot(X, Wx))
    print("X . Wx + Bx:")
    print(np.dot(X, Wx) + Bx)
    print("Activation:")
    A = sigmoid(np.dot(X, Wx) + Bx)
    print(A)

    print("---------------------------------------------------")
    print("First activation done! Moving on to output values!")
    print("Activation (A): ")
    print(A)
    print("Wa:")
    print(Wa)
    print("A . Wa")
    print(np.dot(A, Wa))
    print("A . Wa + Ba")
    print((np.dot(A, Wa) + Ba))
    output = sigmoid((np.dot(A, Wa) + Ba))
    print("Output:")
    print(output)
    print("---------------------------------------")
    print("Output Calculated! Calculatiing loss and Error...")
    Y = np.array([[0], [0], [1], [1]])
    error = 0
    for i in range(len(Y)):
            error += (output[i] - Y[i]) **2

    error = error / (len(Y))
    print("Mean squared Error: ")
    print(error)
    Loss = np.dot(.5, (Y - output) **2)
    print("Loss:")
    print(Loss)

    dZa = (output - Y) * output * (1 - output)
    dWa = np.dot(np.transpose(A), dZa)
    dBa = np.sum(dZa, axis=0)

    dA = np.dot(dZa, np.transpose(Wa))
    dZx = dA * A * (1 - A)
    dWx = np.dot(np.transpose(X), dZx)
    dBx = np.sum(dZx, axis = 0)

    Wx = Wx- learning_rate*dWx
    Wa = Wa - learning_rate * dWa
    Bx = Bx - learning_rate * dBx
    Ba = Ba - learning_rate * dBa


