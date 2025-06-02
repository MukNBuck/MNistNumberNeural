#file open
import numpy as np
import csv



i = 0
Y = np.zeros((10000, 10))
X = np.zeros((10000, 784))

np.set_printoptions(threshold=np.inf)
with open("D:/projects/Data/mnist_test.csv", "r") as csvfile:
    csv_reader = csv.reader(csvfile)
    for line in csv_reader:
        Y[i][int(line[0])] = 1
        X[i] = np.array(list(map(float, line[1:]))) / 255
        i += 1


def sigmoid(x):
    # return 1 / (1+np.exp(-x))
    return np.maximum(0, x)

def deriv_sigmoid(x):
      # return sigmoid(x) * (1 - sigmoid(x))
    return (x > 0).astype(float)
      
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # prevent overflow
    return e_x / np.sum(e_x, axis=1, keepdims=True)

running = 1
n_in = 784 #input feaTURES
n_out = 10  # hidden neurons
limit = np.sqrt(6 / (n_in + n_out))
learning_rate = .01
Bx = np.zeros((1, n_out))
Wx = np.random.uniform(-limit, limit, size=(n_in, n_out))

n_in = 10 # number of neurons
n_out = 10 # 10 outputs
limit = np.sqrt(6 / (n_in + n_out))
Ba = np.zeros((1, n_out)) # bias
#data = np.load("weights.npz")
#Wx = data["Wx"]
#Wa = data["Wa"]
#Bx = data["Bx"]
#Ba = data["Ba"]
Wa = np.random.uniform(-limit, limit, size=(n_in, n_out)) #weights
for x in range(10000):


    """print("Inputs (X):")
    print(X)
    print("Wx:")
    print(Wx)
    print("X . Wx")
    print(np.dot(X, Wx))
    print("X . Wx + Bx:")
    print(np.dot(X, Wx) + Bx)
    print("Activation:")
     """
    A = sigmoid(np.dot(X, Wx) + Bx)
    """
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
     """
    # output = sigmoid((np.dot(A, Wa) + Ba))
    output = softmax(np.dot(A, Wa) + Ba)
    """
    print("Output:")
    print(output)
    print("---------------------------------------")
    print("Output Calculated! Calculatiing loss and Error...")
     """

    error = np.mean((output - Y) ** 2)
    #print("Mean squared Error: ")
    #print(error)
    Loss = np.dot(.5, (Y - output) **2)
    """
    print("Loss:")
    print(Loss)
    """

    # dZa = (output - Y) * output * (1 - output)
    dZa = (output - Y) / Y.shape[0]
    dWa = np.dot(np.transpose(A), dZa)
    dBa = np.sum(dZa, axis=0)

    dA = np.dot(dZa, np.transpose(Wa))
    dZx = dA * deriv_sigmoid(np.dot(X, Wx) + Bx)
    dWx = np.dot(np.transpose(X), dZx)
    dBx = np.sum(dZx, axis = 0)

    Wx = Wx- learning_rate*dWx
    Wa = Wa - learning_rate * dWa
    Bx = Bx - learning_rate * dBx
    Ba = Ba - learning_rate * dBa

    if x % 100 == 0:
        print(f"Hi {x}")


np.savez("weights.npz", Wx=Wx, Wa=Wa, Bx=Bx, Ba=Ba)
print(error)
print("Done!")