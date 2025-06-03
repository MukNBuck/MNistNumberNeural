# double layer neural netowrk - TO TRAIN OFF TRAIN DATASET
# #file open
import numpy as np
import csv
import time
import os



def relu(x):
    return np.maximum(0, x)

def deriv_relu(x):
    return (x > 0).astype(float)
      
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # prevent overflow
    return e_x / np.sum(e_x, axis=1, keepdims=True)



def main():
    start_time = time.time()
    i = 0
    Y = np.zeros((60000, 10))
    X = np.zeros((60000, 784))

    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "Data", "mnist_train.csv")

    # Load data
    np.set_printoptions(threshold=np.inf)
    with open(file_path, "r") as csvfile:
        csv_reader = csv.reader(csvfile)
        for line in csv_reader:
            Y[i][int(line[0])] = 1
            X[i] = np.array(list(map(float, line[1:]))) / 255
            i += 1
    running = 1
    input_size = 784   #input feaTURES
    hidden_neurons = 30 # hidden neurons
    output_size = 10
    learning_rate = .01
    batch_size = 500
    batches = 120
    shuffles = 500

    # Initialize random weights and zeros for bases
    limit = np.sqrt(6 / (input_size + hidden_neurons))
    Bx = np.zeros((1, hidden_neurons))
    Wx = np.random.uniform(-limit, limit, size=(input_size   , hidden_neurons))

    limit = np.sqrt(6 / (hidden_neurons + hidden_neurons))
    Ba = np.zeros((1, hidden_neurons)) # bias
    Wa = np.random.uniform(-limit, limit, size=(hidden_neurons, hidden_neurons)) #weights - making both hidden layers same size so can initially be same inputs for random

    limit = np.sqrt(6 / (hidden_neurons + output_size))
    Wb = np.random.uniform(-limit, limit, size=(hidden_neurons, output_size))
    Bb = np.zeros((1, output_size))

    # Repeats using train set, shuffling data rach time
    for shuffle in range(shuffles):
        #Shuffle data
        i = np.arange(X.shape[0])
        np.random.shuffle(i)
        X = X[i]
        Y = Y[i]
        # Does mini-batches
        for batch in range(batches):
            start = batch * batch_size
            end = start + batch_size
            X_batch = X[start:end]
            Y_batch = Y[start:end]

            #forward pass
            A = relu(np.dot(X_batch, Wx) + Bx)
            B = relu(np.dot(A, Wa) + Ba)
            output = softmax(np.dot(B, Wb) + Bb)

            #Loss
            Loss = .5 * np.mean((Y_batch - output) **2)


            #Backpropagation
            dZb = (output - Y_batch) / Y_batch.shape[0]
            dWb = np.dot(np.transpose(B), dZb)
            dBb = np.sum(dZb, axis=0, keepdims=True)
            dB = np.dot(dZb, np.transpose(Wb))

            dZa = dB * deriv_relu(np.dot(A, Wa) + Ba)
            dWa = np.dot(np.transpose(A), dZa)
            dBa = np.sum(dZa, axis = 0, keepdims=True)
            dA = np.dot(dZa, np.transpose(Wa))


            dZx = dA * deriv_relu(np.dot(X_batch, Wx) + Bx)
            dWx = np.dot(np.transpose(X_batch), dZx)
            dBx = np.sum(dZx, axis=0, keepdims=True)

            #Update weights
            Wb -= learning_rate*dWb
            Bb -= learning_rate * dBb
            Wa -= learning_rate * dWa
            Ba -= learning_rate * dBa
            Bx -= learning_rate * dBx
            Wx -= learning_rate*dWx

            #Accuracy
            predictions = np.argmax(output, axis =1)
            labels = np.argmax(Y_batch, axis=1)
            accuracy = np.mean(predictions == labels)
            
        print(f"Shuffle {shuffle + 1} Loss: {Loss:.4f}, Accuracy: {accuracy * 100:.2f}%")

    #Save weights 
    np.savez("weights.npz", Wx=Wx, Wa=Wa, Wb = Wb, Bx=Bx, Ba=Ba, Bb = Bb)
    print("Saving to:", os.path.abspath("weights.npz"))

    print("Done!")
    end_time = time.time()
    print(f"The execution time of the code is: {end_time - start_time} seconds")

if __name__ == '__main__':
    main()