#file open
import numpy as np
import csv
import doublelayerMnist as dm
import os

i = 0
Y = np.zeros((10000, 10))
X = np.zeros((10000, 784))

base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path, "Data", "mnist_test.csv")

np.set_printoptions(threshold=np.inf)
with open(file_path, "r") as csvfile:
    csv_reader = csv.reader(csvfile)
    for line in csv_reader:
        Y[i][int(line[0])] = 1
        X[i] = np.array(list(map(int, line[1:]))) / 255   
        i += 1

weights = np.load("weights.npz")
Wx = weights['Wx']
Wa = weights['Wa']
Wb = weights['Wb']
Bx = weights['Bx']
Ba = weights['Ba']
Bb = weights['Bb']

# Forward pass
A = dm.relu(np.dot(X, Wx) + Bx)
B = dm.relu(np.dot(A, Wa) + Ba)
output = dm.softmax(np.dot(B, Wb) + Bb)

# Predictions and accuracy
predictions = np.argmax(output, axis=1)
labels = np.argmax(Y, axis=1)
accuracy = np.mean(predictions==labels)

print(f"Test accuracy: {accuracy * 100:.2f}%")
