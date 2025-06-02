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
        X[i] = list(map(int, line[1:]))   
        i += 1

