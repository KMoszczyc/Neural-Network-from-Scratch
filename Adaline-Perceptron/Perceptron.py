import numpy as np
import random
#------------------------------------------------------------------

class Perceptron:
    def __init__(self, inputs_num, max_iterations=10000, lr=1, theta = 0, weights_range = 1):
        self.max_iterations = max_iterations
        self.lr = lr
        self.weights = np.random.rand(inputs_num)*weights_range*2-weights_range
        # self.weights = np.zeros(inputs_num)
        self.bias = 1
        self.theta = theta
        # print(self.weights)

    def predict(self, inputs):
        sum = np.dot(inputs, self.weights) + self.bias
        # sum = np.dot(inputs, self.weights)

        return self.bipolar_activation(sum)

    def unipolar_activation(self, sum):
        if sum > self.theta:
            return 1
        else:
            return 0

    def bipolar_activation(self, sum):
        if sum > self.theta:
            return 1
        else:
            return -1

    def train(self, training_inputs, labels):
        epochs_num = 0
        for i in range(self.max_iterations):
            # print("epoch: "+str(i))
            stop_learning = True
            epochs_num = i
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                # print(str(prediction) + ": "+ str(inputs[0]) +" "+str(inputs[1]))
                self.weights += self.lr * (label - prediction) * inputs
                self.bias += self.lr * (label - prediction)
                if label!=prediction:
                    stop_learning = False

            if stop_learning:
                break

        return epochs_num



def printArray(arr):
    s=""
    for elem in arr:
        s+=str(elem) + "\t"
    print(s)

#data
#------------------------------------------------------------------

# unipolar data
# training_inputs = [np.array([1, 1]),
#                    np.array([1, 0]),
#                    np.array([0, 1]),
#                    np.array([0, 0])]
# labels = np.array([1, 0, 0, 0])

# bipolar data
training_inputs = [np.array([1, 1]),
                   np.array([1, -1]),
                   np.array([-1, 1]),
                   np.array([-1, -1])]
labels = np.array([1, -1, -1, -1])


#tests, research simulations
#------------------------------------------------------------------

sum = 0
iterations_num = 100
s=""

#zad 2
weights_ranges=[0, 0.001, 0.01, 0.1, 0.3, 0.5, 0.9, 1]
for wr in weights_ranges:
    sum = 0
    for i in range(iterations_num):
        perceptron = Perceptron(2, weights_range = wr)
        sum += perceptron.train(training_inputs, labels)

    print("weights range: " +str(wr) +" avg num of epochs: "+ str(sum/iterations_num))
    s+=str(sum/iterations_num) + "\t"

printArray(weights_ranges)
print(s)

#zad 3
# learning_rates=[0.001, 0.01,0.1,0.3,0.5,0.9, 1]
# for lr in learning_rates:
#     sum = 0
#     for i in range(iterations_num):
#         perceptron = Perceptron(2, lr=lr)
#         sum += perceptron.train(training_inputs, labels)
#     print("lr: " +str(lr) +" avg num of epochs: "+ str(sum/iterations_num))
#     s += str(sum / iterations_num) + "\t"
# printArray(learning_rates)
# print(s)

# #zad 4
# for i in range(iterations_num):
#     perceptron = Perceptron(2)
#     sum += perceptron.train(training_inputs, labels)
# print("avg num of epochs: "+ str(sum/iterations_num))
# s += str(sum / iterations_num) + "\t"
# print(s)



#test
perceptron = Perceptron(2, weights_range=1)
sum += perceptron.train(training_inputs, labels)

# inputs = np.array([1, 1])
# print(perceptron.predict(inputs))
#
# inputs = np.array([0, 1])
# print(perceptron.predict(inputs))
#
# inputs = np.array([1, 0])
# print(perceptron.predict(inputs))
#
# inputs = np.array([0, 0])
# print(perceptron.predict(inputs))

inputs = np.array([1, 1])
print(perceptron.predict(inputs))

inputs = np.array([-1, 1])
print(perceptron.predict(inputs))

inputs = np.array([1, -1])
print(perceptron.predict(inputs))

inputs = np.array([-1, -1])
print(perceptron.predict(inputs))


print(str(perceptron.weights) +" "+ str(perceptron.bias))

