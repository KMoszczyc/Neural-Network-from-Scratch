import numpy as np
import random
import matplotlib.pyplot as plt


class Adaline:
    def __init__(self, inputs_num, max_iterations=10000, lr=0.2, theta=0, weights_range=1, min_cost=0.35):
        self.max_iterations = max_iterations
        self.lr = lr
        self.weights = np.random.rand(inputs_num) * weights_range * 2 - weights_range
        # self.weights = np.zeros(inputs_num)
        self.bias = 1
        self.theta = theta
        self.cost = []
        self.min_cost = min_cost
        # print(self.weights)

    def predict(self, inputs):
        values = np.dot(inputs, self.weights) + self.bias
        # sum = np.dot(inputs, self.weights)

        return self.bipolar_activation(values)

    def raw_predict(self, inputs):
        return np.dot(inputs, self.weights) + self.bias

    def unipolar_activation(self, values):
        if values > self.theta:
            return 1
        else:
            return 0

    def bipolar_activation(self, values):
        if values > self.theta:
            return 1
        else:
            return -1

    def train(self, training_inputs, labels):
        epochs_num = 0
        for i in range(self.max_iterations):
            cost = []
            epochs_num = i

            for inputs, label in zip(training_inputs, labels):
                prediction = self.raw_predict(inputs)
                # prediction = self.predict(inputs)

                errors = label - prediction
                self.weights += self.lr * inputs * errors
                self.bias += self.lr * errors
                cost.append(errors ** 2)

            lms = sum(cost) / len(cost)
            self.cost.append(lms)
            # print(str(lms) + " " + str(self.min_cost))
            if lms < self.min_cost:
                # print(self.cost)
                break

        return epochs_num

    def is_correct(self, training_inputs, labels):
        is_wrong = False
        for inputs, label in zip(training_inputs, labels):
            prediction = self.predict(inputs)
            if label != prediction:
                is_wrong = True
        return not is_wrong


def printArray(arr):
    s = ""
    for elem in arr:
        s += str(elem) + "\t"
    print(s)


# bipolar data
training_inputs = [np.array([1, 1]),
                   np.array([1, -1]),
                   np.array([-1, 1]),
                   np.array([-1, -1])]
labels = np.array([1, -1, -1, -1])

epochs = 0
iterations_num = 100
s = ""
correct = ""
correct_count = 0
costs = ""
# #zad 1
# weights_ranges=[0, 0.001, 0.01, 0.1, 0.3, 0.5, 0.9, 1]
# for wr in weights_ranges:
#     epochs = 0
#     correct_count = 0
#     for i in range(iterations_num):
#         adaline = Adaline(2, weights_range = wr)
#         epochs += adaline.train(training_inputs, labels)
#
#         if adaline.is_correct(training_inputs,labels):
#             correct_count+=1
#     print("weights range: " +str(wr) +" avg num of epochs: "+ str(epochs/iterations_num) + " correct: "+str(correct_count))
#     s+=str(epochs/iterations_num) + "\t"
#     correct += str(correct_count) + "\t"
# printArray(weights_ranges)

# zad 2
# learning_rates = [0.001, 0.01, 0.03, 0.05, 0.1, 0.15, 0.3, 0.5, 0.9]
# for lr in learning_rates:
#     epochs = 0
#     correct_count = 0
#     for i in range(iterations_num):
#         adaline = Adaline(2, lr=lr)
#         epochs += adaline.train(training_inputs, labels)
#         if adaline.is_correct(training_inputs, labels):
#             correct_count += 1
#     print("lr: " + str(lr) + " avg num of epochs: " + str(epochs / iterations_num) + " correct: " + str(correct_count))
#     s += str(epochs / iterations_num) + "\t"
#     correct += str(correct_count) + "\t"
# printArray(learning_rates)

# zad 3
min_costs = [2, 1.5, 1, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.1]
# min_costs = [2, 1.5, 1, 0.7, 0.6, 0.5]

for min_cost in min_costs:
    epochs = 0
    correct_count = 0
    cost=0
    for i in range(iterations_num):
        adaline = Adaline(2, min_cost=min_cost)
        epochs += adaline.train(training_inputs, labels)
        if adaline.is_correct(training_inputs, labels):
            correct_count += 1
        cost= adaline.cost[-1]
    print("min cost: " + str(min_cost) + " avg num of epochs: " + str(epochs / iterations_num) + " correct: " + str(
        correct_count) + " real cost: "+ "%.2f" % cost)
    s += str(epochs / iterations_num) + "\t"
    correct += str(correct_count) + "\t"
    costs += "%.2f" % cost + "\t"
printArray(min_costs)

# zad 4 porównanie adaline

# for i in range(iterations_num):
#     adaline = Adaline(2)
#     epochs += adaline.train(training_inputs, labels)
#     if adaline.is_correct(training_inputs, labels):
#         correct_count += 1
# print(" avg num of epochs: " + str(epochs / iterations_num) + " correct: " + str(
#     correct_count))
# s += str(epochs / iterations_num) + "\t"
# correct += str(correct_count) + "\t"

print(s)
print(costs)
print(correct)
print(correct_count)

# test
adaline = Adaline(2, weights_range=1)
epochs += adaline.train(training_inputs, labels)

inputs = np.array([1, 1])
print(adaline.predict(inputs))

inputs = np.array([-1, 1])
print(adaline.predict(inputs))

inputs = np.array([1, -1])
print(adaline.predict(inputs))

inputs = np.array([-1, -1])
print(adaline.predict(inputs))

print(str(adaline.weights) + " " + str(adaline.bias))
print(adaline.cost)
print(epochs)
# plt.plot(adaline.cost)
# plt.ylabel('cost')
# plt.xlabel('epochs')
# plt.show()
