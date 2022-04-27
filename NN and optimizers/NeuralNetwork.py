import numpy as np
import matplotlib.pyplot as plt
import math


class NeuralNetwork:
    def __init__(self, *layers, weights_range=0.01, weight_init_method="He"):
        self.weights = []
        self.bias = []
        self.best_val_accuracy = 0
        self.best_epoch = 0
        self.weight_ranges = weights_range

        for i in range(len(layers) - 1):

            if weight_init_method == "":
                self.weights.append(np.random.randn(layers[i], layers[i + 1]) * self.weight_ranges)

            if weight_init_method == "He":
                self.weights.append(np.random.randn(layers[i], layers[i + 1]) * np.sqrt(2 / layers[i]))

            if weight_init_method == "Xavier":
                self.weights.append(np.random.randn(layers[i], layers[i + 1]) * np.sqrt(2 / (layers[i] + layers[i + 1])))

            self.bias.append(np.random.randn(layers[i + 1]) * self.weight_ranges)

        # for Optimizers
        self.prev_weight_updates = np.asarray([np.zeros(self.weights[i].shape) for i in range(len(self.weights))])
        self.prev_bias_updates = np.asarray([np.zeros(self.bias[i].shape) for i in range(len(self.bias))])

        self.accumulated_weight_updates = np.asarray([np.zeros(self.weights[i].shape) for i in range(len(self.weights))])
        self.accumulated_bias_updates = np.asarray([np.zeros(self.bias[i].shape) for i in range(len(self.bias))])

        self.accumulated_weight_gradients = np.asarray([np.zeros(self.weights[i].shape) for i in range(len(self.weights))])
        self.accumulated_bias_gradients = np.asarray([np.zeros(self.bias[i].shape) for i in range(len(self.bias))])

        self.accumulated_weight_gradients_sqr = np.asarray([np.zeros(self.weights[i].shape) for i in range(len(self.weights))])
        self.accumulated_bias_gradients_sqr = np.asarray([np.zeros(self.bias[i].shape) for i in range(len(self.bias))])

        self.step_count = 1

    def feed_forward(self, X):
        activations = [X]
        for i in range(len(self.weights)):
            excitation = activations[-1] @ self.weights[i] + self.bias[i]

            if i == len(self.weights) - 1:
                layer_activations = self.softmax(excitation)
            else:
                layer_activations = self.ReLu(excitation)

            activations.append(layer_activations)
        return activations

    def predict(self, X):
        return self.feed_forward(X)[-1]

    def predict_label(self, X):
        return np.argmax(self.predict(X))

    def backpropagation(self, inputs, labels):
        weight_gradients = np.empty_like(self.weights)
        bias_gradients = np.empty_like(self.bias)

        activations = self.feed_forward(inputs)
        errors = labels - activations[-1]
        # last layer of weights and biases
        # temp = self.ReLu_d(activations[-1]) * errors
        # temp = self.softmax_d(activations[-1]) @ errors.reshape(errors.size,)

        temp = errors
        weight_gradients[-1] = activations[-2].T.dot(temp)
        bias_gradients[-1] = np.sum(errors, axis=0) / len(errors)

        # previous layers
        deltas = errors
        for i in range(len(activations) - 2, 0, -1):
            deltas = self.ReLu_d(activations[i]) * deltas.dot(self.weights[i].T)
            weight_gradients[i - 1] = activations[i - 1].T.dot(deltas)
            bias_gradients[i - 1] = np.sum(deltas, axis=0) / deltas.shape[1]

        weight_gradients /= len(inputs)

        return weight_gradients, bias_gradients

    def train(self, trainX, trainY, valX, valY, num_epochs=20, batch_size=32, learning_rate=0.1, momentum=0.7):
        train_acc_list = []
        val_acc_list = []
        train_loss_list = []
        val_loss_list = []

        for i in range(num_epochs):
            for j in range(0, len(trainX), batch_size):
                inputs, labels = trainX[j:j + batch_size], trainY[j:j + batch_size]
                weight_gradients, bias_gradients = self.backpropagation(inputs, labels)

                # self.no_optimizer(0.2, weight_gradients, bias_gradients)
                # self.momentum_SGD(0.2, 0.5, weight_gradients, bias_gradients)
                # self.nesterov_momentum_NAG(0.2, 0.8, weight_gradients, bias_gradients)
                # self.AdaGrad(0.2, weight_gradients, bias_gradients)
                # self.AdaDelta(0.9, weight_gradients, bias_gradients)
                self.Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, weight_gradients=weight_gradients, bias_gradients=bias_gradients)
                self.step_count += 1

            predictionTrOneHot = self.feed_forward(trainX[0:10000])[-1]
            predictionValOneHot = self.feed_forward(valX)[-1]

            predictionTr = np.argmax(predictionTrOneHot, axis=1)
            predictionVal = np.argmax(predictionValOneHot, axis=1)

            train_acc = np.mean(predictionTr == np.argmax(trainY[0:10000], axis=1))
            val_acc = np.mean(predictionVal == np.argmax(valY, axis=1))

            train_loss = self.cross_entropy_loss(labels=trainY[0:10000], predictions=predictionTrOneHot)
            val_loss = self.cross_entropy_loss(labels=valY, predictions=predictionValOneHot)

            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.best_epoch = i

            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)

            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)

            print('epoch: ', i, 'train_acc:', train_acc, 'val_acc:', val_acc, 'train_loss:', "{:.4f}".format(train_loss),
                  'val_loss:', "{:.4f}".format(val_loss))
        return train_acc_list, val_acc_list, train_loss_list, val_loss_list

    def saveWeights(self):
        np.save('weights.npy', self.weights)  # save

    def loadWeights(self):
        self.weights = np.load('weights.npy')  # load

    def calc_accuracy(self, testX, testY):
        prediction = np.argmax(self.feed_forward(testX)[-1], axis=1)
        return np.mean(prediction == np.argmax(testY, axis=1))

    def evaluate(self, testX, testY):
        for i in range(1000, 1020):
            print(self.feed_forward(testX[i].reshape((1, 784)))[-1])
            prediction = np.argmax(self.feed_forward(testX[i].reshape((1, 784)))[-1], axis=1)
            print(np.argmax(testY[i]), prediction)
            plt.imshow(testX[i].reshape((28, 28)), cmap='gray')
            plt.text(-1, -1, 'prediction: ' + str(prediction[0]))
            plt.show()

    def confusionMatrix(self, testX, testY):
        m = np.zeros((10, 10), dtype=float)
        for i in range(len(testX)):
            prediction = np.argmax(self.feed_forward(testX[i].reshape((1, 784)))[-1], axis=1)[0]
            m[np.argmax(testY[i])][prediction] += 1

        for i in range(len(m)):
            m[i] /= np.sum(m[i])
        m *= 100

        return m

    def ReLu(self, values):
        return np.maximum(values, 0)

    # derivative of ReLU
    def ReLu_d(self, values):
        return values > 0

    def sigmoid(self, values):
        return 1 / (1 + math.e ** (-values))

    def sigmoid_d(self, values):
        return values * (1 - values)

    def softmax(self, values):
        e_x = np.exp(values.T - np.max(values, axis=-1))
        return (e_x / e_x.sum(axis=0)).T

    def softmax_d(self, values):
        SM = values.reshape((-1, 1))
        jac = np.diagflat(values) - np.dot(SM, SM.T)
        return jac

    def tanh_d(self, values):
        cosh = np.cosh(values)
        return 1 / (cosh * cosh)

    def dropout(self, drop_prob):
        temp_weights_all = []
        for w in self.weights:
            temp_weights = w.copy()
            indices = np.random.choice(temp_weights.shape[1] * temp_weights.shape[0], replace=False,
                                       size=int(temp_weights.shape[1] * temp_weights.shape[0] * drop_prob))
            temp_weights[np.unravel_index(indices, temp_weights.shape)] = 0
            temp_weights_all.append(temp_weights)
        return temp_weights_all

    def cross_entropy_loss(self, labels, predictions):
        return -np.sum(np.multiply(labels, np.log(predictions + 1e-20))) / len(labels)

    # --------------------------------------------Optimizers---------------------------------------------
    def no_optimizer(self, learning_rate, weight_gradients, bias_gradients):
        self.weights += learning_rate * weight_gradients
        self.bias += learning_rate * bias_gradients

    def momentum_SGD(self, learning_rate, momentum, weight_gradients, bias_gradients):
        weight_gradients *= learning_rate
        bias_gradients *= learning_rate

        weight_gradients += self.prev_weight_updates * momentum
        bias_gradients += self.prev_bias_updates * momentum

        self.weights += weight_gradients
        self.bias += bias_gradients

        self.prev_weight_updates = weight_gradients.copy()
        self.prev_bias_updates = bias_gradients.copy()

    def nesterov_momentum_NAG(self, learning_rate, momentum, weight_gradients, bias_gradients):
        weight_gradients *= learning_rate
        bias_gradients *= learning_rate

        weight_gradients += self.prev_weight_updates * momentum
        bias_gradients += self.prev_bias_updates * momentum

        self.weights += weight_gradients + weight_gradients * momentum
        self.bias += bias_gradients + bias_gradients * momentum

        self.prev_weight_updates = weight_gradients.copy()
        self.prev_bias_updates = bias_gradients.copy()

    def AdaGrad(self, learning_rate, weight_gradients, bias_gradients):
        epsilon = 1e-8
        self.accumulated_weight_gradients_sqr += weight_gradients ** 2
        self.accumulated_bias_gradients_sqr += bias_gradients ** 2

        # Sqrt in loop cuz its 3 dim arr and numpy has problems
        sqrt_weight_gradients = np.empty_like(self.accumulated_weight_gradients_sqr)
        sqrt_bias_gradients = np.empty_like(self.accumulated_bias_gradients_sqr)
        for i in range(len(self.accumulated_weight_gradients_sqr)):
            sqrt_weight_gradients[i] = np.sqrt(self.accumulated_weight_gradients_sqr[i] + epsilon)
            sqrt_bias_gradients[i] = np.sqrt(self.accumulated_bias_gradients_sqr[i] + epsilon)

        self.weights += learning_rate / sqrt_weight_gradients * weight_gradients
        self.bias += learning_rate / sqrt_bias_gradients * bias_gradients

    def RMSprop(self, learning_rate, beta, weight_gradients, bias_gradients):
        # suggested lr: 0.001, beta = 0.9

        epsilon = 1e-8
        self.accumulated_weight_gradients_sqr = beta * self.accumulated_weight_gradients_sqr + (1 - beta) * (weight_gradients ** 2)
        self.accumulated_bias_gradients_sqr = beta * self.accumulated_bias_gradients_sqr + (1 - beta) * (bias_gradients ** 2)

        # Sqrt in loop cuz its 3 dim arr and numpy has problems
        sqrt_weights = np.empty_like(self.accumulated_weight_gradients_sqr)
        sqrt_bias = np.empty_like(self.accumulated_bias_gradients_sqr)
        for i in range(len(self.accumulated_weight_gradients_sqr)):
            sqrt_weights[i] = np.sqrt(self.accumulated_weight_gradients_sqr[i] + epsilon)
            sqrt_bias[i] = np.sqrt(self.accumulated_bias_gradients_sqr[i] + epsilon)

        self.weights += learning_rate / sqrt_weights * weight_gradients
        self.bias += learning_rate / sqrt_bias * bias_gradients

    def AdaDelta(self, beta, weight_gradients, bias_gradients):
        # suggested lr: 0.001, beta = 0.9

        epsilon = 1e-8

        # numerator
        rms_weight_updates = np.empty_like(self.weights)
        rms_bias_updates = np.empty_like(self.bias)

        for i in range(len(self.accumulated_weight_gradients_sqr)):
            rms_weight_updates[i] = np.sqrt(self.accumulated_weight_updates[i] / self.step_count + epsilon)
            rms_bias_updates[i] = np.sqrt(self.accumulated_bias_updates[i] / self.step_count + epsilon)

        # denominator
        self.accumulated_weight_gradients_sqr = beta * self.accumulated_weight_gradients_sqr + (1 - beta) * (weight_gradients ** 2)
        self.accumulated_bias_gradients_sqr = beta * self.accumulated_bias_gradients_sqr + (1 - beta) * (bias_gradients ** 2)

        rms_weight_gradients = np.empty_like(self.accumulated_weight_gradients_sqr)
        rms_bias_gradients = np.empty_like(self.accumulated_bias_gradients_sqr)

        for i in range(len(self.accumulated_weight_gradients_sqr)):
            rms_weight_gradients[i] = np.sqrt(self.accumulated_weight_gradients_sqr[i] / self.step_count + epsilon)
            rms_bias_gradients[i] = np.sqrt(self.accumulated_bias_gradients_sqr[i] / self.step_count + epsilon)

        #  Update
        current_weight_update = rms_weight_updates / rms_weight_gradients * weight_gradients
        current_bias_update = rms_bias_updates / rms_bias_gradients * bias_gradients

        self.weights += current_weight_update
        self.bias += current_bias_update

        self.accumulated_weight_updates = beta * self.accumulated_weight_updates + (1 - beta) * (current_weight_update ** 2)
        self.accumulated_bias_updates = beta * self.accumulated_bias_updates + (1 - beta) * (current_bias_update ** 2)

    def Adam(self, learning_rate, beta1, beta2, weight_gradients, bias_gradients):
        epsilon = 1e-8

        self.accumulated_weight_gradients = beta1 * self.accumulated_weight_gradients + (1 - beta1) * weight_gradients
        self.accumulated_bias_gradients = (beta1 * self.accumulated_bias_gradients + (1 - beta1) * bias_gradients)

        self.accumulated_weight_gradients_sqr = beta2 * self.accumulated_weight_gradients_sqr + (1 - beta2) * weight_gradients ** 2
        self.accumulated_bias_gradients_sqr = beta2 * self.accumulated_bias_gradients_sqr + (1 - beta2) * bias_gradients ** 2

        # Correcting with B ** step_count
        corrected_acc_weight_grad = self.accumulated_weight_gradients / (1 - beta1 ** self.step_count)
        corrected_acc_bias_grad = self.accumulated_bias_gradients / (1 - beta1 ** self.step_count)

        corrected_acc_weight_grad_sqr = self.accumulated_weight_gradients_sqr / (1 - beta2 ** self.step_count)
        corrected_acc_bias_grad_sqr = self.accumulated_bias_gradients_sqr / (1 - beta2 ** self.step_count)

        # Sqrt in loop cuz its 3 dim arr and numpy has problems
        sqrt_weight_gradients_sqr = np.empty_like(corrected_acc_weight_grad_sqr)
        sqrt_bias_gradients_sqr = np.empty_like(corrected_acc_bias_grad_sqr)

        for i in range(len(self.accumulated_weight_gradients_sqr)):
            sqrt_weight_gradients_sqr[i] = np.sqrt(corrected_acc_weight_grad_sqr[i]) + epsilon
            sqrt_bias_gradients_sqr[i] = np.sqrt(corrected_acc_bias_grad_sqr[i]) + epsilon

        self.weights += learning_rate / sqrt_weight_gradients_sqr * corrected_acc_weight_grad
        self.bias += learning_rate / sqrt_bias_gradients_sqr * corrected_acc_bias_grad
