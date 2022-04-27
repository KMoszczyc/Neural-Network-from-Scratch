import math
import numpy as np
import matplotlib.pyplot as plt
import cv2

from utils import Utils


class CNN:
    def __init__(self, num_filters=32, f_size=3, layers=(128, 10)):
        self.f_size = f_size
        self.num_filters = num_filters
        self.filters = np.random.randn(num_filters, f_size, f_size) * np.sqrt(2 / (num_filters * f_size * f_size))
        self.filters_bias = np.random.randn(num_filters, f_size, f_size) * np.sqrt(2 / (num_filters * f_size * f_size))

        self.weights = []
        self.bias = []

        self.conv_output_shape = self.filters.shape
        self.max_pooling_output_shape = self.filters.shape

        self.cache_max_pooling_input = np.array((28, 28, 32))
        self.cache_conv_input = np.array((28, 28, 32))

        layers = (int(num_filters * 28 * 28 / 4),) + layers
        print(layers)
        self.bias.append(np.random.randn(layers[0]) * np.sqrt(2 / layers[0]))
        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i], layers[i + 1]) * np.sqrt(2 / (layers[i])))
            self.bias.append(np.random.randn(layers[i + 1]) * np.sqrt(2 / layers[i + 1]))

        print(self.bias[0].shape)

        # Adam for fully connected
        self.accumulated_weight_gradients = np.asarray([np.zeros(self.weights[i].shape) for i in range(len(self.weights))])
        self.accumulated_bias_gradients = np.asarray([np.zeros(self.bias[i].shape) for i in range(len(self.bias))])

        self.accumulated_weight_gradients_sqr = np.asarray([np.zeros(self.weights[i].shape) for i in range(len(self.weights))])
        self.accumulated_bias_gradients_sqr = np.asarray([np.zeros(self.bias[i].shape) for i in range(len(self.bias))])

        # Adam for convolutions
        self.accumulated_filters_gradients = np.asarray([np.zeros(self.filters[i].shape) for i in range(len(self.filters))])
        self.accumulated_filters_bias_gradients = np.asarray([np.zeros(self.filters_bias[i].shape) for i in range(len(self.filters_bias))])

        self.accumulated_filters_gradients_sqr = np.asarray([np.zeros(self.filters[i].shape) for i in range(len(self.filters))])
        self.accumulated_filters_bias_gradients_sqr = np.asarray([np.zeros(self.filters_bias[i].shape) for i in range(len(self.filters_bias))])

        self.step_count = 1

    def conv2d_forward(self, X):
        h, w = X.shape
        input_padded = np.zeros((h + 2, w + 2))
        input_padded[1:-1, 1:-1] = X

        output = np.zeros((h, w, self.num_filters))

        for i in range(h - self.f_size + 1):
            for j in range(w - self.f_size + 1):
                x_cut = input_padded[i:(i + self.f_size), j:(j + self.f_size)]
                output[i, j] = np.sum(x_cut * self.filters, axis=(1, 2))
        return output

    def max_pooling(self, X):
        h, w, num_filters = X.shape
        output = np.zeros((h // 2, w // 2, num_filters))
        new_h = h // 2
        new_w = w // 2
        for i in range(new_h):
            for j in range(new_w):
                x_cut = X[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                output[i, j] = np.amax(x_cut, axis=(0, 1))

        return output

    def fully_connected_forward(self, X):
        activations = [X]
        for i in range(len(self.weights)):
            excitation = activations[-1] @ self.weights[i]

            if i == len(self.weights) - 1:
                layer_activations = self.softmax(excitation)
            else:
                layer_activations = self.ReLu(excitation)

            activations.append(layer_activations)
        return activations

    def feed_forward_all(self, X):
        self.cache_max_pooling_input = self.conv2d_forward(X)
        # print(self.conv2d_forward(X))
        self.conv_output_shape = self.cache_max_pooling_input.shape
        output = self.max_pooling(self.cache_max_pooling_input)

        self.max_pooling_output_shape = output.shape
        output = output.flatten()
        output = self.fully_connected_forward(output)
        return output

    def max_pooling_backprop(self, gradients):
        gradients = gradients.reshape(self.max_pooling_output_shape)
        h, w, num_filters = self.conv_output_shape
        max_pool_grad = np.zeros(self.conv_output_shape)

        new_h = h // 2
        new_w = w // 2
        for i in range(new_h):
            for j in range(new_w):

                x_cut = self.cache_max_pooling_input[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                h, w, f = x_cut.shape
                amax = np.amax(x_cut, axis=(0, 1))

                for i2 in range(h):
                    for j2 in range(w):
                        for f2 in range(f):
                            if x_cut[i2, j2, f2] == amax[f2]:
                                max_pool_grad[i * 2 + i2, j * 2 + j2, f2] = gradients[i, j, f2]

        return max_pool_grad

    def conv2d_backprop(self, max_pool_gradients):
        h, w, _ = max_pool_gradients.shape
        filter_gradients = np.zeros(self.filters.shape)
        filter_bias_gradients = np.zeros(self.filters_bias.shape)

        for i in range(h - self.f_size + 1):
            for j in range(w - self.f_size + 1):
                x_cut = max_pool_gradients[i:(i + self.f_size), j:(j + self.f_size)]
                for f in range(self.num_filters):
                    filter_gradients[f] += max_pool_gradients[i, j, f] * x_cut[:, :, f]
                    filter_bias_gradients[f] += x_cut[:, :, f]

        return filter_gradients, filter_bias_gradients

    def backpropagation_FC(self, inputs, labels):
        weight_gradients = np.empty_like(self.weights)
        bias_gradients = np.empty_like(self.bias)

        activations = self.feed_forward_all(inputs)
        errors = (labels - activations[-1]).reshape(1, 10)
        # last layer of weights and biases

        temp = errors
        # print(activations[-1], labels)
        weight_gradients[-1] = activations[-2].reshape(activations[-2].shape[0], -1).dot(temp)

        bias_gradients[-1] = np.sum(errors, axis=0) / len(errors)

        # previous layers
        deltas = errors

        for i in range(len(activations) - 2, 0, -1):
            deltas = self.ReLu_d(activations[i]) * deltas.dot(self.weights[i].T)
            weight_gradients[i - 1] = activations[i - 1].reshape(activations[i - 1].shape[0], -1).dot(deltas)
            bias_gradients[i] = np.sum(deltas, axis=0) / deltas.shape[1]

        deltas = self.ReLu_d(activations[0]) * deltas.dot(self.weights[0].T)
        bias_gradients[0] = np.sum(deltas, axis=0) / deltas.shape[1]

        weight_gradients /= len(inputs)
        return weight_gradients, bias_gradients

    def train(self, trainX, trainY, valX, valY, num_epochs=1, batch_size=32, learning_rate=0.1, momentum=0.7):
        train_acc_list = []
        val_acc_list = []
        for i in range(num_epochs):
            for j in range(0, 5001):
                weight_gradients, bias_gradients = self.backpropagation_FC(trainX[j], trainY[j])

                # self.weights += learning_rate * weight_gradients
                # self.bias += learning_rate * bias_gradients
                self.Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, weight_gradients=weight_gradients, bias_gradients=bias_gradients)

                max_pool_grads = self.max_pooling_backprop(bias_gradients[0])
                filters_grads, filters_bias_grads = self.conv2d_backprop(max_pool_grads)
                self.AdamForConv(learning_rate=0.001, beta1=0.9, beta2=0.999, filter_gradients=filters_grads, filter_bias_gradients=filters_bias_grads)

                # self.filters += learning_rate * filters_grads
                # self.filters_bias += learning_rate * filters_bias_grads

                if j % 100 == 0 and j > 0:
                    train_acc, train_loss = self.evaluate(trainX, trainY, j)
                    val_acc, val_loss = self.evaluate(valX, valY, 2000)
                    train_acc_list.append(train_acc)
                    val_acc_list.append(val_acc)
                    print('step: ', j, 'train_acc: ', train_acc, 'train_loss: ', train_loss, 'val_acc: ', val_acc, 'val_loss: ', val_loss)
        return train_acc_list, val_acc_list

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

    def evaluate(self, X, Y, num=1000):
        predictions = np.zeros((num, 10))

        for i in range(num):
            predictions[i] = self.feed_forward_all(X[i])[-1]

        predictionsArgMax = np.argmax(predictions, axis=1)
        acc = np.mean(predictionsArgMax == np.argmax(Y[0:num], axis=1))
        loss = self.cross_entropy_loss(Y[0:num], predictions)

        return acc, loss

    def cross_entropy_loss(self, labels, predictions):
        return -np.sum(np.multiply(labels, np.log(predictions + 1e-20))) / len(labels)

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

        for i in range(len(self.accumulated_bias_gradients_sqr)):
            sqrt_bias_gradients_sqr[i] = np.sqrt(corrected_acc_bias_grad_sqr[i]) + epsilon

        self.weights += learning_rate / sqrt_weight_gradients_sqr * corrected_acc_weight_grad
        self.bias += learning_rate / sqrt_bias_gradients_sqr * corrected_acc_bias_grad

    def AdamForConv(self, learning_rate, beta1, beta2, filter_gradients, filter_bias_gradients):
        epsilon = 1e-8

        self.accumulated_filters_gradients = beta1 * self.accumulated_filters_gradients + (1 - beta1) * filter_gradients
        self.accumulated_filters_bias_gradients = (beta1 * self.accumulated_filters_bias_gradients + (1 - beta1) * filter_bias_gradients)

        self.accumulated_filters_gradients_sqr = beta2 * self.accumulated_filters_gradients_sqr + (1 - beta2) * filter_gradients ** 2
        self.accumulated_filters_bias_gradients_sqr = beta2 * self.accumulated_filters_bias_gradients_sqr + (1 - beta2) * filter_bias_gradients ** 2

        # Correcting with B ** step_count
        corrected_acc_weight_grad = self.accumulated_filters_gradients / (1 - beta1 ** self.step_count)
        corrected_acc_bias_grad = self.accumulated_filters_bias_gradients / (1 - beta1 ** self.step_count)

        corrected_acc_weight_grad_sqr = self.accumulated_filters_gradients_sqr / (1 - beta2 ** self.step_count)
        corrected_acc_bias_grad_sqr = self.accumulated_filters_bias_gradients_sqr / (1 - beta2 ** self.step_count)

        # Sqrt in loop cuz its 3 dim arr and numpy has problems
        sqrt_weight_gradients_sqr = np.empty_like(corrected_acc_weight_grad_sqr)
        sqrt_bias_gradients_sqr = np.empty_like(corrected_acc_bias_grad_sqr)

        for i in range(len(self.accumulated_filters_gradients_sqr)):
            sqrt_weight_gradients_sqr[i] = np.sqrt(corrected_acc_weight_grad_sqr[i]) + epsilon

        for i in range(len(self.accumulated_filters_bias_gradients_sqr)):
            sqrt_bias_gradients_sqr[i] = np.sqrt(corrected_acc_bias_grad_sqr[i]) + epsilon

        self.filters += learning_rate / sqrt_weight_gradients_sqr * corrected_acc_weight_grad
        self.filters_bias += learning_rate / sqrt_bias_gradients_sqr * corrected_acc_bias_grad


trainX, trainY, valX, valY, testX, testY = Utils.loadData28x28()

cnn = CNN(num_filters=8, f_size=2, layers=(128, 10))

prediction = cnn.feed_forward_all(trainX[0])
print(trainY.shape)
print(prediction[-1])
train_acc_list, val_acc_list = cnn.train(trainX, trainY, valX, valY)

# train_acc, train_loss = cnn.evaluate(trainX, trainY, 5000)
# val_acc, val_loss = cnn.evaluate(valX, valY, 5000)
# print('step: ', 5000, 'train_acc: ', train_acc, 'train_loss: ', train_loss, 'val_acc: ', val_acc, 'val_loss: ', val_loss)
# print('&',train_acc)
# print('&',train_loss)
# print('&',val_acc)
# print('&',val_loss)


plt.plot(train_acc_list, label='train accuracy')
plt.plot(val_acc_list, label='validate accuracy')
plt.xlabel('steps')
plt.ylabel('accuracy')
plt.legend(loc="upper left")
plt.xticks(range(0, len(train_acc_list)))
# plt.yticks(np.arange(round(min(x.min(), y.min()), 2), 1, 0.01))
plt.show()


# plt.imshow(trainX[0].reshape(28,28), cmap='gray')
# plt.show()
#
# output = cnn.conv2d_forward(trainX[0].reshape(28,28))
#
# print(output.shape)
# plt.imshow(output[:,:,0], cmap='gray')
# plt.show()
#
# output = cnn.max_pooling(output)
#
# print(output.shape)
# plt.imshow(output[:,:,0], cmap='gray')
# plt.show()
#
# output = output.flatten()
# output = cnn.fully_connected_forward(output)
#
# print(output[-1])


def show_conv_result():
    img_cat = plt.imread('cat.png')
    plt.imshow(img_cat)
    plt.show()
    gray_cat = cv2.cvtColor(img_cat, cv2.COLOR_RGB2GRAY)
    plt.imshow(gray_cat, cmap='gray')
    plt.show()

    output = cnn.conv2d_forward(trainX[5])
    output = cnn.max_pooling(output)
    print(output.shape)  # (26, 26, 8)
    plt.imshow(trainX[5], cmap='gray')
    plt.show()

    for i in range(output.shape[2]):
        plt.imshow(output[:, :, i], cmap='gray')
        plt.show()


show_conv_result()
