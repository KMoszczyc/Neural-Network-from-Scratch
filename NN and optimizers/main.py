import numpy as np
import matplotlib.pyplot as plt
import time
from NeuralNetwork import NeuralNetwork
from utils import Utils
import random

trainX, trainY, valX, valY, testX, testY = Utils.loadData()

print(trainX.shape, trainY.shape)
print(valX.shape, valY.shape)


def optimize_params():
    s = ""

    learning_rates = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    momentums = [0.1, 0.3, 0.5, 0.7, 0.9]
    batch_sizes = [8, 16, 24, 32, 48, 64, 96, 128]
    weight_ranges = [0.01, 0.05, 0.1, 0.2, 0.4, 0.7, 1]
    neurons = [256, 128, 64, 32]

    train_acc_list = []
    val_acc_list = []
    train_loss_list = []
    val_loss_list = []

    train_acc_s = 0
    val_acc_s = 0
    test_acc_s = 0
    val_epoch_s = 0
    iterations = 1
    avg_times = [0] * iterations
    for i in range(iterations):
        times = []
        index = 0
        # for lr in learning_rates:
        nn = NeuralNetwork(784, 128, 10, weights_range=0.01, weight_init_method="He")

        start_time = time.time()
        train_acc, val_acc, train_loss, val_loss = nn.train(trainX, trainY, valX, valY, num_epochs=10, batch_size=32, learning_rate=0.2, momentum=0.7)
        time_elapsed = time.time() - start_time
        avg_times[index] += time_elapsed
        index += 1
        test_acc = nn.calc_accuracy(testX, testY)

        print('time elapsed: ', time_elapsed)
        print('best val accuracy: ', nn.best_val_accuracy, ' epoch: ', nn.best_epoch)
        print('test accuracy: ', test_acc)

        print('ReLu')
        print('train accuracy', train_acc)
        print('val accuracy', val_acc)

        # text_file.write("\n" + str(wr) + " train accuracy: " + listToLateX(train_acc))
        # text_file.write("\n" + str(wr) + " val accuracy: " + listToLateX(val_acc))

        train_acc_s += train_acc[-1]
        val_acc_s += nn.best_val_accuracy
        test_acc_s += test_acc
        val_epoch_s += nn.best_epoch

        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        # nn.saveWeights()

    avg_train_acc = Utils.avgListsByBin(train_acc_list)
    avg_val_acc = Utils.avgListsByBin(val_acc_list)
    avg_train_loss = Utils.avgListsByBin(train_loss_list)
    avg_val_loss = Utils.avgListsByBin(val_loss_list)

    # accuracy plot
    plt.plot(avg_train_acc, label='train accuracy')
    plt.plot(avg_val_acc, label='validate accuracy')
    plt.xlabel('steps')
    plt.ylabel('accuracy')
    plt.legend(loc="upper left")
    plt.xticks(range(0, len(train_acc)))
    # plt.yticks(np.arange(round(min(x.min(), y.min()), 2), 1, 0.01))
    plt.show()

    # loss plot
    plt.plot(avg_train_loss, label='train  loss')
    plt.plot(avg_val_loss, label='validate loss')
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.legend(loc="upper left")
    plt.xticks(range(0, len(train_acc)))
    plt.show()


def confusion_matrix():
    nn = NeuralNetwork(784, 256, 64, 10)
    nn.loadWeights()

    confusionMatrix = nn.confusionMatrix(testX, testY)
    np.set_printoptions(precision=2, suppress=True)
    print(confusionMatrix)
    print(Utils.twoDimArrToLateX(confusionMatrix))
    nn.evaluate(testX, testY)


def predict_image(X, y, nn):
    img = X.reshape(28, 28)
    prediction = nn.predict_label(X)
    correct_label = np.argmax(y)
    print(f'prediction: {prediction} \t real: {correct_label}')

    plt.imshow(img)
    plt.text(-1, -1.5, f'prediction: {prediction}     real: {correct_label}', fontsize=20)
    plt.show()


def predict_random_images(nn):
    for i in range(100):
        img_id = random.randint(0, len(testX))
        predict_image(testX[img_id], testY[img_id], nn)


if __name__ == "__main__":
    nn = NeuralNetwork(784, 256, 64, 10)
    nn.loadWeights()

    predict_random_images(nn)
