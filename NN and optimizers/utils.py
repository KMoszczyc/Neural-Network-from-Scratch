import numpy as np
import gzip
import sys
import pickle
import random

np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)


class Utils:
    def loadData(path='mnist.pkl.gz'):
        f = gzip.open(path, 'rb')
        if sys.version_info < (3,):
            data = pickle.load(f)
        else:
            data = pickle.load(f, encoding='bytes')
        f.close()

        (x_train, temp_y_train), (x_test, temp_y_test) = data
        y_train = np.zeros((len(temp_y_train), 10))
        y_test = np.zeros((len(temp_y_test), 10))

        for i in range(len(temp_y_train)):
            y_train[i][temp_y_train[i]] = 1
        for i in range(len(temp_y_test)):
            y_test[i][temp_y_test[i]] = 1

        x_tr_temp = (x_train.reshape(60000, 784) / 255)

        # shuffleData(x_tr_temp, y_train)

        x_tr = x_tr_temp[0:50000]
        y_tr = y_train[0:50000]

        x_val = x_tr_temp[50000:60000]
        y_val = y_train[50000:60000]

        x_ts = x_test.reshape(10000, 784) / 255

        return x_tr, y_tr, x_val, y_val, x_ts, y_test

    def loadData28x28(a=""):
        x_tr, y_tr, x_val, y_val, x_test, y_test = Utils.loadData()
        return x_tr.reshape(x_tr.shape[0], 28, 28), y_tr, x_val.reshape(x_val.shape[0], 28, 28), y_val, x_test.reshape(x_test.shape[0], 28, 28), y_test

    def shuffleData(X, Y):
        for i in range(len(X)):
            tempX = X[i]
            tempY = Y[i]
            index = random.randint(0, len(X) - 1)
            X[index] = tempX
            Y[index] = tempY

    def listToLateX(list):
        s = ""
        for elem in list:
            s += " & " + str(elem)
        return s

    def twoDimArrToLateX(arr):
        s = ""
        for i in range(len(arr)):
            for j in range(arr.shape[1]):
                s += " & " + str(round(arr[i][j], 2))
            s += "\n"
        return s

    def avgListsByBin(list_of_lists):
        avg_list = np.empty_like(list_of_lists[0])
        print(avg_list)
        for i in range(len(list_of_lists)):
            avg_list += np.array(list_of_lists[i])

        return avg_list / len(list_of_lists)


a = [[1, 1], [8, 4], [6, 2]]
print(Utils.avgListsByBin(a))
