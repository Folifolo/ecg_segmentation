import os
from os import listdir
from os.path import isfile, join
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.model_selection import train_test_split

from dataset import load_dataset
from metrics import statistics, F_score


def draw_one(model_path, x, y, pacient, win_len):
    offsets = (5000 - win_len)//2
    model = load_model(model_path)
    X = np.expand_dims(x[pacient, :, :], axis=0)
    Y = np.expand_dims(y[pacient,offsets:5000 - offsets,:], axis=0)

    prediction = np.array(model.predict(X))
    prediction = prediction[:,offsets:5000-offsets,:]

    x_axis = np.arange(offsets/500, (win_len +offsets)/500, 1/500)
    plt.figure(figsize=(20, 5))
    plt.plot(x_axis, x[pacient, offsets:5000 - offsets, 0], 'k')
    i = 0
    predict_rounded = np.argmax(prediction, axis=2)[i]
    one_hot = np.zeros((predict_rounded.size, predict_rounded.max()+1))
    one_hot[np.arange(predict_rounded.size), predict_rounded] = 1

    plt.fill_between(x_axis, Y[i, :win_len, 1]*40 + -50, -50, color='r', alpha=0.3)
    plt.fill_between(x_axis, Y[i, :win_len, 2]*40 + -50, -50, color='g', alpha=0.3)
    plt.fill_between(x_axis, Y[i, :win_len, 0]*40 + -50, -50, color='b', alpha=0.3)
    plt.fill_between(x_axis, list(one_hot[:win_len, 1]*40), 0, color='r', alpha=0.3)
    plt.fill_between(x_axis, list(one_hot[:win_len, 2]*40), 0, color='g', alpha=0.3)
    plt.fill_between(x_axis, list(one_hot[:win_len, 0]*40), 0, color='b', alpha=0.3)


    stat = statistics(Y, prediction)
    F = F_score(stat)
    print(stat)
    print(F)
    plt.show()

def draw_all(model_path, x, y, win_len, model2=None):
    offsets = (5000 - win_len)//2
    model = load_model(model_path)
    X = x
    Y = y[:,offsets:5000 - offsets,:]

    prediction = np.array(model.predict(X))
    prediction = prediction[:,offsets:5000-offsets,:]
    if model2 != None:
        model2 =  load_model(model2)
        prediction2 = np.array(model2.predict(X))[:,offsets:5000-offsets,:]

    x_axis = np.arange(offsets/500, (win_len +offsets)/500, 1/500)
    for i in range(len(X)):
        plt.figure(figsize=(20, 5))
        plt.plot(x_axis, x[i, offsets:5000 - offsets, 0], 'k')
        predict_rounded = np.argmax(prediction, axis=2)[i]
        one_hot = np.zeros((predict_rounded.size, predict_rounded.max()+1))
        one_hot[np.arange(predict_rounded.size), predict_rounded] = 1

        plt.fill_between(x_axis, Y[i, :win_len, 1]*40 + -50, -50, color='r', alpha=0.3)
        plt.fill_between(x_axis, Y[i, :win_len, 2]*40 + -50, -50, color='g', alpha=0.3)
        plt.fill_between(x_axis, Y[i, :win_len, 0]*40 + -50, -50, color='b', alpha=0.3)
        plt.fill_between(x_axis, list(one_hot[:win_len, 1]*40), 0, color='r', alpha=0.3)
        plt.fill_between(x_axis, list(one_hot[:win_len, 2]*40), 0, color='g', alpha=0.3)
        plt.fill_between(x_axis, list(one_hot[:win_len, 0]*40), 0, color='b', alpha=0.3)

        if model2 != None:
            predict_rounded = np.argmax(prediction2, axis=2)[i]
            one_hot = np.zeros((predict_rounded.size, predict_rounded.max()+1))
            one_hot[np.arange(predict_rounded.size), predict_rounded] = 1
            plt.fill_between(x_axis, list(one_hot[:win_len, 1]*40+50), 50, color='r', alpha=0.3)
            plt.fill_between(x_axis, list(one_hot[:win_len, 2]*40+50), 50, color='g', alpha=0.3)
            plt.fill_between(x_axis, list(one_hot[:win_len, 0]*40+50), 50, color='b', alpha=0.3)

        stat = statistics(Y, prediction)
        F = F_score(stat)
        print(stat)
        print(F)
        plt.savefig("ill"+str(i)+".png")
        plt.clf()

def ranging(model_path, x, y, win_len, col= "k", is_path = True):
    offsets = (5000 - win_len)//2
    Y = y[:,offsets:5000 - offsets,:]

    if is_path:
        model = load_model(model_path)
        prediction = np.array(model.predict(x))
    else:
        prediction = model_path
    prediction = prediction[:,offsets:5000-offsets,:]

    dict = {}
    for i in range(len(x)):
        prediction_i = prediction[i,:,:]
        y_i = Y[i,:,:]
        stat = statistics(np.expand_dims(y_i, axis=0), np.expand_dims(prediction_i, axis=0))
        F = F_score(stat)
        dict[i] = F

    dict = sorted(dict.items())
    x, y_i = zip(*dict)
    plt.scatter(x, y_i, c=col, alpha=0.3)
    return y_i

def split_data(xy1, xy2):
    x1 = xy1["x"]
    y1 = xy1["y"]
    x2 = xy2["x"]
    for i in range(len(x2)):
        for j in range(len(x1)):
            if np.array_equal(x1[j,:,:],x2[i,:,:]):
                x1 = np.delete(x1, j, 0)
                y1 = np.delete(y1, j, 0)
                break
    return x1, y1

if __name__ == "__main__":
    win_len = 3072
    leads = 12
    path_to_models = "C:\\ecg_segmentation\\ensemble\\trained_models"
    model_paths_list = [join(path_to_models,f) for f in listdir(path_to_models) if isfile(join(path_to_models, f))]

    xy = load_dataset()
    X = xy["x"]
    Y = xy["y"]
    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.33, random_state=42)

    for model_path in model_paths_list:
        ranging(model_path, xtrain, ytrain, win_len, col = np.random.rand(1,3))