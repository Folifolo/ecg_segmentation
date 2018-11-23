import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from sklearn.model_selection import train_test_split
from dataset import load_dataset
import os
from os import listdir
from os.path import isfile, join, split
from ensemble.visualisation import ranging
import pickle as pkl
from metrics import statistics, F_score

def ensemble_predict(model_paths_list, x):
    model = load_model(model_paths_list[0])
    ens_predict = np.array(model.predict(x))

    for path in model_paths_list[1:]:
        model = load_model(path)
        predict = np.array(model.predict(x))
        ens_predict = predict + ens_predict

    ens_predict = ens_predict/len(model_paths_list)
    return ens_predict

def histogram(model_paths_list, x, y, win_len, threshold = 0.99):
    dict = {}
    for path in model_paths_list:
        _, filename = split(path)
        model_num = int(filename[len("ens_model_"):-3])
        dict[model_num] = 0
        model = load_model(path)
        predict = np.array(model.predict(x))
        for i in range(len(x)):
            pred = predict[i,win_len//2:5000-win_len//2,:]
            y_i = y[i,win_len//2:5000-win_len//2,:]
            stat = statistics(np.expand_dims(y_i, axis=0), np.expand_dims(pred, axis=0))
            F = F_score(stat)
            if F >=threshold:
                dict[model_num] += 1

    return dict

if __name__ == "__main__":
    win_len = 3072
    path_to_models = "C:\\ecg_segmentation\\ensemble\\trained_models"
    path_to_data = "C:\\ecg_segmentation\\ensemble\\data"

    model_paths_list = [join(path_to_models,f) for f in listdir(path_to_models) if isfile(join(path_to_models, f))]

    xy = load_dataset()

    X = xy["x"]
    Y = xy["y"]

    xtrain, xtest, ytrain, ytest= train_test_split(X, Y, test_size=0.33, random_state=42)

    dict = histogram(model_paths_list, xtrain, ytrain, win_len, threshold = 0.99)
    plt.bar(list(dict.keys()), dict.values(), color='g', alpha = 0.5)
    plt.savefig("hist.png")

    pred = ensemble_predict(model_paths_list, xtest)

    stat = statistics(ytest[:,win_len//2:5000-win_len//2,:], pred[:,win_len//2:5000-win_len//2,:])
    print(F_score(stat))
    stat.to_csv("stats_one_test.csv", sep = ';')