import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import time
import shap
import os

from tensorflow.keras.models import Sequential, load_model

def DatasetToParameter(dataSet: pd.DataFrame):
    X = dataSet.drop(["label"], axis=1)
    Y = dataSet["label"]
    return X, Y


def GetXY(data, manyToOne=False):
    X = data[:, :, :-3]
    Y = data[:, :, -3:]
    if manyToOne:
        Y = Y[:, 1, :]
    return X.reshape(X.shape[0], X.shape[1] * X.shape[2]), Y

# Model files and test sets to be trained
PathDict = {
    "0Label-ZHANG1": "./0Label-86.48/",
    "0Label-ZHANG0": "./0Label-45.27/",
    "ALLIn-ZHANG1": "./AllIn-79.44/",
    "0Label-ZHANG1-NoRLDD": "./0Label-78.14-NoRLDD/",
    "AllIn-LSTM": "./LSTM/Specified users/AllStandardScaler/positive/",
}

Path = PathDict["AllIn-LSTM"]
print(Path)
print(os.listdir(Path))

## tree model
# trainSet = pd.read_csv(os.path.join(Path, "train.csv"))
# testSet = pd.read_csv(os.path.join(Path, "test.csv"))
# model = joblib.load(os.path.join(Path, "RF_model.pkl"))

# X, Y = DatasetToParameter(testSet)
# print(model.score(X,Y))
# start = time.time()
# explainer = shap.explainers.Tree(model, approximate=True)
# # explainer = shap.explainers.GPUTree(model)
# shap_values = explainer.shap_values(X)
# print("use time:", time.time() - start)
# nowTime = time.strftime("%m%d%H%M%S", time.localtime())
# joblib.dump(shap_values, './shap_values{}.pkl'.format(nowTime))
# joblib.dump(explainer, './explainer{}.pkl'.format(nowTime))
# print("fileName:{}".format(nowTime))
# fig, ax = plt.subplots(constrained_layout=True)
# shap.summary_plot(shap_values[0], X)

# deep model
trainSet = np.load(os.path.join(Path, "trainSet.npy"))
testSet = np.load(os.path.join(Path, "testSet.npy"))
model = load_model(os.path.join(Path,"LSTMModeltest.h5"))

X, Y = GetXY(testSet)
print(X.shape, Y.shape)
explainer = shap.DeepExplainer(model, X[:20])
shap_values = explainer.shap_values(X[:5])
shap.summary_plot(shap_values[0], X)
