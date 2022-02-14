import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import time
import shap
import os
from sklearn.inspection import permutation_importance
import eli5
from eli5.sklearn import PermutationImportance
from tensorflow.keras.models import Sequential, load_model
from sklearn.metrics import accuracy_score, confusion_matrix

Au_XX_R = [' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r', ' AU10_r', ' AU12_r', ' AU14_r',
           ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r', ' AU26_r', ' AU45_r']
def GetXY(data, manyToOne=False):
    X = data[:, :, :-3]
    Y = data[:, :, -3:]
    if manyToOne:
        Y = Y[:, 1, :]
    return X, Y

def DatasetToParameter(dataSet: pd.DataFrame):
    X = dataSet.drop(["label"], axis=1)
    Y = dataSet["label"]
    return X, Y

# The model files that need to be trained, the test set, the shap_values and explainer calculated by SHAP(can get from GetShap.py).
# If use Permutation Importance that need the model files, the test set
PathDict = {
    "0Label-ZHANG1": "./RF/zhang/0Label-86.48/",
    "0Label-ZHANG0": "./RF/zhang/0Label-45.27/",
    "ALLIn-ZHANG1": "./RF/zhang/AllIn-79.44/",
    "0Label-ZHANG1-NoRLDD": "./RF/Specified users/0Label-78.14-NoRLDD/",
    "Each-positive": "./RF/Specified users/EachStandardScaler/positive/",
    "ALLIN-RF": "./RF/Specified users/AllStandardScaler/positive/",
    "Each-LSTM":"./LSTM/UnSpecified users/EachStandardScaler/4"
}

Path = PathDict["Each-LSTM"]
print(os.listdir(Path))

## tree model
# trainSet = pd.read_pickle(os.path.join(Path, "train.pkl"))
# testSet = pd.read_csv(os.path.join(Path, "test.csv"))
# model = joblib.load(os.path.join(Path, "RF_model.pkl"))
# X, Y = DatasetToParameter(testSet)
# X_value = X.values
# Y_value = Y.values
# print(model.score(X,Y))

## deep model
testSet = np.load(os.path.join(Path, "testSet.npy"))
model = load_model(os.path.join(Path,"LSTMModel.h5"))
testSet = testSet[4,:,:]
print(testSet.shape)
X, Y = GetXY(testSet,manyToOne=True)
print(X.shape)
Y_P = np.argmax(Y, axis=-1)
pree = model.predict(X)
pree = np.argmax(pree, axis=-1)
print(accuracy_score(Y_P, pree))

## output shap plt
# start = time.time()
# explainer = joblib.load(os.path.join(Path, "explainer.pkl"))
# shap_values = joblib.load(os.path.join(Path, "shap_values.pkl"))
# print("Load time:", time.time() - start)
# fig, ax = plt.subplots(constrained_layout=True)
# shap.summary_plot(shap_values[2], X)

## Permutation Importance(only for tree model)
# perm_importance = permutation_importance(model, X_value, Y_value)
# sorted_idx = perm_importance.importances_mean.argsort()
# plt.barh(X.columns.values[sorted_idx], perm_importance.importances_mean[sorted_idx])
# plt.xlabel("Permutation Importance")
# plt.show()

