import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import EchoBot
import joblib
import time
import os

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

# choose Used Au
useAu = ['04', '06', '07', '10', '17', '25', '45']
# useAu = 'Full'

# choose Au Model
# useModel = 'C'
useModel = 'R'
# useModel = 'mix'

# choose pretreatmentMethod
# preMethod = None
preMethod = 'average'
# preMethod = 'highMapping'

# choose preParameter
preParameter = 150

# Hyper parameter
modelParameters = {
    'criterion': 'gini',
    'n_estimators': 5,
    'max_depth': 10,
    'min_samples_leaf': 50,
    'min_samples_split': 2,
    'random_state': 42
}

# choose Train&Test Set
useForTrainSet = [1,3,2,5]
useForTestSet = [4]

# Frame Open Set
# useFrameOpenMode = True
useFrameOpenMode = False

# Frame Open Parameter
frameOpenSet = [1, 2, 3, 4, 5]
splitRate = 0.8

# Send To Echo
useEchoBot = False
# useEchoBot = True

# Persistence
dumpModel = True
# dumpModel = False

# set Path
path = r"D:/UTA Real-Life Drowsiness Dataset AU Preprocessing/Group{0}/"
outPath = r"D:/UTA Real-Life Drowsiness Dataset AU Preprocessing/Train&Test Set/"
modelPath = r"D:/UTA Real-Life Drowsiness Dataset AU Preprocessing/model/"

# Full Au Parameter
Au_XX_R = [' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r', ' AU10_r', ' AU12_r', ' AU14_r',
           ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r', ' AU26_r', ' AU45_r']
Au_XX_C = [' AU01_c', ' AU02_c', ' AU04_c', ' AU05_c', ' AU06_c', ' AU07_c', ' AU09_c', ' AU10_c', ' AU12_c', ' AU14_c',
           ' AU15_c', ' AU17_c', ' AU20_c', ' AU23_c', ' AU25_c', ' AU26_c', ' AU28_c', ' AU45_c']


def PickUpParameter():
    cList = []
    rList = []
    template = ' AU{0}_{1}'

    if useAu == 'Full':
        cList = Au_XX_C
        rList = Au_XX_R
    else:
        for i in useAu:
            cList.append(template.format(i, 'c'))
            rList.append(template.format(i, 'r'))

    if useModel == 'C':
        cList.append('label')
        return cList
    elif useModel == 'R':
        rList.append('label')
        return rList
    elif useModel == 'mix':
        cList.append('label')
        return rList + cList


def concatDF(Set, newDataSet):
    if Set is None:
        Set = newDataSet
    else:
        Set = pd.concat([Set, newDataSet], ignore_index=True)
    return Set


def concatByGroup(GroupPath, AuParameter):
    dataSet = None

    folderList = [folder for folder in os.listdir(GroupPath) if os.path.isdir(os.path.join(GroupPath, folder))]
    for folder in folderList:
        samplePath = GroupPath + folder
        fileList = os.listdir(samplePath)
        for file in fileList:
            if os.path.splitext(file)[1] == '.csv' and "mix" in os.path.splitext(file)[0]:
                csvPath = samplePath + '/' + file
                data = pd.read_csv(csvPath)[AuParameter]
                # Use pretreatmentMethod
                if preMethod == 'average':
                    data = PreAverage(data)
                elif preMethod == 'highMapping':
                    data = PreHighMapping(data)

                dataSet = concatDF(dataSet, data)
                print('folder:{0},size:{1},SetSize:{2}'.format(csvPath, data.shape, dataSet.shape))
    print('GroupPath:{0},Total Size:{1}'.format(GroupPath, dataSet.shape))
    return dataSet


def PreAverage(originalData):
    averageGroup = []
    label = originalData['label']
    originalData = originalData.drop(['label'], axis=1)
    groupNum = originalData.shape[0] // preParameter

    for i in range(groupNum):
        first = i * preParameter
        last = (i + 1) * preParameter
        averageData = originalData[first:last].mean(axis=0)
        averageGroup.append(averageData)
    # Handle the remainder that is not divisible
    if originalData.shape[0] % preParameter != 0:
        averageGroup.append(originalData[groupNum * preParameter:].mean(axis=0))

    averageGroup = pd.DataFrame(averageGroup)
    label = label[:averageGroup.shape[0]]

    if useModel == 'C':
        averageGroup[averageGroup >= 0.5] = 1
        averageGroup[averageGroup < 0.5] = 0
    elif useModel == 'mix':
        if useAu == 'Full':
            dividingLine = len(Au_XX_R)
        else:
            dividingLine = len(useAu)

        rMergeData = averageGroup.iloc[:, 0:dividingLine]
        cMergeData = averageGroup.iloc[:, dividingLine:]
        cMergeData = cMergeData.where(cMergeData >= 0.5, 0)
        cMergeData = cMergeData.where(cMergeData < 0.5, 1)
        averageGroup = pd.concat([rMergeData, cMergeData], axis=1)
    averageGroup = pd.concat([averageGroup, label], axis=1)

    return averageGroup


def PreHighMapping(originalData):
    averageGroup = []
    label = originalData['label']
    originalDataValues = originalData.drop(['label'], axis=1).values
    originalColumns = originalData.drop(['label'], axis=1).columns.values.tolist()
    groupNum = originalData.shape[0] // preParameter

    for i in range(groupNum):
        first = i * preParameter
        last = (i + 1) * preParameter
        averageData = originalDataValues[first:last].flatten()
        averageGroup.append(averageData)
    averageGroup = np.array(averageGroup)

    label = label[:groupNum]
    averageGroup = np.column_stack((averageGroup, label))

    newColumns = []
    for i in range(preParameter):
        index = [j + '_' + str(i + 1) for j in originalColumns]
        newColumns.extend(index)
    newColumns.append('label')

    averageGroup = pd.DataFrame(averageGroup)
    averageGroup.columns = newColumns

    return averageGroup


def DatasetToParameter(dataSet: pd.DataFrame):
    X = dataSet.drop(['label'], axis=1)
    Y = dataSet['label']
    return X, Y


def Train(train, test):
    train_X, train_Y = DatasetToParameter(train)
    test_X, test_Y = DatasetToParameter(test)

    RFC = RandomForestClassifier(criterion=modelParameters.get('criterion'), max_depth=modelParameters.get('max_depth'),
                                 min_samples_leaf=modelParameters.get('min_samples_leaf'),
                                 min_samples_split=modelParameters.get('min_samples_split'),
                                 n_estimators=modelParameters.get('n_estimators'),
                                 random_state=modelParameters.get('random_state'))

    print('Start Training ' + useModel + ' Model')
    start = time.time()
    RFC.fit(train_X, train_Y)
    end = time.time()
    nowTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    template = '{0} \n' \
               'estimator: {1} \n' \
               'trainSet score: {2:.2%}\n' \
               'testSet score: {3:.2%}\n' \
               'use Time:{4} \n' \
               'Complete training'
    msg = template.format(nowTime, RFC, RFC.score(train_X, train_Y), RFC.score(test_X, test_Y), end - start)
    print(msg)
    if useEchoBot:
        EchoBot.SendMsgToTelegram(msg)
    DrawConfusionMatrix(RFC, test_X, test_Y, label=['0', '5', '10'])

    return RFC


def TrainByGVSearch(train, test):
    train_X, train_Y = DatasetToParameter(train)
    test_X, test_Y = DatasetToParameter(test)

    # Hyper parameter
    parameters = {
        'n_estimators': [5, 10, 17, 18, 30, 35, 60, 100, 200],  # 决策树的数量
        'max_depth': [3, 5, 8, 10, 20, 40, 50],  # 决策数的深度
        'random_state': [42],
        'min_samples_leaf': [2, 5, 10, 20, 50],  # 叶子节点最少的样本数
        'min_samples_split': [2, 5, 10, 20, 50]  # 每个划分最少的样本数
    }
    CLF = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parameters, cv=2, verbose=1, n_jobs=-1)
    print('Start Training ' + useModel + ' Model')
    start = time.time()
    CLF.fit(train_X, train_Y)
    end = time.time()
    best_clf = CLF.best_estimator_
    nowTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    template = '{0} \n' \
               'estimator: {1} \n' \
               'best parameters: {2}\n' \
               'trainSet score: {3:.2%}\n' \
               'testSet score: {4:.2%}\n' \
               'use Time:{5} \n' \
               'Complete training'
    msg = template.format(nowTime, best_clf, CLF.best_params_, best_clf.score(train_X, train_Y),
                          best_clf.score(test_X, test_Y), end - start)
    print(msg)
    if useEchoBot:
        EchoBot.SendMsgToTelegram(msg)
    DrawConfusionMatrix(best_clf, test_X, test_Y, label=['0', '5', '10'])

    return best_clf


def GeneralTTSet():
    groupList = []
    TrainSet = []
    TestSet = []
    parameter = PickUpParameter()
    # print(parameter)

    print("Generate Train&Test Set")
    if useFrameOpenMode:
        useGroup = frameOpenSet
        for i in tqdm(useGroup):
            groupSet = concatByGroup(path.format(i), parameter)
            groupList.append(groupSet)
        groupList = pd.concat(groupList, ignore_index=True)
        TrainSet, TestSet = train_test_split(groupList, train_size=splitRate, random_state=None, )
    else:
        useGroup = useForTrainSet + useForTestSet
        for i in tqdm(useGroup):
            groupSet = concatByGroup(path.format(i), parameter)
            if i in useForTrainSet:
                TrainSet.append(groupSet)
            if i in useForTestSet:
                TestSet.append(groupSet)
        TrainSet = pd.concat(TrainSet, ignore_index=True)
        TestSet = pd.concat(TestSet, ignore_index=True)

    return TrainSet, TestSet


def DrawConfusionMatrix(model, test_X, test_Y, label):
    Y_pred = model.predict(test_X)
    fig, ax = plt.subplots(constrained_layout=True)

    # Output matrix: rows are real values and columns are predicted values
    cm = confusion_matrix(test_Y, Y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_normalized, annot=True, cmap=plt.cm.binary, xticklabels=label, yticklabels=label,
                annot_kws={"fontsize": 12})

    fig.set_size_inches(w=7, h=3)

    ax.set_xlabel('True Label', fontsize=12)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('Predicted Label', fontsize=12)

    print('Confusion Matrix:')
    print(cm_normalized)
    if useEchoBot:
        EchoBot.SendPlotToTelegram(plt)
    plt.show()


def SaveModel(model):
    nowTime = time.strftime("%m%d%H%M%S", time.localtime())

    fileName = "{0}_{1}_{2}_{3}_{4}_{5}.pkl".format(nowTime, useModel, preMethod, preParameter, useForTrainSet,
                                                    useForTestSet)
    print("out model in:"+modelPath + fileName)
    joblib.dump(model, modelPath + fileName)


if __name__ == "__main__":
    trainSet, testSet = GeneralTTSet()

    # trainSet.to_csv(outPath + 'train.csv')
    # testSet.to_csv(outPath + 'test.csv')

    print('UseAu:{0}\n'
          'UseModel:{1}\n'
          'PreMethod:{2}\n'
          'PreParameter:{3}\n'
          'FrameOpenMode:{4}'.format(useAu, useModel, preMethod, preParameter, useFrameOpenMode))
    if useFrameOpenMode:
        print('trainSet:{0},testSet:{1}'.format(trainSet.shape, testSet.shape))
    else:
        print('trainSet{0}:{1},testSet{2}:{3}'.format(useForTrainSet, trainSet.shape, useForTestSet, testSet.shape))

    # Training
    trainModel = Train(trainSet, testSet)
    # trainModel = TrainByGVSearch(trainSet, testSet)
    if dumpModel:
        SaveModel(trainModel)
