import pandas as pd
import numpy as np
import time
import os

from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# choose Used Au
# useAu = ['04', '06', '07', '10', '17', '25', '45']
useAu = 'Full'

# choose Au Model
# useModel = 'C'
useModel = 'R'
# useModel = 'mix'

# choose pretreatmentMethod
preMethod = None
# preMethod = 'average'
# preMethod = 'highMapping'

# choose preParameter
preParameter = 5

# set Path
path = r"D:/UTA Real-Life Drowsiness Dataset AU Preprocessing/Group{0}/"
outPath = r"D:/UTA Real-Life Drowsiness Dataset AU Preprocessing/Train&Test Set/"

# choose Train&Test Set
useForTrainSet = [1,2,3,5]
useForTestSet = [4]

# Full Parameter
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

    RFC = RandomForestClassifier(criterion='gini', max_depth=40, min_samples_leaf=2, min_samples_split=2,
                                 n_estimators=200, random_state=42)
    print('Start Training ' + useModel + ' Model')
    start = time.time()
    RFC.fit(train_X, train_Y)
    end = time.time()
    print('trainSet score: {:.2%}'.format(RFC.score(train_X, train_Y)))
    print('testSet score: {:.2%}'.format(RFC.score(test_X, test_Y)))
    print('use Time:{0}'.format(end - start))
    print('Complete training')

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
    print("best parameters:")
    print(CLF.best_params_)
    print('trainSet score: {:.2%}'.format(best_clf.score(train_X, train_Y)))
    print('testSet score: {:.2%}'.format(best_clf.score(test_X, test_Y)))
    print('use Time:{0}'.format(end - start))
    print('Complete training')

    return CLF.best_estimator_


if __name__ == "__main__":
    groupSet = None
    groupList = []
    useGroup = useForTrainSet + useForTestSet

    trainSet = []
    testSet = []

    parameter = PickUpParameter()
    # print(parameter)

    print("Generate Train&Test Set")
    for i in tqdm(useGroup):
        groupSet = concatByGroup(path.format(i), parameter)
        if i in useForTrainSet:
            trainSet.append(groupSet)
        if i in useForTestSet:
            testSet.append(groupSet)

    trainSet = pd.concat(trainSet, ignore_index=True)
    testSet = pd.concat(testSet, ignore_index=True)

    trainSet.to_csv(outPath + 'train.csv')
    testSet.to_csv(outPath + 'test.csv')
    print('UseAu:{0},UseModel:{1},PreMethod:{2},PreParameter:{3}'.format(useAu, useModel, preMethod, preParameter))
    print('trainSet{0}:{1},testSet{2}:{3}'.format(useForTrainSet, trainSet.shape, useForTestSet, testSet.shape))

    # Training
    # trainModel = Train(trainSet, testSet)
    trainModel = TrainByGVSearch(trainSet, testSet)
    print(trainModel)
