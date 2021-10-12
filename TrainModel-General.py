import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import EchoBot
import joblib
import time
import os

from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from pandas.testing import assert_frame_equal
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

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
# preMethod = 'highMappingV2'
# preMethod = 'max'
# preMethod = 'maxAverage'
# preMethod = 'maxAverageMin'

# choose preParameter
preParameter = 150

# Hyper parameter
modelParameters = {
    'criterion': 'gini',
    'n_estimators': 200,
    'max_depth': 40,
    'min_samples_leaf': 2,
    'min_samples_split': 2,
    'random_state': 42
}

# choose Train&Test Set
useForTrainSet = [1, 2, 3, 5]
useForTestSet = [4]

# Frame Open Set
useFrameOpenMode = True
# useFrameOpenMode = False

# Frame Open Parameter
frameOpenSet = [1, 2, 3, 4, 5]
# [train,drop,test]
splitRate = [7, 2, 1]

# Send To Echo
useEchoBot = False
# useEchoBot = True

# Persistence
# dumpModel = True
dumpModel = False

# set Path
path = r"D:/UTA Real-Life Drowsiness Dataset AU Preprocessing/Group{0}/"
outPath = r"D:/UTA Real-Life Drowsiness Dataset AU Preprocessing/Train&Test Set/"
modelPath = r"D:/UTA Real-Life Drowsiness Dataset AU Preprocessing/model/FrameOpen/"

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
        cList = Au_XX_C.copy()
        rList = Au_XX_R.copy()
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


def Normalized(SampleData):
    dataNoLabel = SampleData.drop('label', axis=1)
    dataLabel = SampleData['label']
    # has [divide by zero errors]
    # minMaxNormalized = (dataNoLabel - dataNoLabel.min()) / (dataNoLabel.max() - dataNoLabel.min())

    minMaxNormalized = pd.DataFrame(MinMaxScaler().fit_transform(dataNoLabel), columns=dataNoLabel.columns,
                                    index=dataNoLabel.index)
    SampleData = pd.concat([minMaxNormalized, dataLabel], axis=1)
    return SampleData


def LoadSet(GroupPath, AuParameter):
    sampleList = []

    folderList = [folder for folder in os.listdir(GroupPath) if os.path.isdir(os.path.join(GroupPath, folder))]
    for folder in folderList:
        samplePath = GroupPath + folder
        fileList = os.listdir(samplePath)
        sampleSet = None
        splitPointList = [0]
        for file in fileList:
            if os.path.splitext(file)[1] == '.csv' and "mix" in os.path.splitext(file)[0]:
                csvPath = samplePath + '/' + file
                data = pd.read_csv(csvPath)[AuParameter]
                splitPointList.append(data.shape[0])
                sampleSet = concatDF(sampleSet, data)
        sampleSet = Normalized(sampleSet)
        for i in range(len(splitPointList) - 1):
            sampleList.append(sampleSet[splitPointList[i]:splitPointList[i] + splitPointList[i + 1]])

    return sampleList


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
                elif preMethod == 'max':
                    data = PreMax(data)
                elif preMethod == 'maxAverage':
                    data = PreMaxAverage(data)
                elif preMethod == 'maxAverageMin':
                    data = PreMaxAverageMin(data)
                dataSet = concatDF(dataSet, data)
                print('folder:{0},size:{1},SetSize:{2}'.format(csvPath, data.shape, dataSet.shape))
    print('GroupPath:{0},Total Size:{1}'.format(GroupPath, dataSet.shape))

    return dataSet


def GeneralTTSetByGroup(GroupPath, AuParameter):
    sampleList = LoadSet(GroupPath, AuParameter)
    TrainSet = None
    TestSet = None

    for sample in sampleList:
        data = sample
        if preMethod == 'average':
            data = PreAverage(data)
        elif preMethod == 'highMapping':
            data = PreHighMapping(data)
        elif preMethod == 'max':
            data = PreMax(data)
        elif preMethod == 'maxAverage':
            data = PreMaxAverage(data)
        elif preMethod == 'maxAverageMin':
            data = PreMaxAverageMin(data)
        elif preMethod == 'highMappingV2':
            data = PreHighMappingV2(data)
        if splitMethod == 'positive':
            trainData = data[:data.shape[0] * splitRate[0] // 10]
            testData = data[-data.shape[0] * splitRate[2] // 10:]
        elif splitMethod == 'reverse':
            trainData = data[-data.shape[0] * splitRate[0] // 10:]
            testData = data[:data.shape[0] * splitRate[2] // 10]
        elif splitMethod == 'random':
            data = data.sample(frac=1, random_state=42)
            trainData = data[:data.shape[0] * splitRate[0] // 10]
            testData = data[-data.shape[0] * splitRate[2] // 10:]
        TrainSet = concatDF(TrainSet, trainData)
        TestSet = concatDF(TestSet, testData)
    print('GroupPath:{0},Train Size:{1},Test Size:{2}'.format(GroupPath, TrainSet.shape, TestSet.shape))
    return TrainSet, TestSet


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


def PreMax(originalData):
    maxGroup = []
    label = originalData['label']
    originalData = originalData.drop(['label'], axis=1)
    groupNum = originalData.shape[0] // preParameter

    for i in range(groupNum):
        first = i * preParameter
        last = (i + 1) * preParameter
        maxData = originalData[first:last].max(axis=0)
        maxGroup.append(maxData)
    # Handle the remainder that is not divisible
    if originalData.shape[0] % preParameter != 0:
        maxGroup.append(originalData[groupNum * preParameter:].max(axis=0))

    maxGroup = pd.DataFrame(maxGroup)
    label = label[:maxGroup.shape[0]]

    maxGroup = pd.concat([maxGroup, label], axis=1)

    return maxGroup


def PreHighMapping(originalData):
    highMappingGroup = []
    label = originalData['label']
    originalDataValues = originalData.drop(['label'], axis=1).values
    originalColumns = originalData.drop(['label'], axis=1).columns.values.tolist()
    groupNum = originalData.shape[0] // preParameter

    for i in range(groupNum):
        first = i * preParameter
        last = (i + 1) * preParameter
        highMappingData = originalDataValues[first:last].flatten()
        highMappingGroup.append(highMappingData)
    highMappingGroup = np.array(highMappingGroup)

    label = label[:groupNum]
    highMappingGroup = np.column_stack((highMappingGroup, label))

    newColumns = []
    for i in range(preParameter):
        index = [j + '_' + str(i + 1) for j in originalColumns]
        newColumns.extend(index)
    newColumns.append('label')

    highMappingGroup = pd.DataFrame(highMappingGroup)
    highMappingGroup.columns = newColumns

    return highMappingGroup


def PreHighMappingV2(originalData):
    highMappingGroup = []
    label = originalData['label']
    originalDataValues = originalData.drop(['label'], axis=1).values
    originalColumns = originalData.drop(['label'], axis=1).columns.values.tolist()
    groupNum = originalData.shape[0] // preParameter

    initialData = np.zeros((1, originalDataValues.shape[1] * preParameter))

    for i in range(groupNum):
        first = i * preParameter
        last = (i + 1) * preParameter
        nowData = originalDataValues[first:last].flatten()
        if i == 0:
            highMappingData = np.append(initialData, nowData)
        else:
            highMappingData = np.append(preData, nowData)
        preData = nowData
        highMappingGroup.append(highMappingData)
    highMappingGroup = np.array(highMappingGroup)

    label = label[:groupNum]
    highMappingGroup = np.column_stack((highMappingGroup, label))

    newColumns = []
    for i in range(preParameter * 2):
        index = [j + '_' + str(i + 1) for j in originalColumns]
        newColumns.extend(index)
    newColumns.append('label')

    highMappingGroup = pd.DataFrame(highMappingGroup)
    highMappingGroup.columns = newColumns

    return highMappingGroup


def PreMaxAverage(originalData):
    highMappingGroup = []
    label = originalData['label']
    originalDataValues = originalData.drop(['label'], axis=1).values
    originalColumns = originalData.drop(['label'], axis=1).columns.values.tolist()
    groupNum = originalData.shape[0] // preParameter

    for i in range(groupNum):
        first = i * preParameter
        last = (i + 1) * preParameter
        maxData = originalDataValues[first:last].max(axis=0)
        averageData = originalDataValues[first:last].mean(axis=0)
        highMappingData = np.hstack((maxData, averageData))
        highMappingGroup.append(highMappingData)

    # Handle the remainder that is not divisible
    if originalData.shape[0] % preParameter != 0:
        highMappingGroup.append(np.hstack((originalDataValues[groupNum * preParameter:].max(axis=0),
                                           originalDataValues[groupNum * preParameter:].mean(axis=0))))

    highMappingGroup = pd.DataFrame(highMappingGroup)
    label = label[:highMappingGroup.shape[0]]

    if useModel == 'C':
        highMappingGroup[highMappingGroup >= 0.5] = 1
        highMappingGroup[highMappingGroup < 0.5] = 0
    elif useModel == 'mix':
        if useAu == 'Full':
            dividingLine = len(Au_XX_C)
        else:
            dividingLine = len(useAu)
        # Data that does not require operation.
        noOperationData = highMappingGroup.iloc[:, :-dividingLine]
        operationData = highMappingGroup.iloc[:, -dividingLine:]
        operationData = operationData.where(operationData >= 0.5, 0)
        operationData = operationData.where(operationData < 0.5, 1)
        highMappingGroup = pd.concat([noOperationData, operationData], axis=1)

    highMappingGroup = pd.concat([highMappingGroup, label], axis=1)
    newColumns = []
    for i in ["max", "average"]:
        index = [j + '_' + i for j in originalColumns]
        newColumns.extend(index)
    newColumns.append('label')

    highMappingGroup.columns = newColumns

    return highMappingGroup


def PreMaxAverageMin(originalData):
    highMappingGroup = []
    label = originalData['label']
    originalDataValues = originalData.drop(['label'], axis=1).values
    originalColumns = originalData.drop(['label'], axis=1).columns.values.tolist()
    groupNum = originalData.shape[0] // preParameter

    for i in range(groupNum):
        first = i * preParameter
        last = (i + 1) * preParameter
        maxData = originalDataValues[first:last].max(axis=0)
        averageData = originalDataValues[first:last].mean(axis=0)
        if useModel == 'C':
            averageData[averageData >= 0.5] = 1
            averageData[averageData < 0.5] = 0
        elif useModel == 'mix':
            if useAu == 'Full':
                dividingLine = len(Au_XX_R)
            else:
                dividingLine = len(useAu)
            operationData = averageData[dividingLine:]
            noOperationData = averageData[:dividingLine]
            operationData[operationData >= 0.5] = 1
            operationData[operationData < 0.5] = 0
            averageData = np.hstack((noOperationData, operationData))
        minData = originalDataValues[first:last].min(axis=0)
        highMappingData = np.hstack((maxData, averageData, minData))
        highMappingGroup.append(highMappingData)

    # Handle the remainder that is not divisible
    if originalData.shape[0] % preParameter != 0:
        highMappingGroup.append(np.hstack((originalDataValues[groupNum * preParameter:].max(axis=0),
                                           originalDataValues[groupNum * preParameter:].mean(axis=0),
                                           originalDataValues[groupNum * preParameter:].min(axis=0))))

    highMappingGroup = pd.DataFrame(highMappingGroup)
    label = label[:highMappingGroup.shape[0]]

    highMappingGroup = pd.concat([highMappingGroup, label], axis=1)
    newColumns = []
    for i in ["max", "average", "min"]:
        index = [j + '_' + i for j in originalColumns]
        newColumns.extend(index)
    newColumns.append('label')

    highMappingGroup.columns = newColumns

    return highMappingGroup


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


def Predict(model, test):
    test_X, test_Y = DatasetToParameter(test)
    predict_Y = model.predict(test_X)
    test.insert(test.shape[1], 'preLabel', predict_Y)
    if useFrameOpenMode:
        test.to_csv(outPath + 'tmp_frameOpen_{0}.csv'.format(splitMethod))
    else:
        test.to_csv(outPath + 'tmp_peopleOpen.csv')
    return test


def GeneralTTSet():
    TrainSet = []
    TestSet = []
    parameter = PickUpParameter()
    # print(parameter)

    print("Generate Train&Test Set")
    if useFrameOpenMode:
        useGroup = frameOpenSet
        for i in tqdm(useGroup):
            TrainData, TestData = GeneralTTSetByGroup(path.format(i), parameter)
            TrainSet.append(TrainData)
            TestSet.append(TestData)
        TrainSet = pd.concat(TrainSet, ignore_index=True)
        TestSet = pd.concat(TestSet, ignore_index=True)
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
    if useFrameOpenMode:
        plt.savefig(outPath + 'tmp_{0}.png'.format(splitMethod))
    # plt.show()


def SaveModel(model):
    nowTime = time.strftime("%m%d%H%M%S", time.localtime())

    fileName = "{0}_{1}_{2}_{3}_{4}_{5}.pkl".format(nowTime, useModel, preMethod, preParameter, useForTrainSet,
                                                    useForTestSet)
    print("out model in:" + modelPath + fileName)
    joblib.dump(model, modelPath + fileName)


if __name__ == "__main__":
    if useFrameOpenMode:
        for splitMethod in ['positive', 'reverse', 'random']:
            trainSet, testSet = GeneralTTSet()
            print('UseAu:{0}\n'
                  'UseModel:{1}\n'
                  'PreMethod:{2}\n'
                  'PreParameter:{3}\n'
                  'FrameOpenMode:{4}\n'
                  'SplitMethod:{5}'.format(useAu, useModel, preMethod, preParameter, useFrameOpenMode, splitMethod))
            print('trainSet:{0},testSet:{1}'.format(trainSet.shape, testSet.shape))
            trainModel = Train(trainSet, testSet)
            predictSet = Predict(trainModel, testSet)
    else:
        trainSet, testSet = GeneralTTSet()

        # trainSet.to_csv(outPath + 'train.csv')
        # testSet.to_csv(outPath + 'test.csv')

        print('UseAu:{0}\n'
              'UseModel:{1}\n'
              'PreMethod:{2}\n'
              'PreParameter:{3}\n'
              'FrameOpenMode:{4}'.format(useAu, useModel, preMethod, preParameter, useFrameOpenMode))
        print('trainSet{0}:{1},testSet{2}:{3}'.format(useForTrainSet, trainSet.shape, useForTestSet, testSet.shape))

        # Training
        trainModel = Train(trainSet, testSet)
        # trainModel = TrainByGVSearch(trainSet, testSet)

        # OutPredict
        predictSet = Predict(trainModel, testSet)
    if dumpModel:
        SaveModel(trainModel)
