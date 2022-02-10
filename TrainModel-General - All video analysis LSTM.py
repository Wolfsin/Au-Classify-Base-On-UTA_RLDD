import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from keras.models import Sequential, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, LSTM, Masking, TimeDistributed, Bidirectional, Reshape

# Test data is fully independent.
# True = Test data not fully independent , False = test data is independent
isSpecified = True
# isSpecified = False

# Use in Test (Only for isSpecified = False)
TestDataGroup = 1

# Segmentation methods for each data set in a specific user model
splitMethod = 'positive'
# splitMethod = 'reverse'
# splitMethod = 'random'

# Preprocessing data
# preProcess = 'None'
preProcess = 'EachStandardScaler'
# preProcess = '0LabelStandardScaler'
# preProcess = 'AllStandardScaler'


# Full Au Parameter
Au_XX_R = [' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r', ' AU10_r', ' AU12_r', ' AU14_r',
           ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r', ' AU26_r', ' AU45_r']
Au_XX_C = [' AU01_c', ' AU02_c', ' AU04_c', ' AU05_c', ' AU06_c', ' AU07_c', ' AU09_c', ' AU10_c', ' AU12_c', ' AU14_c',
           ' AU15_c', ' AU17_c', ' AU20_c', ' AU23_c', ' AU25_c', ' AU26_c', ' AU28_c', ' AU45_c']


def concatDF(Set, newDataSet):
    if Set is None:
        Set = newDataSet
    else:
        Set = np.concatenate((Set, newDataSet), axis=0)
    return Set


def PickUpParameter():
    rList = Au_XX_R.copy()
    rList.append('label')
    return rList


def DataExpansion(data, drawFrameCount=20, interval=20, splitFlag=False, label=None, mixFlag=False):
    """
    drawFrameCount must be less than interval, otherwise the same data will appear multiple times in the result.
    """
    resList = []
    trainList = []
    valList = []
    testList = []
    if mixFlag:
        for sample in data:
            res = DataExpansion(sample, splitFlag=splitFlag, mixFlag=False)
            if isinstance(res,tuple):
                trainList.append(res[0])
                valList.append(res[1])
                testList.append(res[2])
            else:
                resList.extend(res)
    else:
        for i in range(drawFrameCount):
            supData = data[i:, :]
            expansionData = supData[::interval, :]
            resList.append(expansionData)

    if splitFlag:
        if not mixFlag:
            if splitMethod == 'positive':
                trainList = resList[:int(len(resList) * 0.8)]
                valList = resList[int(len(resList) * 0.8):int(len(resList) * 0.9)]
                testList = resList[int(len(resList) * 0.9):]
            elif splitMethod == 'reverse':
                trainList = resList[-int(len(resList) * 0.8):]
                valList = resList[-int(len(resList) * 0.9):-int(len(resList) * 0.8)]
                testList = resList[:-int(len(resList) * 0.9)]
            elif splitMethod == 'random':
                trainList, valList = train_test_split(resList, test_size=0.2, random_state=42)
                valList, testList = train_test_split(valList, test_size=0.5, random_state=42)
        if preProcess == 'EachStandardScaler':
            trainList = PreEachStandardScalerAfterSplit(trainList)
            valList = PreEachStandardScalerAfterSplit(valList)
            testList = PreEachStandardScalerAfterSplit(testList)
        elif preProcess == '0LabelStandardScaler':
            if label == 0:
                trainList = PreEachStandardScalerAfterSplit(trainList)
                valList = PreEachStandardScalerAfterSplit(valList, TestFlag=True)
                testList = PreEachStandardScalerAfterSplit(testList, TestFlag=True)
            elif label == 5:
                trainList = PreEachStandardScalerAfterSplit(trainList, TestFlag=True)
                valList = PreEachStandardScalerAfterSplit(valList, TestFlag=True)
                testList = PreEachStandardScalerAfterSplit(testList, TestFlag=True)
            elif label == 10:
                trainList = PreEachStandardScalerAfterSplit(trainList, TestFlag=True)
                valList = PreEachStandardScalerAfterSplit(valList, TestFlag=True)
                testList = PreEachStandardScalerAfterSplit(testList, TestFlag=True)
        elif preProcess == 'AllStandardScaler' and mixFlag:
            trainList = PreEachStandardScalerAfterSplit(trainList, humanFlag=True)
            valList = PreEachStandardScalerAfterSplit(valList, TestFlag=True, humanFlag=True)
            testList = PreEachStandardScalerAfterSplit(testList, TestFlag=True, humanFlag=True)
        return trainList, valList, testList
    else:
        return resList


def PreEachStandardScaler(data, TestFlag=False, humanFlag=False):
    global Scaler
    dataNoLabel = None

    if TestFlag:
        if humanFlag:
            res = []
            for sample in data:
                dataNoLabel = sample[:, :-3]
                label = sample[:, -3:]
                scalData = Scaler.transform(dataNoLabel)
                scalData = np.concatenate((scalData, label), axis=1)
                res.append(scalData)
            return res
        else:
            dataNoLabel = data[:, :-3]
            label = data[:, -3:]
            scalData = Scaler.transform(dataNoLabel)
            scalData = np.concatenate((scalData, label), axis=1)
    else:
        Scaler = StandardScaler()
        if humanFlag:
            for sample in data:
                dataNoLabel = concatDF(dataNoLabel, sample[:, :-3])
        else:
            dataNoLabel = data[:, :-3]
        Scaler.fit_transform(dataNoLabel)
        scalData = PreEachStandardScaler(data, TestFlag=True, humanFlag=humanFlag)

    return scalData


def PreEachStandardScalerAfterSplit(dataList, TestFlag=False, humanFlag=False):
    global Scaler
    scalData = None
    resList = []
    if TestFlag:
        if humanFlag:
            for sampleList in dataList:
                for data in sampleList:
                    dataNoLabel = data[:, :-3]
                    label = data[:, -3:]
                    scalData = Scaler.transform(dataNoLabel)
                    scalData = np.concatenate((scalData, label), axis=1)
                    resList.append(scalData)
        else:
            for data in dataList:
                dataNoLabel = data[:, :-3]
                label = data[:, -3:]
                scalData = Scaler.transform(dataNoLabel)
                scalData = np.concatenate((scalData, label), axis=1)
                resList.append(scalData)
    else:
        Scaler = StandardScaler()
        if humanFlag:
            for sampleList in dataList:
                for data in sampleList:
                    data = data[:, :-3]
                    scalData = concatDF(scalData, data)
        else:
            for data in dataList:
                data = data[:, :-3]
                scalData = concatDF(scalData, data)
        Scaler.fit_transform(scalData)
        resList = PreEachStandardScalerAfterSplit(dataList, TestFlag=True, humanFlag=humanFlag)
    return resList


def GetXY(data, manyToOne=False):
    X = data[:, :, :-3]
    Y = data[:, :, -3:]
    if manyToOne:
        Y = Y[:, 1, :]
    return X, Y


def GenerateInputSet(Path=r'D:/UTA Real-Life Drowsiness Dataset AU Preprocessing/Group{}/', LoadWithSplit=False):
    GroupList = []
    TestList = []
    trainSet, valSet, testSet = [], [], []

    for i in tqdm(range(1, 6), desc='Loading Group'):
        if i == TestDataGroup and (not isSpecified):
            # i > 5 all group are in GroupList,TestList will be empty
            TestList.extend(LoadSet(Path.format(i), True) )
            # TestList.extend(LoadSetByHuman(Path.format(i),testSetFlag=False))
        else:
            # GroupList.extend(LoadSetByHuman(Path.format(i)))
            if LoadWithSplit:
                res = LoadSetWithSplitTVT(Path.format(i))
                trainSet.extend(res[0])
                valSet.extend(res[1])
                testSet.extend(res[2])
            else:
                GroupList.extend(LoadSet(Path.format(i)))
    if LoadWithSplit:
        trainSet = pad_sequences(trainSet, maxlen=900, padding="post", truncating="pre", value=-99, dtype='float32')
        valSet = pad_sequences(valSet, maxlen=900, padding="post", truncating="pre", value=-99, dtype='float32')
        testSet = pad_sequences(testSet, maxlen=900, padding="post", truncating="pre", value=-99, dtype='float32')
    else:
        rawInputs = np.array(GroupList)
        paddedInputs = pad_sequences(rawInputs, maxlen=900, padding="post", truncating="pre", value=-99,
                                     dtype='float32')
        trainSet, valSet = train_test_split(paddedInputs, test_size=0.2, random_state=42)
        for index, sample in enumerate(TestList):
            TestList[index] = pad_sequences(sample, maxlen=900, padding="post", truncating="pre", value=-99, dtype='float32')
        testSet = np.array(TestList)

    np.save(r'D:/UTA Real-Life Drowsiness Dataset AU Preprocessing/Train&Test Set/trainSet.npy', trainSet)
    np.save(r'D:/UTA Real-Life Drowsiness Dataset AU Preprocessing/Train&Test Set/valSet.npy', valSet)
    np.save(r'D:/UTA Real-Life Drowsiness Dataset AU Preprocessing/Train&Test Set/testSet.npy', testSet)

    # testSet = np.array(TestList)
    # testSet = pad_sequences(testSet,maxlen=900, padding="post", truncating="pre", value=-1, dtype='float32')
    # valSet,testSet = train_test_split(testSet, test_size=0.5)
    # trainSet = paddedInputs

    zhangTrain, zhangTest = LoadZHANGSet()
    zhangTrain = pad_sequences(zhangTrain, maxlen=900, padding="post", truncating="pre", value=-99, dtype='float32')
    zhangTest = pad_sequences(zhangTest, maxlen=900, padding="post", truncating="pre", value=-99, dtype='float32')
    zhangVal, zhangTest = train_test_split(zhangTest, test_size=0.5, random_state=42)

    trainSet = np.concatenate((trainSet, zhangTrain), axis=0)
    valSet = np.concatenate((valSet, zhangVal), axis=0)
    testSet = zhangTest
    # Only use zhangData for train
    # trainSet = zhangTrain

    # valSet = np.concatenate((valSet, zhangVal), axis=0)
    # valSet = zhangVal
    # testSet = np.concatenate((testSet, zhangTest), axis=0)
    # testSet = zhangTest
    print("Complete loading \n"
          "TrainSet Shape: {}\n"
          "ValSet Shape: {}\n"
          "TestSet Shape: {}".format(trainSet.shape, valSet.shape, testSet.shape))
    return trainSet, valSet, testSet


def LoadSet(GroupPath=r'D:/UTA Real-Life Drowsiness Dataset AU Preprocessing/Group1/', testSetFlag=False):
    sampleList = []
    AuParameter = PickUpParameter()
    res = []

    folderList = [folder for folder in os.listdir(GroupPath) if os.path.isdir(os.path.join(GroupPath, folder))]
    for folder in folderList:
        samplePath = GroupPath + folder
        fileList = os.listdir(samplePath)
        for file in fileList:
            if os.path.splitext(file)[1] == '.csv' and "mix" in os.path.splitext(file)[0]:
                csvPath = samplePath + '/' + file
                data = pd.read_csv(csvPath)[AuParameter]
                labelOneHot = data['label']
                if labelOneHot[0] == 0:
                    label = 0
                    labelOneHot = np.array([[1, 0, 0]] * len(labelOneHot))
                elif labelOneHot[0] == 5:
                    label = 5
                    labelOneHot = np.array([[0, 1, 0]] * len(labelOneHot))
                elif labelOneHot[0] == 10:
                    label = 10
                    labelOneHot = np.array([[0, 0, 1]] * len(labelOneHot))

                data = data.drop(['label'], axis=1).to_numpy()
                data = np.concatenate((data, labelOneHot), axis=1)
                if preProcess == "EachStandardScaler":
                    data = PreEachStandardScaler(data)
                elif preProcess == "0LabelStandardScaler":
                    if label == 0:
                        data = PreEachStandardScaler(data)
                    else:
                        data = PreEachStandardScaler(data,TestFlag=True)
                if preProcess == "AllStandardScaler":
                    res.append(data)
                    continue
                if testSetFlag:
                    res.extend(DataExpansion(data))
                else:
                    sampleList.extend(DataExpansion(data))
        if preProcess == "AllStandardScaler":
            res = DataExpansion(PreEachStandardScaler(res, humanFlag=True), mixFlag=True)
            if testSetFlag:
                sampleList.append(res)
            else:
                sampleList.extend(res)
        else:
            if testSetFlag:
                sampleList.append(res)
        res = []
    return sampleList


def LoadSetWithSplitTVT(GroupPath=r'D:/UTA Real-Life Drowsiness Dataset AU Preprocessing/Group1/'):
    trainList = []
    valList = []
    testList = []
    AuParameter = PickUpParameter()
    label = None
    res = []

    folderList = [folder for folder in os.listdir(GroupPath) if os.path.isdir(os.path.join(GroupPath, folder))]
    for folder in folderList:
        samplePath = GroupPath + folder
        fileList = os.listdir(samplePath)
        for file in fileList:
            if os.path.splitext(file)[1] == '.csv' and "mix" in os.path.splitext(file)[0]:
                csvPath = samplePath + '/' + file
                data = pd.read_csv(csvPath)[AuParameter]
                labelOneHot = data['label']
                if labelOneHot[0] == 0:
                    label = 0
                    labelOneHot = np.array([[1, 0, 0]] * len(labelOneHot))
                elif labelOneHot[0] == 5:
                    label = 5
                    labelOneHot = np.array([[0, 1, 0]] * len(labelOneHot))
                elif labelOneHot[0] == 10:
                    label = 10
                    labelOneHot = np.array([[0, 0, 1]] * len(labelOneHot))

                data = data.drop(['label'], axis=1).to_numpy()
                data = np.concatenate((data, labelOneHot), axis=1)
                if preProcess != "AllStandardScaler":
                    res = DataExpansion(data, splitFlag=True, label=label)
                    trainList.extend(res[0])
                    valList.extend(res[1])
                    testList.extend(res[2])
                else:
                    res.append(data)
        if preProcess == "AllStandardScaler":
            res = DataExpansion(res, splitFlag=True, mixFlag=True)
            trainList.extend(res[0])
            valList.extend(res[1])
            testList.extend(res[2])
            res = []
    return trainList, valList, testList


def GetFinalTestData(AuParameter, Path=r'D:\UTA Real-Life Drowsiness Dataset AU Preprocessing\FinalTestGroup\zhang_2'):
    dictList = {}
    labelList = os.listdir(Path)
    for label in labelList:
        sampleList = []
        dataList = []
        folderPath = os.path.join(Path, label)
        folderList = os.listdir(folderPath)
        for file in folderList:
            samplePath = os.path.join(folderPath, file)
            sampleList = os.listdir(samplePath)
            for sample in sampleList:
                if os.path.splitext(sample)[1] == '.csv' and "mix" in os.path.splitext(sample)[0]:
                    csvPath = os.path.join(samplePath, sample)
                    data = pd.read_csv(csvPath)[AuParameter]
                    # data = PreEachStandardScaler(data)
                    labelOneHot = data['label']
                    if labelOneHot[0] == 0:
                        labelOneHot = np.array([[1, 0, 0]] * len(labelOneHot))
                    elif labelOneHot[0] == 5:
                        labelOneHot = np.array([[0, 1, 0]] * len(labelOneHot))
                    elif labelOneHot[0] == 10:
                        labelOneHot = np.array([[0, 0, 1]] * len(labelOneHot))
                    data = data.drop(['label'], axis=1).to_numpy()
                    data = np.concatenate((data, labelOneHot), axis=1)
                    data = PreEachStandardScaler(data)
                    dataList.append(DataExpansion(data))
        dictList[label] = dataList
    return dictList


def dataSplit(fullList, firstSubListItemNum=[1]):
    sublist1, sublist2 = [], []
    for i in range(len(fullList)):
        if i in firstSubListItemNum:
            sublist1.extend(fullList[i])
        else:
            sublist2.extend(fullList[i])
    return sublist1, sublist2


def LoadZHANGSet():
    trainSet = []
    testSet = []

    finalTestDict = GetFinalTestData(PickUpParameter())
    for key, value in finalTestDict.items():
        testTemp, trainTemp = dataSplit(value, [1])
        trainSet.extend(trainTemp)
        testSet.extend(testTemp)

    # zhang_0_0 = \
    #     pd.read_csv(
    #         r'D:\UTA Real-Life Drowsiness Dataset AU Preprocessing\FinalTestGroup\zhang_0\0_Au_XX_R_ALLSet.csv')[
    #         PickUpParameter()]
    # zhang_0_10 = \
    #     pd.read_csv(
    #         r'D:\UTA Real-Life Drowsiness Dataset AU Preprocessing\FinalTestGroup\zhang_0\10_Au_XX_R_ALLSet.csv')[
    #         PickUpParameter()]
    # zhang_1_0 = \
    #     pd.read_csv(
    #         r'D:\UTA Real-Life Drowsiness Dataset AU Preprocessing\FinalTestGroup\zhang_1\0_Au_XX_R_ALLSet.csv')[
    #         PickUpParameter()]
    # zhang_1_10 = \
    #     pd.read_csv(
    #         r'D:\UTA Real-Life Drowsiness Dataset AU Preprocessing\FinalTestGroup\zhang_1\10_Au_XX_R_ALLSet.csv')[
    #         PickUpParameter()]
    #
    # zhang_0_0 = PreEachStandardScaler(zhang_0_0)
    # zhang_0_10 = PreEachStandardScaler(zhang_0_10)
    # zhang_1_0 = PreEachStandardScaler(zhang_1_0)
    # zhang_1_10 = PreEachStandardScaler(zhang_1_10)
    #
    # zhang_0_0_label = np.array([[1, 0, 0]] * zhang_0_0.shape[0])
    # zhang_0_10_label = np.array([[0, 0, 1]] * zhang_0_10.shape[0])
    # zhang_1_0_label = np.array([[1, 0, 0]] * zhang_1_0.shape[0])
    # zhang_1_10_label = np.array([[0, 0, 1]] * zhang_1_10.shape[0])
    #
    # zhang_0_0 = np.concatenate((zhang_0_0.drop(['label'], axis=1).to_numpy(), zhang_0_0_label), axis=1)
    # zhang_0_10 = np.concatenate((zhang_0_10.drop(['label'], axis=1).to_numpy(), zhang_0_10_label), axis=1)
    # zhang_1_0 = np.concatenate((zhang_1_0.drop(['label'], axis=1).to_numpy(), zhang_1_0_label), axis=1)
    # zhang_1_10 = np.concatenate((zhang_1_10.drop(['label'], axis=1).to_numpy(), zhang_1_10_label), axis=1)
    #
    # trainSet.extend(DataExpansion(zhang_0_0))
    # trainSet.extend(DataExpansion(zhang_0_10))
    # testSet.extend(DataExpansion(zhang_1_0))
    # testSet.extend(DataExpansion(zhang_1_10))

    return np.array(trainSet), np.array(testSet)


def LoadSetByHuman(GroupPath=r'D:/UTA Real-Life Drowsiness Dataset AU Preprocessing/Group1/', testSetFlag=False):
    sampleList = []
    AuParameter = PickUpParameter()

    folderList = [folder for folder in os.listdir(GroupPath) if os.path.isdir(os.path.join(GroupPath, folder))]
    for folder in folderList:
        samplePath = GroupPath + folder
        fileList = os.listdir(samplePath)
        splitPoint = 0
        splitPointList = [0]
        allSet = None
        for file in fileList:
            if os.path.splitext(file)[1] == '.csv' and "mix" in os.path.splitext(file)[0]:
                csvPath = samplePath + '/' + file
                data = pd.read_csv(csvPath)[AuParameter]
                splitPoint += data.shape[0]
                splitPointList.append(splitPoint)
                if allSet is None:
                    allSet = data
                else:
                    allSet = pd.concat([allSet, data], axis=0)
        allSet = PreEachStandardScalerAfterSplit(allSet, )
        for i in range(len(splitPointList) - 1):
            data = allSet[splitPointList[i]:splitPointList[i + 1]]
            labelOneHot = data['label']
            if labelOneHot[0] == 0:
                labelOneHot = np.array([[1, 0, 0]] * len(labelOneHot))
            elif labelOneHot[0] == 5:
                labelOneHot = np.array([[0, 1, 0]] * len(labelOneHot))
            elif labelOneHot[0] == 10:
                labelOneHot = np.array([[0, 0, 1]] * len(labelOneHot))
            data = data.drop(['label'], axis=1).to_numpy()
            data = np.concatenate((data, labelOneHot), axis=1)
            if testSetFlag:
                sampleList.append(data)
            else:
                sampleList.extend(DataExpansion(data))
    return sampleList


def CreateModel(sequenceLength):
    model = Sequential()
    model.add(Masking(mask_value=-99, input_shape=(sequenceLength, 17,)))
    model.add(LSTM(units=16, kernel_initializer='Orthogonal', return_sequences=False))
    model.add(Dense(3, activation='softmax'))

    # frame base
    # model.add(Bidirectional(LSTM(units=64, return_sequences=True, kernel_initializer='Orthogonal'),merge_mode='concat'))
    # model.add(Bidirectional(LSTM(units=32, return_sequences=True, kernel_initializer='Orthogonal'), merge_mode='concat'))
    # model.add(Bidirectional(LSTM(units=16, return_sequences=True, kernel_initializer='Orthogonal'), merge_mode='concat'))
    # model.add(TimeDistributed(Dense(3, activation='softmax')))

    adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['categorical_accuracy'])
    model.summary()
    return model


def OutPlt(history):
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('categorical_accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def DrawConfusionMatrix(test_Y, predict_Y, label):
    plt.clf()
    fig, ax = plt.subplots(constrained_layout=True)

    # Output matrix: rows are real values and columns are predicted values
    cm = confusion_matrix(test_Y, predict_Y)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print(cm)
    sns.heatmap(cm_normalized, annot=True, cmap=plt.cm.binary, xticklabels=label, yticklabels=label,
                annot_kws={"fontsize": 12})

    fig.set_size_inches(w=7, h=3)

    ax.set_xlabel('True Label', fontsize=12)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('Predicted Label', fontsize=12)

    print('Confusion Matrix:')
    print(cm_normalized)
    plt.show()


Scaler = None
train, val, test = GenerateInputSet(LoadWithSplit=isSpecified)
trainX, trainY = GetXY(train, manyToOne=True)
valX, valY = GetXY(val, manyToOne=True)

testSS = test.copy()
testSS = testSS.reshape(testSS.shape[0]*testSS.shape[1],-1 , 20)
LSTMModel = CreateModel(trainX.shape[1])
hist = LSTMModel.fit(trainX, trainY, validation_data=(valX, valY), batch_size=180, epochs=200, shuffle=True,
                     verbose=1, callbacks=[EarlyStopping(monitor='val_loss', patience=20, min_delta=0.0001)])

if not isSpecified:
    testX, testY = GetXY(testSS, manyToOne=True)
    testY = np.argmax(testY, axis=1)
    pre = np.argmax(hist.model.predict(testX), axis=1)
    print(accuracy_score(testY, pre))
    DrawConfusionMatrix(testY, pre, label=['0', '5', '10'])
    # for i in range(test.shape[0]):
    #     res = GetXY(test[i], manyToOne=True)
    #     testX = res[0]
    #     testY = np.argmax(res[1], axis=-1)
    #     pre = hist.model.predict(testX)
    #     pre = np.argmax(pre, axis=-1)
    #     print("{}\n"
    #           "accuracy: {}\n".format(i, accuracy_score(testY, pre)))
    #     DrawConfusionMatrix(testY, pre, label=['0', '5', '10'])
else:
    testX, testY = GetXY(test, manyToOne=True)
    testY = np.argmax(testY, axis=-1)
    pre = hist.model.predict(testX)
    pre = np.argmax(pre, axis=-1)

    print(accuracy_score(testY, pre))
    DrawConfusionMatrix(testY, pre, label=['0', '5', '10'])

OutPlt(hist)

model = load_model(
    r'D:\UTA Real-Life Drowsiness Dataset AU Preprocessing\Train&Test Set\LSTM Model\LSTMModel_0.5_test4.h5')
pree = model.predict(testX)
pree = np.argmax(pree, axis=-1)
print(accuracy_score(testY, pree))
DrawConfusionMatrix(testY, pree, label=['0', '5', '10'])

hist.model.save('D:/UTA Real-Life Drowsiness Dataset AU Preprocessing/Train&Test Set/LSTMModel.h5')
