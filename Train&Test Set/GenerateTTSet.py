import os
import pandas as pd
from tqdm import trange

inPath = r"D:/UTA Real-Life Drowsiness Dataset AU Preprocessing/Train&Test Set/TestSet By Group{0}"
outPath = r"D:/UTA Real-Life Drowsiness Dataset AU Preprocessing/Train&Test Set/test"
allSet = {1, 2, 3, 4, 5}
# setType = '/average/'
setType = '/highMapping_150/'
numofSet = {1,2,3,4}
Au_XX_C_GroupList = []
Au_XX_R_GroupList = []
Au_XX_mix_GroupList = []


def concatDF(Set, newDataSet):
    if Set is None:
        Set = newDataSet
    else:
        Set = pd.concat([Set, newDataSet], ignore_index=True)
    return Set


def getTestSet(GroupPath):
    Au_XX_C_Set = None
    Au_XX_R_Set = None
    Au_XX_mix_Set = None

    folderList = os.listdir(GroupPath)

    for file in folderList:
        if os.path.splitext(file)[1] == '.csv':
            fileName = os.path.splitext(file)[0]
            if 'Au_XX_mix_TestSet' in fileName:
                Au_XX_mix_Set = pd.read_csv(GroupPath+file)
            elif 'Au_XX_C_TestSet' in fileName:
                Au_XX_C_Set = pd.read_csv(GroupPath+file)
            elif 'Au_XX_R_TestSet' in fileName:
                Au_XX_R_Set = pd.read_csv(GroupPath+file)

    return Au_XX_C_Set, Au_XX_R_Set, Au_XX_mix_Set


if __name__ == "__main__":

    for i in numofSet:
        Au_XX_C_Set, Au_XX_R_Set, Au_XX_mix_Set = getTestSet(inPath.format(i)+setType)
        Au_XX_C_GroupList.append(Au_XX_C_Set)
        Au_XX_R_GroupList.append(Au_XX_R_Set)
        Au_XX_mix_GroupList.append(Au_XX_mix_Set)

    print("Get TestSet")

    Au_XX_C_TrainSet = pd.concat(Au_XX_C_GroupList, ignore_index=True)
    Au_XX_R_TrainSet = pd.concat(Au_XX_R_GroupList, ignore_index=True)
    Au_XX_mix_TrainSet = pd.concat(Au_XX_mix_GroupList, ignore_index=True)

    Au_XX_C_TrainSet.to_csv(outPath + '/Au_XX_C_TrainSet{0}.csv'.format(numofSet), index=False)
    Au_XX_R_TrainSet.to_csv(outPath + '/Au_XX_R_TrainSet{0}.csv'.format(numofSet), index=False)
    Au_XX_mix_TrainSet.to_csv(outPath +'/Au_XX_mix_TrainSet{0}.csv'.format(numofSet), index=False)
