import os
import pandas as pd
from tqdm import trange
import CreateFolder

path = r"D:/UTA Real-Life Drowsiness Dataset AU Preprocessing/Group{0}/"
outPath = r"D:/UTA Real-Life Drowsiness Dataset AU Preprocessing/Train&Test Set/TestSet By Group{0}"
allSet = {1, 2, 3, 4, 5}
# setType = '/average'
setType = '/highMapping_150'
Au_XX_C_GroupList = []
Au_XX_R_GroupList = []
Au_XX_mix_GroupList = []


def concatDF(Set, newDataSet):
    if Set is None:
        Set = newDataSet
    else:
        Set = pd.concat([Set, newDataSet], ignore_index=True)
    return Set


def concatByGroup(GroupPath):
    Au_XX_C_Set = None
    Au_XX_R_Set = None
    Au_XX_mix_Set = None

    folderList = os.listdir(GroupPath)
    for folder in folderList:
        samplePath = GroupPath + folder + setType
        fileList = os.listdir(samplePath)
        for file in fileList:
            if os.path.splitext(file)[1] == '.csv':
                csvPath = samplePath + '/' + file
                fileName = os.path.splitext(file)[0]
                data = pd.read_csv(csvPath)
                if 'mix' in fileName:
                    Au_XX_mix_Set = concatDF(Au_XX_mix_Set, data)
                    print('folder:{0},size:{1},SetSize:{2}'.format(csvPath, data.shape, Au_XX_mix_Set.shape))
                elif 'C' in fileName:
                    Au_XX_C_Set = concatDF(Au_XX_C_Set, data)
                    print('folder:{0},size:{1},SetSize:{2}'.format(csvPath, data.shape, Au_XX_C_Set.shape))
                elif 'R' in fileName:
                    Au_XX_R_Set = concatDF(Au_XX_R_Set, data)
                    print('folder:{0},size:{1},SetSize:{2}'.format(csvPath, data.shape, Au_XX_R_Set.shape))
    print('GroupPath:{0},Total Size:{1}'.format(GroupPath, Au_XX_mix_Set.shape))

    return Au_XX_C_Set, Au_XX_R_Set, Au_XX_mix_Set


if __name__ == "__main__":

    for i in range(1, 6):
        Au_XX_C_Set, Au_XX_R_Set, Au_XX_mix_Set = concatByGroup(path.format(i))
        Au_XX_C_GroupList.append(Au_XX_C_Set)
        Au_XX_R_GroupList.append(Au_XX_R_Set)
        Au_XX_mix_GroupList.append(Au_XX_mix_Set)
        Au_XX_C_Set.to_csv(outPath.format(i) + setType + '/Au_XX_C_TestSet.csv', index=False)
        Au_XX_R_Set.to_csv(outPath.format(i) + setType + '/Au_XX_R_TestSet.csv', index=False)
        Au_XX_mix_Set.to_csv(outPath.format(i) + setType + '/Au_XX_mix_TestSet.csv', index=False)

    print("Generate Train Set")

    for j in trange(1, 6):
        Au_XX_C_TestSet = Au_XX_C_GroupList[j - 1]
        Au_XX_R_TestSet = Au_XX_R_GroupList[j - 1]
        Au_XX_mix_TestSet = Au_XX_mix_GroupList[j - 1]

        Au_XX_C_TrainSet = Au_XX_C_GroupList.copy()
        Au_XX_R_TrainSet = Au_XX_R_GroupList.copy()
        Au_XX_mix_TrainSet = Au_XX_mix_GroupList.copy()

        Au_XX_C_TrainSet.pop(j-1)
        Au_XX_R_TrainSet.pop(j-1)
        Au_XX_mix_TrainSet.pop(j-1)

        Au_XX_C_TrainSet = pd.concat(Au_XX_C_TrainSet, ignore_index=True)
        Au_XX_R_TrainSet = pd.concat(Au_XX_R_TrainSet, ignore_index=True)
        Au_XX_mix_TrainSet = pd.concat(Au_XX_mix_TrainSet, ignore_index=True)

        Au_XX_C_TrainSet.to_csv(outPath.format(j) + setType + '/Au_XX_C_TrainSet.csv', index=False)
        Au_XX_R_TrainSet.to_csv(outPath.format(j) + setType + '/Au_XX_R_TrainSet.csv', index=False)
        Au_XX_mix_TrainSet.to_csv(outPath.format(j) + setType + '/Au_XX_mix_TrainSet.csv', index=False)
