import os

import pandas as pd
import numpy as np

import dataTools as dt
import CreateFolder

Au_XX_R = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r',
           'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']
Au_XX_C = ['AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', 'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c',
           'AU17_c', 'AU20_c', 'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c']


def highMapping(data: pd.DataFrame, mergeNumber):

    data = data.drop(['frame'], axis=1)
    label = data['label']
    data = data.drop(['label'], axis=1)
    dataValue = data.values
    mergeNumber = int(mergeNumber)
    group = dataValue.shape[0] // mergeNumber
    mergeData = []
    for i in range(group):
        first = i * mergeNumber
        last = (i + 1) * mergeNumber
        temp = dataValue[first:last].flatten()
        mergeData.append(temp)
    mergeData = np.array(mergeData)
    label = label[:group]
    mergeData = np.column_stack((mergeData, label))
    return mergeData,int(label[0])


def combined_data(data: np.ndarray, AUType, highMappingNum):
    R_columns = []
    C_columns = []
    mix_columns = []
    for i in range(highMappingNum):
        index = '_' + str(i + 1)
        R_index = [j + index for j in Au_XX_R]
        C_index = [j + index for j in Au_XX_C]
        R_columns.extend(R_index)
        C_columns.extend(C_index)
        mix_columns.extend(R_index+C_index)
    R_columns.append('label')
    C_columns.append('label')
    mix_columns.append('label')

    data = pd.DataFrame(data)

    if AUType == 'C':
        data.columns = C_columns
    elif AUType == 'R':
        data.columns = R_columns
    elif AUType == 'mix':
        data.columns = mix_columns
    return data


path = r"D:/UTA Real-Life Drowsiness Dataset AU Preprocessing/Group{0}/"
fileList = []
mergeNumber = 150

if __name__ == "__main__":
    for i in range(1, 6):
        nowPath = path.format(i)
        folderList = os.listdir(nowPath)
        for folder in folderList:
            samplePath = nowPath + folder
            fileList = os.listdir(samplePath)
            nowOutPath = samplePath + '/highMapping_{0}/'.format(mergeNumber)
            if CreateFolder.mkdir(nowOutPath):
                for files in fileList:
                    if os.path.splitext(files)[1] == '.csv':
                        csvPath = samplePath + "/" + files
                        fileName = os.path.splitext(files)[0]
                        data = pd.read_csv(csvPath)
                        if 'mix' in fileName:
                            AUType = 'mix'
                        elif 'C' in fileName:
                            AUType = 'C'
                        elif 'R' in fileName:
                            AUType = 'R'
                        data,label = highMapping(data, mergeNumber)
                        data = combined_data(data,AUType,mergeNumber)
                        print(nowOutPath+str(label)+'_Au_XX_{0}_HighMappingSet.csv'.format(AUType))
                        data.to_csv(nowOutPath+str(label)+'_Au_XX_{0}_HighMappingSet.csv'.format(AUType), index=False)
            else:
                print('Data already exists in the specified directory')
