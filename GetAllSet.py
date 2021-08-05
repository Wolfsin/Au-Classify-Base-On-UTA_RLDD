import os
import pandas as pd
import numpy as np

import dataTools as dt


def addLabel(group: int, Au_XX_R_merge, Au_XX_C_merge, label):
    labelArray = np.zeros([group, 1], int)
    labelArray[:, :] = label
    Au_XX_R_merge_AddLabel = np.hstack((Au_XX_R_merge, labelArray))
    Au_XX_C_merge_AddLabel = np.hstack((Au_XX_C_merge, labelArray))
    Au_XX_mix_merge_AddLabel = np.hstack((Au_XX_R_merge, Au_XX_C_merge, labelArray))

    return Au_XX_C_merge_AddLabel, Au_XX_R_merge_AddLabel, Au_XX_mix_merge_AddLabel


path = r"D:/UTA Real-Life Drowsiness Dataset OpenFace analysis/Group{0}/"
AuPath = r"D:/UTA Real-Life Drowsiness Dataset AU Preprocessing/Group{0}/"
fileList = []

if __name__ == "__main__":
    for i in range(1, 6):
        nowPath = path.format(i)
        folderList = os.listdir(nowPath)
        for folder in folderList:
            samplePath = nowPath + folder
            fileList = os.listdir(samplePath)
            for files in fileList:
                if os.path.splitext(files)[1] == ".csv":
                    csvPath = samplePath + "/" + files
                    fileName = os.path.splitext(files)[0]
                    data = pd.read_csv(csvPath)
                    print(data.shape)
                    frame, Au_XX_R_merge, Au_XX_R_values, Au_XX_C_merge, Au_XX_C_values = dt.get_parameter(data)
                    if '0' in fileName and '10' not in fileName:
                        label = 0
                        print("0" + csvPath)
                    elif '5' in fileName:
                        label = 5
                        print("5" + csvPath)
                    elif '10' in fileName:
                        label = 10
                        print("10" + csvPath)
                    # add label
                    Au_XX_C_AddLabel, Au_XX_R_AddLabel, Au_XX_mix_AddLabel = addLabel(frame.size,Au_XX_R_values,Au_XX_C_values,label)
                    Au_XX_R_DataFrame, Au_XX_C_DataFrame, Au_XX_mix_DataFrame = dt.combined_data(1,frame, Au_XX_R_AddLabel, Au_XX_C_AddLabel, Au_XX_mix_AddLabel)
                    Au_XX_R_DataFrame.to_csv(AuPath.format(i)+folder + '/' + str(label) +'_Au_XX_R_ALLSet.csv', index=False)
                    Au_XX_C_DataFrame.to_csv(AuPath.format(i)+folder + '/' + str(label) +'_Au_XX_C_ALLSet.csv', index=False)
                    Au_XX_mix_DataFrame.to_csv(AuPath.format(i)+folder + '/' + str(label) +'_Au_XX_mix_ALLSet.csv', index=False)
