import pandas
import os

import pandas as pd

from GenerateTTSet import concatDF

def out_correlation(data: pandas.DataFrame,outPath):
    src = os.path.join(outPath,'Au_XX_R_AllSet_correlation.csv')
    test = data.corr()
    pandas.DataFrame({'AU': data.corr().label.index, 'correlation': data.corr().label.values}).to_csv(src, index=False)


Path = r'D:\UTA Real-Life Drowsiness Dataset AU Preprocessing\Group{0}'
if __name__ == "__main__":
    # sampleBase
    # for i in range(1,6):
    #     nowPath = Path.format(i)
    #     folderList = os.listdir(nowPath)
    #     for sampleFolder in folderList:
    #         samplePath = os.path.join(nowPath,sampleFolder)
    #         concatSet = None
    #         for files in os.listdir(samplePath):
    #             csvPath = os.path.join(samplePath,files)
    #             fileName = os.path.splitext(files)[0]
    #             if 'R' in files and os.path.splitext(files)[1] == ".csv":
    #                 Set = pandas.read_csv(csvPath)
    #                 concatSet = concatDF(concatSet,Set)
    #         out_correlation(concatSet.drop(['frame'], axis=1),os.path.join(samplePath,'corrcoef'))

    # groupBase
    # for i in range(1,6):
    #     nowPath = Path.format(i)
    #     folderList = os.listdir(nowPath)
    #     concatSet = None
    #     for sampleFolder in folderList:
    #         if '.csv' in sampleFolder:
    #             continue
    #         samplePath = os.path.join(nowPath,sampleFolder)
    #         for files in os.listdir(samplePath):
    #             csvPath = os.path.join(samplePath,files)
    #             fileName = os.path.splitext(files)[0]
    #             if 'R' in files and os.path.splitext(files)[1] == ".csv":
    #                 Set = pandas.read_csv(csvPath)
    #                 concatSet = concatDF(concatSet,Set)
    #     out_correlation(concatSet.drop(['frame'], axis=1),nowPath)

    set_0 = r'D:\UTA Real-Life Drowsiness Dataset AU Preprocessing\Group3\30\0_Au_XX_R_ALLSet.csv'
    set_5 = r'D:\UTA Real-Life Drowsiness Dataset AU Preprocessing\Group3\30\5_Au_XX_R_ALLSet.csv'
    set_10 = r'D:\UTA Real-Life Drowsiness Dataset AU Preprocessing\Group3\30\10_Au_XX_R_ALLSet.csv'

    set_0 = pd.read_csv(set_0)
    set_5 = pd.read_csv(set_5)
    set_10 = pd.read_csv(set_10)

    allSet = concatDF(set_0,set_5)
    allSet = concatDF(allSet,set_10)
    out_correlation(allSet,r'D:\UTA Real-Life Drowsiness Dataset AU Preprocessing\Group3\30')


