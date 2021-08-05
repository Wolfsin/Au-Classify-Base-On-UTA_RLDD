import os
import pandas as pd

def concatDF(Set, newDataSet):
    if Set is None:
        Set = newDataSet
    else:
        Set = pd.concat([Set, newDataSet.iloc[:,1]], ignore_index=True ,axis=1)
    return Set


Path = r'D:\UTA Real-Life Drowsiness Dataset AU Preprocessing\Group{0}'
if __name__ == "__main__":
    for i in range(1,6):
        nowPath = Path.format(i)
        folderList = os.listdir(nowPath)
        groupConcat = None
        for sampleFolder in folderList:
            samplePath = os.path.join(nowPath,sampleFolder)
            targePath = os.path.join(samplePath,'corrcoef/Au_XX_R_AllSet_correlation.csv')
            Set = pd.read_csv(targePath)
            groupConcat = concatDF(groupConcat,Set)
        groupConcat.to_csv(os.path.join(nowPath,'Au_XX_R_AllSet_correlation_by_group.csv'))
        