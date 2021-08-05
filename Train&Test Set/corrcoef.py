import pandas

def out_correlation(data: pandas.DataFrame,index,datatype):
    src = './TestSet By Group{0}/{1}_correlation.csv'.format(index,datatype)
    pandas.DataFrame({'AU': data.corr().label.index, 'correlation': data.corr().label.values}).to_csv(src, index=False)

testsrc = './TestSet By Group{0}/average/Au_XX_R_TestSet.csv'
trainsrc = './TestSet By Group{0}/average/Au_XX_R_TrainSet.csv'
for i in range(1,6):
    testdata = pandas.read_csv(testsrc.format(i))
    traindata = pandas.read_csv(trainsrc.format(i))
    out_correlation(testdata,i,'test')
    out_correlation(traindata,i,'train')