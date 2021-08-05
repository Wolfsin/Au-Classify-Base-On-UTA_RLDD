import numpy
import pandas

Au_XX_R = [' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r', ' AU10_r', ' AU12_r',
           ' AU14_r',
           ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r', ' AU26_r', ' AU45_r']
Au_XX_C = [' AU01_c', ' AU02_c', ' AU04_c', ' AU05_c', ' AU06_c', ' AU07_c', ' AU09_c', ' AU10_c', ' AU12_c',
           ' AU14_c',
           ' AU15_c', ' AU17_c', ' AU20_c', ' AU23_c', ' AU25_c', ' AU26_c', ' AU28_c', ' AU45_c']


def get_parameter(pd_data: pandas.DataFrame):
    """returns: frame, Au_XX_R_merge, Au_XX_R_values, Au_XX_C_merge, Au_XX_C_values"""
    df_clear = pd_data.drop(pd_data[pd_data[' success'] == 0].index)
    pd_frame = df_clear['frame'].values
    Au_XX_R_merge = numpy.zeros((1, 17))
    Au_XX_R_values = df_clear[Au_XX_R].values
    Au_XX_C_merge = numpy.zeros((1, 18))
    Au_XX_C_values = df_clear[Au_XX_C].values

    return pd_frame, Au_XX_R_merge, Au_XX_R_values, Au_XX_C_merge, Au_XX_C_values


def merge_data_average(group_number: int, Au_XX_R_merge: numpy.ndarray, Au_XX_R_values: numpy.ndarray,
                       Au_XX_C_merge: numpy.ndarray, Au_XX_C_values: numpy.ndarray, merge_number: int):
    """returns: Au_XX_R_merge, Au_XX_C_merge"""
    for i in range(group_number):
        first = i * merge_number
        last = (i + 1) * merge_number
        temp_R = numpy.mean(Au_XX_R_values[first:last], axis=0).reshape(-1, 17)
        Au_XX_R_merge = numpy.row_stack((Au_XX_R_merge, temp_R))
        temp_C = numpy.mean(Au_XX_C_values[first:last], axis=0).reshape(-1, 18)
        Au_XX_C_merge = numpy.row_stack((Au_XX_C_merge, temp_C))

    Au_XX_R_merge = numpy.delete(Au_XX_R_merge, 0, axis=0)
    Au_XX_C_merge = numpy.delete(Au_XX_C_merge, 0, axis=0)
    Au_XX_C_merge[Au_XX_C_merge >= 0.6] = 1
    Au_XX_C_merge[Au_XX_C_merge < 0.6] = 0

    return Au_XX_R_merge, Au_XX_C_merge


def merge_data_highMapping(group_number: int, Au_XX_R_values: numpy.ndarray, Au_XX_C_values: numpy.ndarray,
                           merge_number: int):
    Au_XX_R_merge = numpy.zeros((1, 17 * merge_number))
    Au_XX_C_merge = numpy.zeros((1, 18 * merge_number))

    for i in range(group_number):
        first = i * merge_number
        last = (i + 1) * merge_number
        temp_R = Au_XX_R_values[first:last].reshape(1, -1)
        Au_XX_R_merge = numpy.row_stack((Au_XX_R_merge, temp_R))
        temp_C = Au_XX_C_values[first:last].reshape(1, -1)
        Au_XX_C_merge = numpy.row_stack((Au_XX_C_merge, temp_C))

    Au_XX_R_merge = numpy.delete(Au_XX_R_merge, 0, axis=0)
    Au_XX_C_merge = numpy.delete(Au_XX_C_merge, 0, axis=0)
    return Au_XX_R_merge, Au_XX_C_merge


def combined_data(highmapping_number, frame, Au_XX_R_merge: numpy.ndarray, Au_XX_C_merge: numpy.ndarray,
                          Au_XX_mix_merge: numpy.ndarray):
    R_columns = []
    C_columns = []
    Au_XX_R_merge = numpy.hstack((frame.reshape(-1,1),Au_XX_R_merge))
    Au_XX_C_merge = numpy.hstack((frame.reshape(-1,1), Au_XX_C_merge))
    Au_XX_mix_merge = numpy.hstack((frame.reshape(-1,1), Au_XX_mix_merge))

    Au_XX_R_merge_DataFrame = pandas.DataFrame(Au_XX_R_merge)
    Au_XX_C_merge_DataFrame = pandas.DataFrame(Au_XX_C_merge)
    Au_XX_mix_merge_DataFrame = pandas.DataFrame(Au_XX_mix_merge)

    if highmapping_number <= 1:
        R_columns = Au_XX_R[:]
        C_columns = Au_XX_C[:]
    else:
        for i in range(highmapping_number):
            index = '_' + str(i + 1)
        for value in Au_XX_R:
            R_columns.append(value + index)
        for value in Au_XX_C:
            C_columns.append(value + index)
    mix_columns = R_columns + C_columns

    R_columns.append('label')
    C_columns.append('label')
    mix_columns.append('label')

    R_columns.insert(0,'frame')
    C_columns.insert(0,'frame')
    mix_columns.insert(0,'frame')

    Au_XX_R_merge_DataFrame.columns = R_columns
    Au_XX_C_merge_DataFrame.columns = C_columns
    Au_XX_mix_merge_DataFrame.columns = mix_columns

    return Au_XX_R_merge_DataFrame, Au_XX_C_merge_DataFrame, Au_XX_mix_merge_DataFrame


def import_dataset(src='./data/output/'):
    """为 Training-model 和 Run-PreModel 导入预处理完成的数据集"""
    Au_XX_C_trainSet = 'Au_XX_C_trainSet.csv'
    Au_XX_C_testSet = 'Au_XX_C_testSet.csv'
    Au_XX_R_trainSet = 'Au_XX_R_trainSet.csv'
    Au_XX_R_testSet = 'Au_XX_R_testSet.csv'
    Au_XX_mix_trainSet = 'Au_XX_mix_trainSet.csv'
    Au_XX_mix_testSet = 'Au_XX_mix_testSet.csv'

    trainSet_C = pandas.read_csv(src + Au_XX_C_trainSet)
    testSet_C = pandas.read_csv(src + Au_XX_C_testSet)

    trainSet_R = pandas.read_csv(src + Au_XX_R_trainSet)
    testSet_R = pandas.read_csv(src + Au_XX_R_testSet)

    trainSet_mix = pandas.read_csv(src + Au_XX_mix_trainSet)
    testSet_mix = pandas.read_csv(src + Au_XX_mix_testSet)

    return trainSet_C, testSet_C, trainSet_R, testSet_R, trainSet_mix, testSet_mix


def dataset_to_parameter(dataSet: pandas.DataFrame):
    """ return parameter X, Y"""
    X = dataSet.drop(['label'], axis=1)
    Y = dataSet['label']

    return X, Y


def out_correlation(data: pandas.DataFrame, model_name, src):
    src = src + model_name + '_correlation.csv'
    pandas.DataFrame({'AU': data.corr().label.index, 'correlation': data.corr().label.values}).to_csv(src, index=False)


if __name__ == "__main__":
    A = pandas.read_csv('./data/v2.0/normal/Au_XX_R_ALLSet.csv')
    B = pandas.read_csv('./data/v2.0/normal/normal.csv')
    mix_data = pandas.concat([A, B], ignore_index=True)
    mix_data.to_csv('./data/v2.0/normal/' + 'allSet.csv', index=False)
