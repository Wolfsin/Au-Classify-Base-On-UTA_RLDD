import joblib
import pandas as pd
import os

from tqdm import trange
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier

from tune_sklearn import TuneGridSearchCV

from sklearn2pmml import PMMLPipeline
from sklearn2pmml import sklearn2pmml

import time

MODEL = {
    0: "Classifier",
    1: "Regression"
}
# setType = 'average'
setType = 'highMapping'

numofSet = {1,2,3}
src = './TestSet By Group{0}/{1}/'
testSrc = './test/highMapping_150/Au_XX_R_TrainSet{1, 2, 3, 4}.csv'
testSrc2 = './test/highMapping_150/Au_XX_R_TestSet{5}.csv'

def import_dataset(path):
    """为 Training-model 和 Run-PreModel 导入预处理完成的数据集"""
    Au_XX_C_trainSet = 'Au_XX_C_TrainSet.csv'
    Au_XX_C_testSet = 'Au_XX_C_TestSet.csv'
    Au_XX_R_trainSet = 'Au_XX_R_TrainSet.csv'
    Au_XX_R_testSet = 'Au_XX_R_TestSet.csv'
    Au_XX_mix_trainSet = 'Au_XX_mix_TrainSet.csv'
    Au_XX_mix_testSet = 'Au_XX_mix_TestSet.csv'

    trainSet_C = pd.read_csv(path + Au_XX_C_trainSet)
    testSet_C = pd.read_csv(path + Au_XX_C_testSet)

    trainSet_R = pd.read_csv(path + Au_XX_R_trainSet)
    testSet_R = pd.read_csv(path + Au_XX_R_testSet)

    trainSet_mix = pd.read_csv(path + Au_XX_mix_trainSet)
    testSet_mix = pd.read_csv(path + Au_XX_mix_testSet)

    return trainSet_C, testSet_C, trainSet_R, testSet_R, trainSet_mix, testSet_mix


def dataset_to_parameter(dataSet: pd.DataFrame):
    """ return parameter X, Y"""
    X = dataSet.drop(['label'], axis=1)
    Y = dataSet['label']

    return X, Y


# find Best Hyper Parameter
def HyperParameter_search_Classifier(trainSet, testSet, setType):
    # create X,Y parameter
    train_X, train_Y = dataset_to_parameter(trainSet)
    test_X, test_Y = dataset_to_parameter(testSet)

    # Hyper parameter network
    parameters = {
        'n_estimators': [5, 10, 17, 18, 30, 35, 60, 100, 200],  # 决策树的数量
        'max_depth': [3, 5, 8, 10, 20, 40,50],  # 决策数的深度
        'random_state': [42],
        'min_samples_leaf': [2, 5, 10, 20, 50],  # 叶子节点最少的样本数
        'min_samples_split': [2, 5, 10, 20, 50]  # 每个划分最少的样本数
    }
    clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parameters, cv=2, verbose=1, n_jobs=-1)
    print('Start Training ' + setType + ' Model')
    clf.fit(train_X, train_Y)
    best_clf = clf.best_estimator_
    print('trainSet score: {:.2%}'.format(best_clf.score(train_X, train_Y)))
    print('testSet score: {:.2%}'.format(best_clf.score(test_X, test_Y)))
    print('Complete training')

    return clf.best_estimator_


# find Best Hyper Parameter
def HyperParameter_search_Regression(trainSet, testSet, setType):
    # create X,Y parameter
    train_X, train_Y = dataset_to_parameter(trainSet)
    test_X, test_Y = dataset_to_parameter(testSet)

    # Hyper parameter network
    parameters = {
        'criterion': ['mse'],
        'n_estimators': [5, 10, 17, 18, 30, 35, 60, 100, 150, 200],  # 决策树的数量
        'max_depth': [3, 5, 8, 10, 20, 40, 50],  # 决策数的深度
        'random_state': [42],
        'min_samples_leaf': [2, 5, 10, 20, 50],  # 叶子节点最少的样本数,较小的叶片数更容易捕捉到噪音
        'min_samples_split': [2, 5, 10, 20, 50],  # 每个划分最少的样本数
    }
    clf = GridSearchCV(estimator=RandomForestRegressor(), param_grid=parameters, cv=2, verbose=1, n_jobs=-1)
    # clf = TuneGridSearchCV(estimator=RandomForestRegressor(), param_grid=parameters, cv=2, verbose=1, n_jobs=-1)
    print('Start Training ' + setType + ' Model')
    start = time.time()
    clf.fit(train_X, train_Y)
    end = time.time()
    best_clf = clf.best_estimator_
    print("best parameters:")
    print(clf.best_params_)
    print('trainSet score: {:.2%}'.format(best_clf.score(train_X, train_Y)))
    print('testSet score: {:.2%}'.format(best_clf.score(test_X, test_Y)))
    print('use Time:{0}'.format(end - start))
    print('Complete training')

    return clf.best_estimator_

# Train 
def Train_Regression(trainSet, testSet, setType):
    train_X, train_Y = dataset_to_parameter(trainSet)
    test_X, test_Y = dataset_to_parameter(testSet)

    # RFR = RandomForestRegressor(criterion='mse', max_depth=40, min_samples_leaf=2,min_samples_split=2, n_estimators=200,random_state=42)
    RFR = RandomForestClassifier(criterion='gini', max_depth=40, min_samples_leaf=2,min_samples_split=2, n_estimators=200,random_state=42)
    print('Start Training ' + setType + ' Model')
    start = time.time()
    RFR.fit(train_X, train_Y)
    end = time.time()
    print('trainSet score: {:.2%}'.format(RFR.score(train_X, train_Y)))
    print('testSet score: {:.2%}'.format(RFR.score(test_X, test_Y)))
    print('use Time:{0}'.format(end - start))
    print('Complete training')

    return RFR

def Train_GradientBoosting(trainSet, testSet, setType):
    train_X, train_Y = dataset_to_parameter(trainSet)
    test_X, test_Y = dataset_to_parameter(testSet)
    RFR = GradientBoostingClassifier(max_depth=40, min_samples_leaf=2,min_samples_split=2, n_estimators=200,random_state=42)
    print('Start Training ' + setType + ' Model')
    start = time.time()
    RFR.fit(train_X, train_Y)
    end = time.time()
    print('trainSet score: {:.2%}'.format(RFR.score(train_X, train_Y)))
    print('testSet score: {:.2%}'.format(RFR.score(test_X, test_Y)))
    print('use Time:{0}'.format(end - start))
    print('Complete training')

    return RFR
    


if __name__ == "__main__":
    model = 1  # 0 (Classifier) or 1 (Regression)

    # for i in trange(1, 6):
    #     groupPath = src.format(i, setType)
    #     # groupPath = testSrc
    #
    #     trainSet_C, testSet_C, trainSet_R, testSet_R, trainSet_mix, testSet_mix = import_dataset(groupPath)
    #
    #     if MODEL[model] == 'Classifier':
    #         Au_XX_C_Model = HyperParameter_search_Classifier(trainSet_C, testSet_C, 'Au_XX_C')
    #         Au_XX_R_Model = HyperParameter_search_Classifier(trainSet_R, testSet_R, 'Au_XX_R')
    #         Au_XX_mix_Model = HyperParameter_search_Classifier(trainSet_mix, testSet_mix, 'Au_XX_mix')
    #     elif MODEL[model] == 'Regression':
    #         Au_XX_C_Model = HyperParameter_search_Regression(trainSet_C, testSet_C, 'Au_XX_C')
    #         Au_XX_R_Model = HyperParameter_search_Regression(trainSet_R, testSet_R, 'Au_XX_R')
    #         Au_XX_mix_Model = HyperParameter_search_Regression(trainSet_mix, testSet_mix, 'Au_XX_mix')
    #
    #     # output model
    #     joblib.dump(Au_XX_C_Model, './TestSet By Group{0}/model/Au_XX_C_{1}.pkl'.format(i,setType))
    #     joblib.dump(Au_XX_R_Model, './TestSet By Group{0}/model/Au_XX_R_{1}.pkl'.format(i,setType))
    #     joblib.dump(Au_XX_mix_Model, './TestSet By Group{0}/model/Au_XX_mix_{1}.pkl'.format(i,setType))
    #
    #     # output PMML model
    #     pipeline_C = PMMLPipeline([("Au_XX_C_Model", Au_XX_C_Model)])
    #     pipeline_R = PMMLPipeline([("Au_XX_R_Model", Au_XX_R_Model)])
    #     pipeline_mix = PMMLPipeline([("Au_XX_mix_Model", Au_XX_mix_Model)])
    #
    #     sklearn2pmml(pipeline_C, "./TestSet By Group{0}/model/Au_XX_C_{1}.pmml".format(i,setType))
    #     sklearn2pmml(pipeline_R, "./TestSet By Group{0}/model/Au_XX_R_{1}.pmml".format(i,setType))
    #     sklearn2pmml(pipeline_mix, "./TestSet By Group{0}/model/Au_XX_mix_{1}.pmml".format(i,setType))

    absPATH = os.getcwd()
    print(absPATH)
    # Set = pd.read_csv(testSrc)
    # train = Set.sample(frac=0.8,random_state=0,axis=0)
    # test = Set[~Set.index.isin(train.index)]

    TrainSet = pd.read_csv(testSrc)
    TestSet = pd.read_csv(testSrc2)

    # Au_XX_R_Model = HyperParameter_search_Regression(train, test, 'Au_XX_R')
    Au_XX_R_Model = HyperParameter_search_Classifier(TrainSet, TestSet, 'Au_XX_R')

    # Au_XX_R_Model = Train_Regression(train, test, 'Au_XX_R')
    # Au_XX_R_Model = Train_Regression(TrainSet,TestSet,'Au_XX_R')
    # Au_XX_R_Model = Train_GradientBoosting(TrainSet, TestSet, 'Au_XX_R')
    print(Au_XX_R_Model)