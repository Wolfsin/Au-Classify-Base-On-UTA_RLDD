# Au-Classify-Base-On-UTA_RLDD


### 前言
这是我大学院毕业时的毕业研究，利用OpenFace获取Action Units，借此来识别早期疲劳。因为代码经过了多次的重构，所以下面我将来详细说明怎么使用这个项目。（日本語は下にある・English is below.）

## 快速开始

1. 下载本项目到本地
2. 下载UTA_RLDD数据集(https://sites.google.com/view/utarldd/home)
3. 使用OpenFace解析数据集获得AU文件
4. 配置本地环境：
    使用到的包请查看requirements.txt
5. 根据自己的AU文件修改代码/按照我的配置修改你的文件路径
6. 运行 TrainModel-General.py

## 文件路径

AU文件的路径示例：

```
./Group1/01/0_Au_XX_C_ALLSet.csv
./Group1/01/0_Au_XX_mix_ALLSet.csv
./Group1/01/0_Au_XX_R_ALLSet.csv
```

解释：

```
./组别(与RLDD相对应)/编号(与RLDD相对应)/标签(动画的标签)_Au_XX_强度还是存在还是混合_ALLSet.csv
```

## 项目下各个包的解释
```
│  corrcoef.py （相关计算）
│  corrcoef_concat.py （相关计算结果的合成）
│  CreateFolder.py （生成一开始的文件夹结构）
│  dataTools.py （数据导入和处理，最终版已废弃）
│  GenerateTTSet.py （生成训练集和测试集，最终版已废弃）
│  GetAllSet.py （通过OpenFace的输出csv文件中获取只有AU的数据集合）
│  GetAverageSet.py （平均前处理，已整合进最终版）
│  GetHighMappingSet.py （扩张前处理，已整合进最终版）
│  GetShap.py （对训练的模型和测试集计算SHAP）
│  LoadShap.py （输出SHAP的图像/ 计算Permutation Importance）
│  TrainModel-General - All video analysis LSTM.py （训练长LSTM模型）
│  TrainModel-General - Short time analysis LSTM.py （训练短LSTM模型）
│  TrainModel-General - XGBoost.py （训练XGBTree模型）
│  TrainModel-General.py （训练RF模型）
```

最终版只需要在获得`AllSet`之后，运行训练模型的`.py`文件即可，前处理等一系列操作都包含在`TrainModel-General`文件下

# 日本語説明

### はじめに

これは大学院を卒業した時の卒業研究で、OpenFaceを利用してAction Unitsを取得することで、早期疲労を識別します。 コードは何度もリファクタリングされているので、このプロジェクトの使い方を詳しく説明します。

## クイックスタート

1. 本プロジェクトをローカルにダウンロードする
2. UTA_RLDDデータセットのダウンロード(https://sites.google.com/view/utarldd/home)
3. OpenFace解析データセットを使用してAUファイルを取得する
4. ローカル環境の構成:
    使用するパッケージはrequirements.txtを参照してください
5. 自分のAUファイルに基づいてコードを修正する／私の構成に従ってあなたのファイルパスを修正する
6. TrainModel-General.pyを実行する

## ファイルパス

AUファイルへのパスの例:

```
./Group1/01/0_Au_XX_C_ALLSet.csv
./Group1/01/0_Au_XX_mix_ALLSet.csv
./Group1/01/0_Au_XX_R_ALLSet.csv
```

説明:

```
./グループ(RLDDに同じ)/人番号(RLDDに同じ)/ラベル(動画のラベル)_Au_XX_強度・存在・統合したもの_ALLSet.csv
```

## プロジェクトの各ファイルの説明

```
│  corrcoef.py （相関係数の計算）
│  corrcoef_concat.py （相関係数の計算結果を統合）
│  CreateFolder.py （最初のフォルダ構造を生成）
│  dataTools.py （データのインポートと処理関連するモジュール、最終版は廃棄）
│  GenerateTTSet.py （トレーニングセットとテストセットを生成、最終版は廃棄）
│  GetAllSet.py （OpenFaceの出力csvファイルからAUのみのデータセットを取得）
│  GetAverageSet.py （平均の前処理、最終版に統合済み）
│  GetHighMappingSet.py （拡張の前処理、最終版に統合済み）
│  GetShap.py （トレーニング済みのモデルとテストセットに対してSHAPを計算）
│  LoadShap.py （SHAPの画像を出力／Permutation Importanceを計算）
│  TrainModel-General - All video analysis LSTM.py （長時間分析LSTMモデルの訓練）
│  TrainModel-General - Short time analysis LSTM.py （短時間分析LSTMモデルの訓練）
│  TrainModel-General - XGBoost.py （XGBTreeモデルの訓練）
│  TrainModel-General.py （RFモデルの訓練）
```

最終版は`AllSet`を獲得した後、訓練の`.py`ファイルを実行すれば良い、前処理などの一連の操作は`TrainModel-General`ファイルに**統合済み**。

# English Version

### Before the start
This is my graduation research when I graduated from master's degree. I use OpenFace to get Action Units to identify early fatigue. Because the code has been refactored many times, I will explain in detail how to use this project next.

## Quick start

1. Download this project to your local

2. Download UTA_RLDD dataset (https://sites.google.com/view/utarldd/home)

3. Use OpenFace to parse the dataset to obtain the AU file.

4. Configure the local environment:
    See `requirements.txt` for the package used.

5. Modify the code according to your AU file or modify your file path according to my configuration.

6. Run `TrainModel-General.py`

## File path

Path example of AU file：

```
./Group1/01/0_Au_XX_C_ALLSet.csv
./Group1/01/0_Au_XX_mix_ALLSet.csv
./Group1/01/0_Au_XX_R_ALLSet.csv
```

expound：

```
./group(same as RLDD)/people number(same as RLDD)/label(label for video)_Au_XX_(intensity or exist or mix)_ALLSet.csv
```

## Explanation of each file under the project
```
│  corrcoef.py (calculation of correlation)
│  corrcoef_concat.py (Synthesis of related calculation results)
│  CreateFolder.py (Generate the initial folder structure)
│  dataTools.py (Data import and processing, the final version has been obsolete.)
│  GenerateTTSet.py (Training set and test set are generated, and the final version is obsolete.)
│  GetAllSet.py (Get the data set with only AU from the output csv file of OpenFace.)
│  GetAverageSet.py (Average pre-processing has been integrated into the final version.)
│  GetHighMappingSet.py (Pre-processing for expansion has been integrated into the final version.)
│  GetShap.py (Calculate SHAP for the trained model and test set.)
│  LoadShap.py (Output the image of SHAP & Calculate Permutation Importance)
│  TrainModel-General - All video analysis LSTM.py (Training long Time LSTM model)
│  TrainModel-General - Short time analysis LSTM.py (Training short Time LSTM model)
│  TrainModel-General - XGBoost.py (Training XGBTree model)
│  TrainModel-General.py (Training RF model)
```

The final version only needs to run the `py` file of the training model after obtaining `AllSet`, and a series of operations such as pre-processing are included in the `training model-general`file.
