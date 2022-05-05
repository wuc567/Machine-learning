"""
IMDB datasets demo for gcforestCS
Usage:
    define the model within scripts:
        python main_gcforestCS.py
    get config from json file:
        python main_gcforestCS.py --model imdb-gcForestCS.json

Description: A python 2.7 implementation of gcForestCS proposed in [1]. A demo implementation of gcForest library as well as some demo client scripts to demostrate how to use the code. The implementation is flexible enough for modifying the model or
fit your own datasets.
Reference: [1] M. Pang, K. M. Ting, P. Zhao, and Z.-H. Zhou. Improving deep forest by confidence screening. In ICDM-2018.  (http://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm18.pdf)
Requirements: This package is developed with Python 2.7, please make sure all the demendencies are installed, which is specified in requirements.txt
ATTN: This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Zhi-Hua Zhou(zhouzh@lamda.nju.edu.cn)
ATTN2: This package is developed by Mr. Ming Pang(pangm@lamda.nju.edu.cn), which is based on the gcForest package (http://lamda.nju.edu.cn/code_gcForest.ashx). The readme file and demo roughly explains how to use the codes. For any problem concerning the codes, please feel free to contact Mr. Pang.
"""
import sys
print(sys.path)
import argparse
import sys
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
sys.path.insert(0, "../lib")
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

from gcforest.DBCForest import DBCForest
from gcforest.gcforestCS import GCForestCS

from gcforest.utils.config_utils import load_json
from gcforest.utils.log_utils import get_logger
#from keras.datasets import cifar10
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn import datasets
from tensorflow.examples.tutorials.mnist import input_data
import time
import os
from sklearn.model_selection import KFold

import scipy.io as sio


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", type=str, default=None, help="gcforestCS Net Model File")
    args = parser.parse_args()
    return args


def get_config():
    config = {

           "net": {
                       "outputs": ["pool1/7x7/ets", "pool1/7x7/rf", "pool1/10x10/ets", "pool1/10x10/rf",
                                   "pool1/13x13/ets", "pool1/13x13/rf"],
                       "layers": [

                    {
                        "type": "FGWinLayer",
                        "name": "win1/7x7",
                        "bottoms": ["X", "y"],
                        "tops": ["win1/7x7/ets", "win1/7x7/rf"],
                        "n_classes": 10,
                        "estimators": [
                            {"n_folds": 3, "type": "ExtraTreesClassifier", "n_estimators": 30, "max_depth": 10,
                             "n_jobs": -1, "min_samples_leaf": 10},
                            {"n_folds": 3, "type": "RandomForestClassifier", "n_estimators":30, "max_depth": 10,
                             "n_jobs": -1, "min_samples_leaf": 10}
                        ],
                        "stride_x": 2,
                        "stride_y": 2,
                        "win_x": 2,
                        "win_y": 2
                    },
                           {
                               "type": "FGWinLayer",
                               "name": "win1/10x10",
                               "bottoms": ["X", "y"],
                               "tops": ["win1/10x10/ets", "win1/10x10/rf"],
                               "n_classes": 10,
                               "estimators": [
                                   {"n_folds": 3, "type": "ExtraTreesClassifier", "n_estimators": 30, "max_depth": 10,
                                    "n_jobs": -1, "min_samples_leaf": 10},
                                   {"n_folds": 3, "type": "RandomForestClassifier", "n_estimators": 30, "max_depth": 10,
                                    "n_jobs": -1, "min_samples_leaf": 10}
                               ],
                               "stride_x": 2,
                               "stride_y": 2,
                               "win_x": 3,
                               "win_y": 3
                           },
                           {
                               "type": "FGWinLayer",
                               "name": "win1/13x13",
                               "bottoms": ["X", "y"],
                               "tops": ["win1/13x13/ets", "win1/13x13/rf"],
                               "n_classes": 10,
                               "estimators": [
                                   {"n_folds": 3, "type": "ExtraTreesClassifier", "n_estimators": 30, "max_depth": 10,
                                    "n_jobs": -1, "min_samples_leaf": 10},
                                   {"n_folds": 3, "type": "RandomForestClassifier", "n_estimators": 30, "max_depth": 10,
                                    "n_jobs": -1, "min_samples_leaf": 10}
                               ],
                               "stride_x": 2,
                               "stride_y": 2,
                               "win_x": 4,
                               "win_y": 4
                           }
                           ,{
                               "type": "FGPoolLayer",
                               "name": "pool1",
                               "bottoms": ["win1/7x7/ets", "win1/7x7/rf", "win1/10x10/ets", "win1/10x10/rf",
                                           "win1/13x13/ets", "win1/13x13/rf"],
                               "tops": ["pool1/7x7/ets", "pool1/7x7/rf", "pool1/10x10/ets", "pool1/10x10/rf",
                                        "pool1/13x13/ets", "pool1/13x13/rf"],
                               "pool_method": "avg",
                               "win_x": 2,
                               "win_y": 2
                           }
                                 ]
            },


    "cascadeCS":{
        "random_state": 0,
        "max_layers": 100,
        "early_stopping_rounds": 2,
        "look_indexs_cycle": [[0,1],[2,3],[4,5]],
        "n_classes": 10,
        "estimators": [
            {"n_folds": 3, "type": "ExtraTreesClassifier", "n_estimators": 50, "max_depth": None, "n_jobs": -1},
            {"n_folds": 3, "type": "RandomForestClassifier", "n_estimators": 50, "max_depth": None, "n_jobs": -1}
        ],
        "part_decay": 2,
        "bin_size": 100,
        "estimators_enlarge":False,
        "keep_model_in_mem":False
    }
    }
    return config

def accuracy(y_true, y_pred):
    acc = 100. * np.sum(np.asarray(y_true) == y_pred) / max(len(y_true), 1.0)
    return acc

if __name__ == "__main__":

    args = parse_args()
    if args.model is None:
        config = get_config()
    else:
        config = load_json(args.model)






    data = sio.loadmat('../dataset/matData/imdb.mat')
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"].ravel(), data["y_test"].ravel()




    for layer in config["net"]["layers"]:
        layer["n_classes"] = len(np.unique(y_train))
    if len(X_train.shape)!=4:
        print("delete net")
        del config["net"]
        del config["cascadeCS"]["look_indexs_cycle"]
    config["cascadeCS"]["n_classes"] = len(np.unique(y_train))
    config_th = config



    model = DBCForest(config_th)
    a = model.fit_transform(X_train, y_train,X_test,y_test)

    model = GCForestCS(config_th)
    a = model.fit_transform(X_train, y_train, X_test, y_test)