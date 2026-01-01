# #REQUIRED LIBRARIES=>
# import os
# import sys
# import dill
#
# #ML LIBRARIES:
# import numpy as np
# import pandas as pd
# from catboost import CatBoostRegressor
# from sklearn.ensemble import (AdaBoostRegressor,
#                               GradientBoostingRegressor,
#                               RandomForestRegressor)
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# from sklearn.model_selection import GridSearchCV
# from sklearn.neighbors import KNeighborsRegressor
# from xgboost import XGBRegressor
# from sklearn.tree import DecisionTreeRegressor
#
# #LOCAL LIBRARIES:
# from src.exception import CustomException
#
# from src.exception import CustomException
#
#
# def save_objects(file_path, obj):
#     try:
#         dir_path = os.path.dirname(file_path)
#
#         os.makedirs(dir_path,exist_ok=True)
#         with open(file_path, 'wb') as file_obj:
#             dill.dump(obj, file_obj)
#
#     except Exception as e:
#         raise CustomException(e,sys)
#
# #FUNCTION FOR EVALUATING MODELS FROM model_trainer.py
# def evaluate_model(x_train,y_train,x_test,y_test,models,param):
#     try:
#         report = {}
#         for i in range(len(list(models))):
#
#             model = list(models.values())[i]
#             para = param[list(models.keys())[i]]
#
#             gs = GridSearchCV(model, para, cv=3)
#             gs.fit(x_train, y_train)
#             y_train_pred = gs.predict(x_train)#PREDICTION ON TRAINING DATASET
#
#             y_test_pred = gs.predict(x_test)#PREDICTION ON TEST DATASET
#
#             #R2 SCORE:
#             train_model_score = r2_score(y_train,y_train_pred)
#             test_model_score = r2_score(y_test, y_test_pred)
#
#             report[list(models.keys())[i]] = test_model_score
#             return report
#     except Exception as e:
#         raise CustomException(e,sys)


import os
import sys

import numpy as np
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_objects(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(x_train, y_train, x_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(x_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train, y_train)

            # model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(x_train)

            y_test_pred = model.predict(x_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
