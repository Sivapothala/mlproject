import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from xgboost import train

# from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRFRegressor

from src.exceptions import CustomExcepion
from src.logger import logging
from src.utils import *

@dataclass()
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_arrray):
        try:
            logging.info("Spliting training and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_arrray[:,:-1],
                test_arrray[:,-1]
                )

            models = {
                "Random Forest" : RandomForestRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "Linear Regression" : LinearRegression(),
                "KNN" : KNeighborsRegressor(),
                "XGBoost Regressor" : XGBRFRegressor(),
                # "CatBoost" : CatBoostRegressor(),
                "AdaBoost" : AdaBoostRegressor()
            } 


            model_report = evaluate_models(X_train=X_train,
                                          y_train=y_train,
                                          X_test=X_test,
                                          y_test=y_test,
                                          models= models)
            
            best_model_score = max(list(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomExcepion("No best Model",sys)
            logging.info(f"Best Model on training and testing dataset")


            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        obj=best_model)
            
            predicted = best_model.predict(X_test)
            return r2_score(y_test,predicted)
        except Exception as e:
            raise CustomExcepion(e,sys)
        



