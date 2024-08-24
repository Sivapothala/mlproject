import logging
import os
import pickle
from pyexpat import model
import dill
import sys
import pandas as pd
import numpy as np
from src import logger
from src.exceptions import CustomExcepion
from sklearn.metrics import r2_score

def save_object(file_path,obj):
    logging.info("Entered the utils save_object")
    try :
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file:
            dill.dump(obj,file)
    except Exception as e:
        raise CustomExcepion(e,sys)
    


def evaluate_models(X_train,y_train,X_test,y_test,models):
    try:
        logging.info("Entered the utils evaluate_model")
        report = {}

        model_values = list(models.values())
        model_keys = list(models.keys())
        for i in range(len(models)):
                model = model_values[i]
                model.fit(X_train,y_train)

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_model_score = r2_score(y_train,y_train_pred)
                test_model_score = r2_score(y_test,y_test_pred)

                report[model_keys[i]] = test_model_score
        return report
    except Exception as e:
         raise CustomExcepion(e,sys)