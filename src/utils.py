import logging
import os
import pickle
import dill
import sys
import pandas as pd
import numpy as np
from src import logger
from src.exceptions import CustomExcepion

def save_object(file_path,obj):
    logging.info("Entered the utils save_object")
    try :
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file:
            dill.dump(obj,file)
    except Exception as e:
        raise CustomExcepion(e,sys)