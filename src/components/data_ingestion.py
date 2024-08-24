import os
import sys
from src.components import data_transformation
from src.exceptions import CustomExcepion
from src.logger import logging
import pandas as pd
logging.info("came")
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer,ModelTrainerConfig

@dataclass
class DataIngestionConfig :
    train_data_path: str = os.path.join('artifacts',"train.csv")
    test_data_path: str = os.path.join('artifacts',"test.csv")
    raw_data_path: str = os.path.join('artifacts',"data.csv")
    pass

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the Data Ingestion method or component")
        try:
            df=pd.read_csv('NoteBook/Data/stud.csv')
            logging.info("The Data Set read as Dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("train test split initiated")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of Data completed")


            return (self.ingestion_config.train_data_path,self.ingestion_config.test_data_path)

        except Exception as e:
            raise CustomExcepion(e,sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data= obj.initiate_data_ingestion()


    data_transformat= DataTransformation()
    train_data ,test_data,_ = data_transformat.initiate_data_transformation(train_data,test_data)


    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_array=train_data,test_arrray=test_data))
