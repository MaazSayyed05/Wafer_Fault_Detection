import os,sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTraining

logging.info("Training Pipeline Initiated.")
# try:
#     train_data_path, test_data_path = DataIngestion.initiate_data_ingestion()
#     # logging.info("Train-Test Dataset path feteched successfully.")

#     preprcessed_train_data, preprcessed_test_data, preprocessor_obj_path = DataTransformation.initiate_data_transformation(train_data_path,test_data_path)
#     # logging.info("Data Transformaiton of Dataset Completed.")

#     Model_training = ModelTraining.initiate_model_training(preprcessed_train_data,preprcessed_test_data)


# except Exception as e:
#     logging.info("Error in training pipeline.")
#     raise CustomException(e,sys)


if __name__ == '__main__':
    data_ingestion_obj = DataIngestion()
    train_data_path, test_data_path = data_ingestion_obj.initiate_data_ingestion()

    data_transformation_obj = DataTransformation()
    preprcessed_train_data, preprcessed_test_data, preprocessor_obj_path = data_transformation_obj.initiate_data_transformation(train_data_path,test_data_path)

    model_trainer_obj = ModelTraining()
    model_trainer_obj.initiate_model_training(preprcessed_train_data,preprcessed_test_data)


