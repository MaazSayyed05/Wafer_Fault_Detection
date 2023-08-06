from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import  StandardScaler,RobustScaler
from sklearn.impute import  SimpleImputer


from sklearn.pipeline import  Pipeline
from sklearn.compose import  ColumnTransformer

import os,sys
from dataclasses import dataclass

# logging.info("Initiate Data Transformation.")
@dataclass 
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try: 
            # logging.info("Iniate Data Tranformation Pipeline.")
            preprocessor_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',RobustScaler())
                ]
            )

            # save_obj(self.data_transformation_config.preprocessor_obj_file_path,preprocessor_pipeline)
            logging.info("Data Transformation Pipeline Completed.")
            return preprocessor_pipeline



        except Exception as e:
            logging.info("Error found in Data Transformation Pipeline.")
            raise CustomException(e,sys)
        
    
    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            
            logging.info("Initiate Data Transformation.")
            train_data_set = pd.read_csv(train_data_path)
            test_data_set = pd.read_csv(test_data_path)

            logging.info("Read train-test data.")

            # logging.info("Obtain Data Transformation Object")
            logging.info("Iniate Data Tranformation Pipeline.")
            preprocessor = self.get_data_transformation_object()
            logging.info("Data Transformation Object fetched successfully.")

            # Define target feature, i/p feature of train and test,preprocessor on i/p feature train and test, np.array, np.c_(i/p,target_train_test)

            target_feature = 'Good/Bad'

            input_features_train_data = train_data_set.drop(target_feature,axis=1)  # X_train
            input_features_test_data  = test_data_set.drop(target_feature,axis=1)  # X_test
            
            target_feature_train_data = train_data_set[[target_feature]] # y_train
            target_feature_test_data  = test_data_set[[target_feature]]  # y_test

            logging.info("Dependent and Independent features extracted successfully.")

            input_features_train_data_array = preprocessor.fit_transform(input_features_train_data)
            input_features_test_data_array  = preprocessor.transform(input_features_test_data)

            prprocessed_train_data_set = np.c_[input_features_train_data,np.array(target_feature_train_data)]
            prprocessed_test_data_set = np.c_[input_features_test_data,np.array(target_feature_test_data)]

            logging.info("Data Transformation Complete.")

            save_obj(
                self.data_transformation_config.preprocessor_obj_file_path,
                preprocessor
            )

            logging.info("Preproceesor Pickle File created successfully.")
            return (prprocessed_train_data_set,prprocessed_test_data_set,self.data_transformation_config.preprocessor_obj_file_path)
        
        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e,sys)








