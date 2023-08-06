# import os,sys
# import pandas as pd
# import numpy as np
# from src.exception import CustomException
# from src.logger import logging
# from dataclasses import dataclass
# from sklearn.model_selection import train_test_split

# @dataclass
# class DataIngestionConfig:
#     raw_data_path = os.path.join('artifacts','raw.csv')
#     train_data_path = os.path.join('artifacts','train.csv')
#     test_data_path = os.path.join('artifacts','test.csv')
#     logging.info("DataIngestionConfig Class.")


# class DataIngestion:
#     def __init__(self):
#         logging.info("DataIngestion Class.")
#         self.data_ingestion_config = DataIngestionConfig()
# # D:\PW_DS\Machine_Learning\Wafer_Fault_Deteciton_2\notebooks\data\wafer_preprocess.csv
#     def initiate_data_ingestion(self):
#         logging.info("Initiated Data Ingesiton.")
#         try: 
#             dataset_df = pd.read_csv(os.path.join('notebooks/data','wafer_preprocess.csv'))
#             logging.info("Dataset imported as DataFrame.")

#             os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path),exist_ok=True)
#             dataset_df.to_csv(self.data_ingestion_config.raw_data_path,header=True)

#             logging.info("Raw data file created successfully.Train-Test Split")

#             train_set, test_set = train_test_split(dataset_df,test_size=0.20,random_state=42)

#             train_set.to_csv(self.data_ingestion_config.train_data_path,header=True,index=False)
#             test_set.to_csv(self.data_ingestion_config.test_data_path,header=True,index=False)      

#             logging.info("Data Ingestion Successful.")

#             return(self.data_ingestion_config.train_data_path,self.data_ingestion_config.test_data_path)

#         except Exception as e:
#             logging.info("Error in Data Ingestion.")
#             raise CustomException(e,sys)

    


# ------------------------------------------------------------------


# This will return train and test data
import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Initialize the data ingestion config.abs
# Corrected DataIngestionconfig class
try: 
    @dataclass
    class DataIngestionconfig:
        train_data_path: str = os.path.join('artifacts', 'train.csv')
        test_data_path: str = os.path.join('artifacts', 'test.csv')
        raw_data_path: str = os.path.join('artifacts', 'raw.csv')


    # Create a data ingestion class
    class DataIngestion:
        def __init__(self):
            self.ingestion_config = DataIngestionconfig()
        
        def initiate_data_ingestion(self):
            logging.info("Data Ingestion Method Starts")

            try:
                df = pd.read_csv(os.path.join('notebooks\data','wafer_preprocess.csv'))
                logging.info("Dataset read as pandas DataFrame")

                os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
                df.to_csv(self.ingestion_config.raw_data_path,index=False)

                logging.info("Train Test Split")

                train_set, test_set = train_test_split(df,test_size=0.20,random_state=42)

                train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
                test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

                logging.info("Ingestion of Data is completed")

                return (
                    self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path
                    )


            except Exception as e:
                logging.info('Error occured in Data Ingestion Config')


except Exception as e:
    raise CustomException(e,sys)












