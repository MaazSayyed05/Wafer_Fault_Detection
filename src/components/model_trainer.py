from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj
import os,sys
from sklearn.metrics import  classification_report,accuracy_score,confusion_matrix,precision_score
from sklearn.linear_model import  LogisticRegression
from sklearn.naive_bayes import  GaussianNB
from sklearn.tree import  DecisionTreeClassifier
# from sklearn.neighbors import  KNeighborsClassifier
from sklearn.ensemble import  RandomForestClassifier

from dataclasses import dataclass

from src.utils import evaluate_model

class ModelTrainingConfig:
    model_file_path = os.path.join('artifacts','model.pkl')

class ModelTraining:
    def __init__(self):
        self.model_training = ModelTrainingConfig()
    
    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info("Model Training Initiated.")
            X_train,X_test,y_train,y_test = train_arr[:,:-1],test_arr[:,:-1],train_arr[:,-1],test_arr[:,-1]

            models = {
                'GaussianNaiveBayes'     : GaussianNB(),
                'RandomForestClassifier' : RandomForestClassifier(),
                'LogisticRegression'     : LogisticRegression(),
                'DecisionTreeClassifier' : DecisionTreeClassifier()
            }


            # best_model, best_model_name, best_acc_score, best_prec_score = evaluate_model(models,X_train,X_test,y_train,y_test)

            # logging.info("Best Model: ")
            # logging.info(best_model_name)
            model_report:dict = evaluate_model(models,X_train,X_test,y_train,y_test)
            # print(model_report)
            # print("\n")
            # print("="*40)

            logging.info(f"Model Report :{model_report}")

            # To get best model score from dict.
            best_model_score = max(sorted(model_report.values()))

            best_model_name  = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            # print(f"Best Model Found, Model Name:{best_model_name}, R2 Score: {best_model_score}")
            # print("="*40)
            logging.info(f"Best Model Found, Model Name:{best_model_name}, Acuracy Score: {best_model_score}")

            save_obj(
                file_path=self.model_training.model_file_path,
                obj=best_model
            )
            logging.info("Pickle File of Model created Successfully.")


        except Exception as e:
            logging.info("Error in Model Training.")
            raise CustomException(e,sys)
    


