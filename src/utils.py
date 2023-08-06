from src.exception import CustomException
from src.logger import logging
import pickle
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix
import os,sys


def save_obj(file_path,obj):
    try:
        logging.info("Object saving started.")
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj) # file_obj as file_path 
        
        logging.info("Object saving completed.")

    except Exception as e:
        logging.info("Error to save object.")
        raise CustomException(e,sys)


def evaluate_model(models,X_train,X_test,y_train,y_test):
    try:
        logging.info("Model Evaluation Initiated.")
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(X_train,y_train)

            # Predict Testing Data
            y_pred = model.predict(X_test)
            test_model_score = accuracy_score(y_pred=y_pred,y_true=y_test)

            report[list(models.keys())[i]] = test_model_score

        return report
        # model_name = []
        # acc_score = []
        # prec_score = []
        # model_obj = []

        # for i in range(len(list(models))):
        #     model_name.append(list(models.keys())[i])

        #     model = list(models.values())[i]
        #     model_obj.append(model)

        #     model.fit(X_train,y_train)

        #     y_pred = model.predict(X_test)
        #     accuracy = accuracy_score(y_pred=y_pred,y_true=y_test)
        #     precision = precision_score(y_pred=y_pred,y_true=y_test)
        #     acc_score.append(accuracy)
        #     prec_score.append(precision)
        
        # max_accuracy_score = acc_score.index(max(acc_score))

        # logging.info("Model Evaluation Completed.")

        # return (
        #     model_obj[max_accuracy_score],
        #     model_name[max_accuracy_score],
        #     acc_score[max_accuracy_score],
        #     prec_score[max_accuracy_score]
        # )
        
    except Exception as e:
        logging.info("Error in Evaluation of model.")
        raise CustomException(e,sys)


def load_obj(obj_file_path):
    try:
        # logging.info("Initiate loading of model.")
        with open(obj_file_path,'rb') as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        logging.info("Error found in Loading of Object.")
        raise CustomException(e,sys) 



