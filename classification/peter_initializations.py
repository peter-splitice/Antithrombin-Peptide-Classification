
import sys
import logging
import os
import pickle

# Basic imports
import pandas as pd
import numpy as np

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from data_preprocessing_packages.feature_extraction import extract_feature

# Models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Global Constants
PATH = os.getcwd()
RAND = 42
FOLDS = 5
VARIANCES = [75, 80, 85, 90, 95, 100]

# Creating a logger to record and save information.
def log_files(logname):
    """
    Create the meachanism for which we log results to a .log file.

    Parameters
    ----------
    logname:

    Returns
    -------
    logger:  The logger object we create to call on in other functions. 
    """

    # Instantiate the logger and set the formatting and minimum level to DEBUG.
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    # Display the logs in the output
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)

    # Write the logs to a file
    file_handler = logging.FileHandler(logname)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Adding the file and output handlers to the logger.
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    return logger

# Data Import + MinMaxScaler transformation
def import_data():
    """Import the dataset and apply MinMaxScaler transformation to it first.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    x_train_full_mms: 80% of the x-data, before k-fold. MinMax Scaled.

    x_test_mms: 20% of the x-data for our test set. MinMax Scaled.

    y_train_full: 80% of the y-data, before k-fold.

    y_test: 20% of the y-data for our test set.

    """

    # Import dataframes
    positive = pd.read_csv('data/Positive data.csv')
    negative = pd.read_csv('data/Negative data.csv')

    # Extract features
    positive_data = extract_feature(positive['Seq'])
    negative_data = extract_feature(negative['Seq'])

    # Add targets
    positive_data['Class'] = 1
    negative_data['Class'] = 0

    positive_data['SeqLength_bins'] = pd.qcut(positive_data['SeqLength'], q=5)
    negative_data['SeqLength_bins'] = pd.qcut(negative_data['SeqLength'], q=5)

    full_data = pd.concat([positive_data, negative_data])

    # Split the dataset:
    x_train_full, x_test, y_train_full, y_test = train_test_split(full_data, full_data['Class'], test_size=0.2, 
                                                                  stratify=full_data[['SeqLength_bins', 'Class']],
                                                                  random_state=RAND)

    # Clean the X data by removing seqlength and class
    x_train_full = x_train_full.drop(columns=['SeqLength_bins', 'Class'])
    x_test = x_test.drop(columns=['SeqLength_bins', 'Class'])

    ## Apply MinMaxScaler to the data and save.
    scaler = MinMaxScaler()

    x_train_full_mms = pd.DataFrame(scaler.fit_transform(x_train_full), index=x_train_full.index,
                                    columns=x_train_full.columns)
    x_test_mms = pd.DataFrame(scaler.transform(x_test), index=x_test.index, columns=x_test.columns)

    with open(PATH + '/peter_classification/Exports/Scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    return x_train_full_mms, x_test_mms, y_train_full, y_test

# Create the classification models.
def classification_models():
    """
    Generate the base models with parameters and put them together into a list
    
    Parameters
    ----------
    None
    
    Returns
    -------
    clf_models:  Zipped list for our models.  Contains names, hyperparameters, and the models.
    """

    # Hyperparameters to tune.
    rbf_params = {'gamma': [1e-1,1e-2,1e-3,1e-4,'scale','auto'], 'C': np.arange(1,101,5),
                  'class_weight': [None,'balanced'], 'break_ties': [False,True]}
    rfc_params = {'criterion': ['gini','entropy'], 'max_features': ['sqrt','log2',1.0,0.3], 
                  'ccp_alpha': np.arange(0,0.3,0.1), 'n_estimators': np.arange(1,25,1), 'max_depth': np.arange(2,11,1)}
    knn_params = {'n_neighbors': np.arange(1,55,2), 'weights': ['uniform', 'distance'], 'leaf_size': np.arange(5,41,2),
                  'p': [1, 2], 'keepdims': [False,True]}
    xgb_params = {'max_depth': np.arange(2,11,1), 'n_estimators': np.arange(1,25,1), 'gamma': np.arange(0,4,1),
                  'subsample': [0.5,1], 'lambda': [1,5,9], 'alpha': np.arange(0,1.1,0.2)}
    params = [rbf_params, rfc_params, knn_params, xgb_params]

    # Names of the models for use later.
    names = ['SVC with RBF Kernel', 'SVC with Linear Kernel', 'Logistic Regression Classifier', 
             'Random Forest Classifier', 'KNN Classifier', 'XGBoost Classifier']
    
    # List of models.
    rbf_model = SVC(class_weight='balanced')
    #lin_model = SVC(kernel = 'linear', class_weight='balanced')
    #log_model = LogisticRegression(class_weight='balanced', max_iter=500)
    rfc_model = RandomForestClassifier(class_weight='balanced')
    knn_model = KNeighborsClassifier()
    xgb_model = XGBClassifier(scale_pos_weight=8.9038)
    models = [rbf_model, rfc_model, knn_model, xgb_model]

    clf_zip = list(zip(names, params, models))

    return clf_zip