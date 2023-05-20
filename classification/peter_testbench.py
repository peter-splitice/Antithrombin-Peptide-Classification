## Imports

# Operating Ssytem
import os
import sys

# Set current and parent directories
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to the current path
sys.path.append(parent_dir)

# Data Analysis
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Initializaitons
from peter_initializations import *

# Tuning
from sklearn.model_selection import GridSearchCV

# Data Processing
from sklearn.model_selection import train_test_split
from data_preprocessing_packages.feature_extraction import extract_feature
from sklearn.preprocessing import MinMaxScaler

# K-Fold Cross Validation
from sklearn.model_selection import StratifiedKFold

# Metrics
from sklearn.metrics import accuracy_score, make_scorer, matthews_corrcoef

# Global Variables
PATH = os.getcwd()
RAND = 42

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
    rbf_params = {}
    lin_params = {}
    log_params = {}
    rfc_params = {}
    knn_params = {}
    xgb_params = {}
    params = [rbf_params, lin_params, log_params, rfc_params, knn_params, xgb_params]

    # Names of the models for use later.
    names = ['SVC with RBF Kernel', 'SVC with Linear Kernel', 'Logistic Regression Classifier', 
             'Random Forest Classifier', 'KNN Classifier', 'XGBoost Classifier']
    
    # List of models.
    rbf_model = SVC(class_weight='balanced')
    lin_model = SVC(kernel = 'linear', class_weight='balanced')
    log_model = LogisticRegression(class_weight='balanced', max_iter=500)
    rfc_model = RandomForestClassifier(class_weight='balanced')
    knn_model = KNeighborsClassifier()
    xgb_model = XGBClassifier(scale_pos_weight=8.9038)
    models = [rbf_model, lin_model, log_model, rfc_model, knn_model, xgb_model]

    clf_zip = list(zip(names, params, models))

    return clf_zip

# Optimization of hyperparameters for regression models using GridSearchCV
def hyperparameter_optimizer(x, y, params, logger=log_files('clf.log'), model=SVC()):
    """
    This function is responsible for running GridSearchCV and opatimizing our hyperparameters.  I might need to fine-tune this.

    Parameters
    ----------
    x: Input values to perform GridSearchCV with.

    y: Output values to create GridSearchCV with.

    params: Dictionary of parameters to run GridSearchCV on.

    model: The model that we are using for GridSearchCV

    Returns
    -------
    bestparams: Optimized hyperparameters for the model that we are running the search on.
    """

    logger.debug('GridSearchCV Starting:')
    logger.debug('-----------------------\n')

    reg = GridSearchCV(model, param_grid=params, scoring=make_scorer(matthews_corrcoef), cv=5, 
                       return_train_score=True, n_jobs=-1)
    reg.fit(x,y)

    # Showing the best parameters found on the development set.
    logger.info('Best parameter set: %s' %(reg.best_params_))
    logger.info('-------------------------\n')

    # Save the best parameters.
    bestparams = reg.best_params_

    return bestparams

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

    with open(PATH + '/dependency.Scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    return x_train_full_mms, x_test_mms, y_train_full, y_test
# Optimization of hyperparameters for regression models using GridSearchCV
def hyperparameter_optimizer(x, y, params, model=SVC()):
    """
    This function is responsible for running GridSearchCV and opatimizing our hyperparameters.  I might need to fine-tune this.

    Parameters
    ----------
    x: Input values to perform GridSearchCV with.

    y: Output values to create GridSearchCV with.

    params: Dictionary of parameters to run GridSearchCV on.

    model: The model that we are using for GridSearchCV

    Returns
    -------
    bestparams: Optimized hyperparameters for the model that we are running the search on.
    """

    logger.debug('GridSearchCV Starting:')
    logger.debug('-----------------------\n')

    reg = GridSearchCV(model, param_grid=params, scoring=make_scorer(matthews_corrcoef), cv=5, 
                       return_train_score=True, n_jobs=-1)
    reg.fit(x,y)

    # Showing the best parameters found on the development set.
    logger.info('Best parameter set: %s' %(reg.best_params_))
    logger.info('-------------------------\n')

    # Save the best parameters.
    bestparams = reg.best_params_

    return bestparams


logger = log_files('clf.log')


# Step 1:  Import the data and models
x_train_full, x_test, y_train_full, y_test = import_data()
clf_zip = classification_models()

for name, params, model in clf_zip:
    print(f'Model:{name}')

    skf = StratifiedKFold(n_splits=5, shuffle=False, random_state=RAND)


# Step 2: Apply Hyperparameter Tuning on all of our models.


# Step 3: Apply Forward Selection on our models.


# Step 4: Apply Hyperparameter Tuning again


# Step 5: 

print('break')