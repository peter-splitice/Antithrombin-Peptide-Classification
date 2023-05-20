## Imports
import os
import sys

# Set current and parent directories
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to the current path
sys.path.append(parent_dir)

# Operating Ssytem
from peter_initializations import *

# Data Analysis
import matplotlib.pyplot as plt

# Tuning
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold


# Metrics
from sklearn.metrics import accuracy_score, make_scorer, matthews_corrcoef

# Optimization of hyperparameters for regression models using GridSearchCV
def hyperparameter_tuning(x, y, params, step, skf=StratifiedKFold(), model=SVC()):
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

    # Set up GridSearchCV with KFold Cross Validation
    clf = GridSearchCV(model, param_grid=params, scoring=make_scorer(matthews_corrcoef), cv=skf,
                       return_train_score=True, n_jobs=1)
    
    logger.info('GridSearchCV Starting:')
    logger.info('----------------------\n')

    clf.fit(x, y)
    model.set_params(**clf.best_params_)

    logger.info(f'Best parameter set: {clf.best_params_}')
    logger.info('----------------------------------------\n')

    return model

# Sequential Forward Selection stage of the pipeline.
def sequentail_selection(x, y, name, model=SVC()):
    """
    Perform Sequential Selection on the given dataset, but for the classifer portion of the 
        model.  MCC is the scorer used.  We perform both forward and backward selection.

    Parameters
    ----------
    x: Input values of the dataset.

    y: Output values for the different classes of the dataset.

    model: Model function used for Sequential Feature Selection.

    Returns
    -------
    final_x_sfs: Input values of the dataset with the proper number of features selected.

    final_sfs: The SequentialFeatureSelector model selected
    """  

    logger.info('Forward Selection Starting')
    logger.info('--------------------------\n')

    # Set the ratios for the number of features we want to extract from forward selection.
    ratios = np.arange(0.05, 0.55, 0.01)

    for ratio in ratios:
        sfs = SequentialFeatureSelector(model, n_jobs=-1, scoring=make_scorer(matthews_corrcoef),
                                        cv=skf, n_features_to_select=ratio, direction='forward')
        
        sfs.fit(x,y)
        x_sfs=pd.DataFrame(sfs.transform(x), columns=sfs.get_feature_names_out())

logger = log_files('clf.log')

# Step 1:  Import the data and models
x_train_full, x_test, y_train_full, y_test = import_data()
clf_zip = classification_models()

# Instantiate our StratifiedKFold object outside of our loop to pass into each stage.
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RAND)

for name, params, model in clf_zip:
    print(f'Model:{name}')

    # Step 2: Perform Hyperparameter Tuning and record the results
    
    model = hyperparameter_tuning(x_train_full, y_train_full, params, step='baseline', skf=skf, model=model)

    # Step 3: Perform Sequential Forward Selection on our model (for now we use brute force)

    print('test\n')

    
#    for train_index, valid_index in skf.split(x_train_full, y_train_full):
#        x_train, x_valid = x_train_full.iloc[train_index], x_train_full.iloc[valid_index]
#        y_train, y_valid = y_train_full.iloc[train_index], y_train_full.iloc[valid_index]


# Step 2: Apply Hyperparameter Tuning on all of our models.


# Step 3: Apply Forward Selection on our models.


# Step 4: Apply Hyperparameter Tuning again


# Step 5: 

print('break')