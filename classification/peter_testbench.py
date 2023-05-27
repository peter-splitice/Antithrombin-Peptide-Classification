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
def hyperparameter_tuning(x_train_full, y_train_full, params, skf=StratifiedKFold(), model=SVC()):
    """
    This function is responsible for running GridSearchCV and opatimizing our hyperparameters.  
    I might need to fine-tune this.

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

    clf.fit(x_train_full, y_train_full)
    model.set_params(**clf.best_params_)

    logger.info(f'Best parameter set: {clf.best_params_}')
    logger.info('----------------------------------------\n')

    # Train the model on the entire train set so we can export it.
    model.fit(x_train_full, y_train_full)

    return model

def kfold_cv(x_train_full, x_test, y_train_full, y_test, stage, skf=StratifiedKFold(), model=SVC()):
    """
    This function performs kfold cross validation on the model with all of our data to check results
        of the training, validation, and test sets.

    Parameters
    ----------
    x_train_full: All of the input values before splitting into train/validation sets.

    x_test: Untouched input values from our test set to check against the train/validation sets.

    y_train_full: All of the output values before splitting into train/validation sets.

    y_test: Untouched output values from our test set to check against the train/validation sets.

    Returns
    -------
    kfold_results_df: Dataframe containing the results from all 5 kfolds as well as their average.
    """
    # Initializations
    train_accuracy_sum = 0
    train_mcc_sum = 0
    valid_accuracy_sum = 0
    valid_mcc_sum = 0
    test_accuracy_sum = 0
    test_mcc_sum = 0
    fold_count = 0

    # Empty dataframe for the kfold results data.
    cols = ['Fold', 'Training Accuracy', 'Training MCC', 'Validation Accuracy', 'Validation MCC', 'Test Accuracy',
            'Test MCC']
    kfold_results_df = pd.DataFrame(columns=cols)

    # KFold implementation
    for train_index, valid_index in skf.split(x_train_full, y_train_full):
        # Split the "full" data into train/test sets.
        x_train, x_valid = x_train_full.iloc[train_index], x_train_full.iloc[valid_index]
        y_train, y_valid = y_train_full.iloc[train_index], y_train_full.iloc[valid_index]

        model.fit(x_train, y_train)

        # Predictions on the training set.
        y_train_pred = model.predict(x_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_mcc = matthews_corrcoef(y_train, y_train_pred)

        # Predictions on the validation set.
        y_valid_pred = model.predict(x_valid)
        valid_accuracy = accuracy_score(y_valid, y_valid_pred)
        valid_mcc = matthews_corrcoef(y_valid, y_valid_pred)

        # Predictions on the test set.
        y_test_pred = model.predict(x_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_mcc = matthews_corrcoef(y_test, y_test_pred)

        # Add to the sums of validation
        train_accuracy_sum += train_accuracy
        train_mcc_sum += train_mcc
        valid_accuracy_sum += valid_accuracy
        valid_mcc_sum += valid_mcc
        test_accuracy_sum += test_accuracy
        test_mcc_sum += test_mcc

        # Add to the counter, and then put our data into the dataframe.
        fold_count += 1
        kfold_results_df.loc[len(kfold_results_df)] = [fold_count, train_accuracy, train_mcc, valid_accuracy,
                                                       valid_mcc, test_accuracy, test_mcc]
    
    # Record the averages too.
    fold_count = f'{stage}'
    train_accuracy_avg = train_accuracy_sum/FOLDS
    train_mcc_avg = train_mcc_sum/FOLDS
    valid_accuracy_avg = valid_accuracy_sum/FOLDS
    valid_mcc_avg = valid_mcc_sum/FOLDS
    test_accuracy_avg = test_accuracy_sum/FOLDS
    test_mcc_avg = test_mcc_sum/FOLDS
    kfold_results_df.loc[len(kfold_results_df)] = [fold_count, train_accuracy_avg, train_mcc_avg, valid_accuracy_avg,
                                                   valid_mcc_avg, test_accuracy_avg, test_mcc_avg]
    
    return kfold_results_df

# Sequential Forward Selection stage of the pipeline.
def sequential_forward_selection(x_train_full, y_train_full, name, skf=StratifiedKFold(), model=SVC()):
    """
    Perform Sequential Selection on the given dataset, but for the classifer portion of the 
        model.  MCC is the scorer used.  We perform both forward and backward selection.

    Parameters
    ----------
    x_train_full: Input values of the dataset.

    y_train_full: Output values for the different classes of the dataset.

    model: Model function used for Sequential Feature Selection.

    Returns
    -------
    final_x_sfs: Input values of the dataset with the proper number of features selected.

    final_sfs: The SequentialFeatureSelector model selected
    """  

    logger.info('Forward Selection Starting')
    logger.info('--------------------------\n')

    # Set the ratios for the number of features we want to extract from forward selection.
    ratios = np.arange(0.05, 0.56, 0.10)
    valid_mcc_high = 0

    # Create a DataFrame to save the intermittent results of feature selection
    cols = ['Features Selected', 'Training Accuracy', 'Training MCC', 'Validation Accuracy', 'Validation MCC']
    scores_df = pd.DataFrame(columns=cols)

    for ratio in ratios:
        logger.info(f'Performing Forward Selection with ratio of {ratio}')
        logger.info('---------------------------------------------------\n')
        sfs = SequentialFeatureSelector(model, n_jobs=-1, scoring=make_scorer(matthews_corrcoef),
                                        cv=skf, n_features_to_select=ratio, direction='forward')
        
        sfs.fit(x_train_full,y_train_full)

        x_train_full_sfs = pd.DataFrame(sfs.transform(x_train_full), columns=sfs.get_feature_names_out())

        # Initialize the score measurements and select the final features selected based on best validation MCC.
        train_accuracy_sum = 0
        train_mcc_sum = 0
        valid_accuracy_sum = 0
        valid_mcc_sum = 0

        for train_index, valid_index in skf.split(x_train_full_sfs, y_train_full):
            # Split into train/validation sets and fit the model on the train set
            x_train, x_valid = x_train_full_sfs.iloc[train_index], x_train_full_sfs.iloc[valid_index]
            y_train, y_valid = y_train_full.iloc[train_index], y_train_full.iloc[valid_index]
            model.fit(x_train, y_train)

            # Predicting on the training set.
            y_train_pred = model.predict(x_train)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            train_mcc = matthews_corrcoef(y_train, y_train_pred)

            # Predicting on the validation set.
            y_valid_pred = model.predict(x_valid)
            valid_accuracy = accuracy_score(y_valid, y_valid_pred)
            valid_mcc = matthews_corrcoef(y_valid, y_valid_pred)

            # Add to the sums of validation
            train_accuracy_sum += train_accuracy
            train_mcc_sum += train_mcc
            valid_accuracy_sum += valid_accuracy
            valid_mcc_sum += valid_mcc
        
        # Calculate averages of the scores across folds for storage
        train_accuracy_avg = train_accuracy_sum/FOLDS
        train_mcc_avg = train_mcc_sum/FOLDS
        valid_accuracy_avg = valid_accuracy_sum/FOLDS
        valid_mcc_avg = valid_mcc_sum/FOLDS
        current_feature_count = x_train_full_sfs.shape[1]

        # If a particular ratio of features gives us the best results, we want to save the x values and the 
        # feature selector.
        if (valid_mcc_avg > valid_mcc_high):
            valid_mcc_high = valid_mcc_avg
            final_x_sfs = x_train_full_sfs
            final_sfs = sfs
            final_feature_count = current_feature_count

        scores_df.loc[len(scores_df)] = [final_feature_count, train_accuracy_avg, train_mcc_avg, valid_accuracy_avg,
                                         valid_mcc_avg]

    logger.info('Forward Selection Finished.')
    logger.info(f'Beat Feature Count:{final_feature_count}')
    logger.info('-----------------------------')

    scores_df.to_csv(PATH + f'/peter_classification/SFS Data/SFS for {name}.csv', index=False)
    
    with open(PATH + f'/peter_classification/Exports/{name}/SFS for {name}.pkl', 'wb') as f:
        pickle.dump(final_sfs, f)
        
    return final_x_sfs, sfs

def principal_component_analysis(x_train_full, x_test, y_train_full, y_test, name, params,
                                 skf=StratifiedKFold(), model=SVC(), stage="PCA"):
    """
    Function for performaing principal component analysis on our data.  Relies on the VARIANCES global constant
    to be an array consisting of the variances we want to try having PCA to account for.

    Parameters
    ----------
    x_train_full: Input values of the dataset.

    x_test: Input values for the test set.

    y_train_full: Output values for the different classes of the dataset.
    
    y_test: Output values for the test set.

    model: Model function used for Sequential Feature Selection. 

    name: Name of the model we're working.

    params: Dict of the parameters to hyperparameter tuning with.

    skf: StratifiedKFold object.

    model: Classifier model we are using.


    
    """

    logger.info(f'PCA Starting {stage}')
    logger.info('--------------------------\n')

    # Fit our instance of PCA and then save the file externally.
    pca = PCA()
    pca.fit(x_train_full)
    with open(PATH + f'/peter_classification/Exports/{name}/{stage} for {name}.pkl', 'wb') as f:
        pickle.dump(pca, f)

    valid_mcc_high = 0

    for variance in VARIANCES:
        # Determine the number of principal components we need to account for our chosen variance.
        ratios = np.array(pca.explained_variance_ratio_)
        ratios = ratios[ratios.cumsum() <= (variance/100)]

        # Apply PCA on both the train/test sets 
        x_train_full_pca = pd.DataFrame(pca.transform(x_train_full))
        x_test_pca = pd.DataFrame(pca.transform(x_test))

        # Readjust the dimensions of x based on the variance we want.
        n_components = len(ratios)
        if n_components > 0:
            logger.info('Selecting %i principal component making up %i%% of the variance:\n' %(n_components, variance))
            logger.info('-----------------------------------------------------------------------\n')

            # Apply the PCA transformation now that we know how many principal components we need.
            x_train_full_pca = x_train_full_pca[x_train_full_pca.columns[0:n_components]]
            x_test_pca = x_test_pca[x_test_pca.columns[0:n_components]]
        else:
            logger.info('Kept all principal components for %i%% of the variance.\n' %(variance))

        var_stage = stage + (' with %i%% variance' %(variance))

        # Run Hyperparameter tuning and k-fold cross validation for each variance.
        model_var, pca_var_results = clf_trainer(x_train_full_pca, x_test_pca, y_train_full, y_test, params, name,
                                                 var_stage, skf, model)
        
        valid_mcc = float(pca_var_results['Validation MCC'][pca_var_results['Fold']==f'{var_stage}'])

        # If the 
        if valid_mcc > valid_mcc_high:
            valid_mcc_high = valid_mcc
            best_variance = variance
            final_pca_var_results = pca_var_results
            final_model_var = model_var
            final_var_stage = var_stage

    with open(PATH + f'/peter_classification/Exports/{name}/{name} model for {final_var_stage} at {best_variance}%.pkl',
             'wb') as f:
        pickle.dump(final_model_var, f)

    return final_model_var, final_pca_var_results
        

def clf_trainer(x_train_full, x_test, y_train_full, y_test, params, name, stage,
                skf = StratifiedKFold(), model=SVC()):
    """
    Mini-pipeline where we do hyperparameter optimization as well as k-fold cross-validtion to log our results
    into a dataframe.

    Parameters
    ----------
    x_train_full: The full training x-values before k-fold CV.  May be modified from PCA, SFS, or both

    x_test: The full test x-values. May be modified from PCA, SFS, or both.

    y_train_full: The full training y-values before k-fold CV

    y_test: The full test y-values.
    
    params: Dict of the parameters to hyperparameter tune with.

    name: Name of the model we're working.

    stage: Stage of the model pipeline (Baseline, PCA, SFS, SFS + PCA)

    skf: The SequentialFeatureSelector object

    model: The model object

    Returns
    -------

    """

    logger.info(f'{stage} hyperparameter tuning and K-Fold Cross-Validation beginning:')
    logger.info('-------------------------------------------------------------------\n')
    model = hyperparameter_tuning(x_train_full, y_train_full, params, skf, model)
    results_df = kfold_cv(x_train_full, x_test, y_train_full, y_test, stage, skf, model)

    results_df.to_csv(PATH + f'/peter_classification/Results/{name}/{name} {stage} KFold Results.csv',
                      index=False)

    return model, results_df
        

logger = log_files('clf.log')

# Step 1:  Import the data and models
x_train_full, x_test, y_train_full, y_test = import_data()
clf_zip = classification_models()

# Instantiate our StratifiedKFold object outside of our loop to pass into each stage.
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RAND)

for name, params, model in clf_zip:
    logger.info(f'Performing operations for {name}:')
    logger.info('-------------------------------------------\n')

    # Step 2: Perform Hyperparameter Tuning and record the results
    model, baseline_results = clf_trainer(x_train_full, x_test, y_train_full, y_test, params,
                                          name, stage='Baseline', skf=skf, model=model)
    with open(PATH + f'/peter_classification/Exports/{name}/Baseline model for {name}.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Step 3: Perform PCA on our model
    pca_model, pca_results = principal_component_analysis(x_train_full, x_test, y_train_full, y_test, name, params, skf,
                                             model, stage='PCA')

    # Step 4: Perform Sequential Forward Selection on our model
    x_train_full_sfs, sfs = sequential_forward_selection(x_train_full, y_train_full, name, skf, model)
    x_test_sfs = pd.DataFrame(sfs.transform(x_test), columns=sfs.get_feature_names_out())
    sfs_model, sfs_results = clf_trainer(x_train_full_sfs, x_test_sfs, y_train_full, y_test, params,
                                         name, stage='SFS', model=model)

    # Step 5: Perform PCA on the Forward selected model.
    sfs_pca_model, sfs_pca_results = principal_component_analysis(x_train_full_sfs, x_test_sfs, y_train_full, y_test, name,
                                                                  params, skf, sfs_model, stage='SFS-PCA')

    # Conslidate our results:
    cols = ['Stage', 'Training Accuracy', 'Training MCC', 'Validation Accuracy', 'Validation MCC',
            'Test Accuracy', 'Test MCC']
    results_df = pd.DataFrame(columns=cols)
    # Add the last row of each DataFrame into the consolidated reuslts dataframe.
    for results in [baseline_results, pca_results, sfs_results, sfs_pca_results]:
        new_row = {'Stage': results['Fold'][5], 
                   'Training Accuracy': results['Training Accuracy'][5],
                   'Training MCC': results['Training MCC'][5],
                   'Validation Accuracy': results['Validation Accuracy'][5],
                   'Validation MCC': results['Validation MCC'][5],
                   'Test Accuracy': results['Test Accuracy'][5],
                   'Test MCC': results['Test MCC'][5]}
        results_df.loc[len(results_df)] = new_row
    
    results_df.to_csv(PATH + f'/peter_classification/Results/Consolidated results for {name}.csv', index=False)