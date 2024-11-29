


import os.path as path
from get_sets import create_set ### create all sets (testing and training)
import generate_model as ttp ### Commun functions for the training and testing of a model

from utils import progress as prgs

import numpy as np
import pandas as pd
import json
import random
import time
import tqdm
from typing import Union, Optional

import multiprocessing

from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              ExtraTreesClassifier, AdaBoostClassifier)
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier


def read_json_file(file_path: str):
    '''Open and store a json file into a python dictionary
    
    :param file_path: The path leading to the json file 
    :type file_path: str 

    :return: The dictionray where the element of the inputed json file are stored. 
    Or None if the json file was empty or not existing at the given path
    :rtype: dict | None
    '''

    try:
        with open(file_path, 'r') as file:
            content = file.read().strip()
            if not content:
                return None
            return json.loads(content)
    except(FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading the JSON file: {e}")
        return None
    

    
def convert_sec_to_hh_mm_ss(sec: Union[int, float]):
    '''Convert a time in seconds into a hh:mm:ss formated time
    
    :param sec: The time in seconds
    :type sec: int | float 

    :return: The formated time as hh:mm:ss
    :rtype: str
    '''
    
    h = sec // 3600
    m = (sec % 3600) // 60
    s = round(sec % 60,3)
    return f'{int(h):02}:{int(m):02}:{s:02}'


def get_random_params_range():
    '''Create a dictionary of parameters for all models in the list 
    - 'RandomForest', 
    - 'ExtraTrees', 
    - 'CatBoost', 
    - 'GradientBoosting', 
    - 'SVM', 
    - 'AdaBoost', 
    - 'DecisionTree'
    
    :return: A dictionray of all parameters possible values for all implemented models
    :rtype: dict 
    '''

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 500, stop = 5000, num = 500)]

    # Number of features to consider at every split 
    max_features = ['sqrt', 'log2']

    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(5, 110, num = 30)]
    max_depth.append(None)

    max_depth_cat = range(17)

    # Minimum number of samples required to split a node
    min_samples_split = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4, 8]

    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    bootstrap_type_cat = ['Bayesian', 'Bernoulli', 'MVS', 'No']

    # A node will be split if this split induces a decrease of the impurity 
    # greater than or equal to this value
    min_impurity_decrease = [float(x) for x in np.linspace(0, 10, num = 100)]

    # The function to measure the quality of a split
    criterion = ['gini', 'entropy', 'log_loss']

    # The strategy used to choose the split at each node
    splitter = ['best', 'random']

    # Regularization parameter
    C_value = [float(x) for x in np.linspace(0.01, 5, num = 100)]

    # Specifies the kernel type to be used in the algorithm
    kernel = ['linear', 'poly', 'rbf', 'sigmoid']
    
    
    # Create the random grid
    random_grid = {
        'RandomForest':{
            'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'bootstrap': bootstrap
        },
        'ExtraTrees':{
            'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'bootstrap': bootstrap
        },
        'CatBoost':{
            'n_estimators': n_estimators,
            'max_depth': max_depth_cat,
            'min_child_samples': min_samples_split,
            'bootstrap_type': bootstrap_type_cat
        },
        'GradientBoosting':{
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'min_impurity_decrease': min_impurity_decrease
        },
        'AdaBoost':{
            'n_estimators': n_estimators
        },
        'DecisionTree':{
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'criterion': criterion,
            'splitter': splitter
        },
        'SVM':{
            'C': C_value,
            'kernel': kernel
        }
    }

    return random_grid



def get_random_param(random_grid: dict):
    '''Select random hyperparameters from the given grid
    
    :param random_grid: The dictionary where all keys are estimators parameters and 
    values are list of possible values
    :type random_grid: dict 

    :return: The ditionary of hyperparameter with randomly-selected values
    :rtype: dict
    '''

    result_grid = {}

    for key in random_grid.keys():
        nb_value = len(random_grid[key])
        random_value = random.randint(0,nb_value-1)
        result_grid[key] = random_grid[key][random_value]
    
    return result_grid


def get_best_grid(
        best_grid: dict, 
        current_grid: dict, 
        metric: str
        ):
    '''Compare the metric of a given model result for a specific parameter grid to a 
    reference model result. Get the best model based on the given metric for the specific 
    parameter grid
    
    :param best_grid: The reference "best" estimator, containing a 'model', a 'metric' and
    the parameters grid 'params' use to create it.
    :type best_grid: dict 

    :param current_grid: The estimator to compare to the reference one. Contains a 
    'model', a 'metric' and the parameters grid 'params' use to create it.
    :type current_grid: dict 

    :param metric: The metric to use for the comparison of the two models. 
    Can be: 'accuracy', 'auroc', 'recall', 'precision', 'f1_score', or 'mcc_score'
    :type metric: str 

    :return: The best model based on the comparison of the two 'metric' values
    :rtype: dict
    '''
    if best_grid == None:
        return current_grid
    elif current_grid['metrics'][metric] > best_grid['metrics'][metric]:
        return current_grid
    else:
        return best_grid





def get_grids_list(
        random_grid_dict: dict, 
        model_name: str, 
        grid_number: int
        ):
    '''Create a list of hyperparameter grids based on the parameter range dictionary provided
    
    :param random_grid_dict: The list of possible parameter and their allowed range 
    for all possible model
    :type random_grid_dict: dict 

    :param model_name: The model name that will use the created parameters
    :type model_name: str 

    :param grid_number: The number of hyperparameters grid to create
    :type grid_number: int 

    :return: The list of hyperparameter grids created for the given model
    :rtype: list
    '''

    res = []
    for i in range(grid_number):
        res.append(get_random_param(random_grid_dict[model_name]))
    return res




def get_grid_split(
        grids_list: list, 
        nb_cpu: int
        ):
    '''Split the list of grids in order to have an equal number of them for each used cpu
    
    :param grids_list: The list of grid to split
    :type grids_list: list 

    :param nb_cpu: The number of cpu available
    :type nb_cpu: int 

    :return: The splitted lists where all cpu should have the same number of grid
    :rtype: list
    '''

    chunk_size = len(grids_list) // nb_cpu
    if chunk_size == 0:
        chunk_size = 1
    chunks = [grids_list[i:i+chunk_size] for i in range(0, len(grids_list), chunk_size)]
    return chunks


def get_result_grid_list(
        grid_list: list, 
        grid_to_try: int, 
        model_name: str, 
        all_data: list, 
        nb_iteration: int, 
        metric: str, 
        ignored_features: list, 
        is_multiprocess: bool
        ):
    '''From a list of result grid, test them by creating models, 
    proceed to the training and testing. Return the best grid based on the provided metric
    
    :param grid_list: The list of grid to test
    :type grid_list: list 

    :param grid_to_try: The total number of grid to test
    :type grid_to_try: int 

    :param model_name: The name of the machine learning model to create
    :type model_name: str 

    :param all_data: The list containing positive and negative entries
    :type all_data: list 

    :param nb_iteration: The number of cross-validation to make for each grid 
    (1 iteration = 1 new train an test sets)
    :type nb_iteration: int 

    :param metric: The name of the metric to use for the result comparison and the selection 
    of the best model can be: "accuracy", "f1_score", "auroc" or "mcc_score"
    :type metric: str 

    :param ignored_features: The list of feature that will not be use for the training of the model
    :type ignored_features: list 

    :param is_multiprocess: Indicate if the function is call for a multiprocessing usage or not. 
    If not, iterations progress will be print in the console.
    :type is_multiprocess: bool 

    :return: The recap of the best condition based on the selected metric. Provide model name, 
    execution time, all metric results and the train and test sets used
    :rtype: dict
    '''
    
    best_grid = None
    if not is_multiprocess:
        prgs.print_loading_bar.start_time = time.time() # Init start time
    for i in range(len(grid_list)):
        grid_result = get_grid_results(
            model_name, 
            grid_list[i], 
            nb_iteration, 
            grid_to_try, 
            all_data, 
            ignored_features, 
            is_multiprocess, 
            current_grid_nb = i
            )
        best_grid = get_best_grid(best_grid, grid_result, metric)
    return best_grid



def get_grid_results(
        model_name: str, 
        grid: dict, 
        nb_iteration: int, 
        grid_to_try: int, 
        all_data: list, 
        ignored_features: list, 
        is_multiprocess: bool, 
        current_grid_nb: Optional[int] = None
        ):
    '''Create a model based on the provided hyperparameters and train/test it multiple times 
    
    :param model_name: The name of the estimator to create. Should belong to the list: 
    'RandomForest', 'ExtraTrees', 'CatBoost', 'GradientBoosting', 'SVM', 'AdaBoost', 'DecisionTree'
    :type model_name: str 

    :param grid: The grid of hyperparameters use to create the estimator
    :type grid: dict 

    :param nb_iteration: The number of iteration (Cross-Validation) to make in order to get final 
    average result. A new train/test set is randomly created for each iteration
    :type nb_iteration: int 

    :param grid_to_try: The total number of grid that will be generated during the all process
    :type grid_to_try: int 

    :param all_data: The list of positive and negative entries
    :type all_data: list 

    :param ignored_features: The list of feature that won't be used for the training of the model
    :type ignored_features: list 

    :param is_multiprocess: Indicate if the function is call for a multiprocessing usage or not. 
    If not, iterations progress will be print in the console.
    :type is_multiprocess: bool 

    :param current_grid_nb: The number of the current tested grid. If not provided, defaults to None 
    :type current_grid_nb: int | None

    :return: The recap of the best condition based on the selected metric. 
    Provide model name, execution time, all metric results and the train and test sets used
    :rtype: dict
    '''

    estimator = create_estimator(model_name, grid)

    #### Initialize result arrays
    grid_result = {
        'execution_time_train':[], 
        'execution_time_test':[],
        'metrics':{
            'accuracy':[], 
            'auroc':[],
            'recall':[],
            'precision':[],
            'f1_score':[],
            'mcc_score':[]
        },
    }

    
    for j in range(nb_iteration):
        if not is_multiprocess:
            prefix = f"Grid {current_grid_nb}/{grid_to_try} | Current iteration: {j}/{nb_iteration}"
            prgs.print_loading_bar(current_grid_nb, grid_to_try-1, prefix=prefix)
        
        X_train, y_train, X_test, y_test = create_set.get_train_test_sets(
                                                            all_data, 
                                                            0, 
                                                            1, 
                                                            pos_neg_ratio= '1:1', 
                                                            train_ratio= 0.7, 
                                                            ignored_features= ignored_features
                                                            )
        data = {
            'X_train':X_train,
            'y_train':y_train,
            'X_test':X_test,
            'y_test':y_test
        }

        results = ttp.train_test_model(estimator, data)
        grid_result = ttp.store_metrics(results, grid_result)
    
    # Get mean values from result lists
    mean_exec_train = np.mean(grid_result['execution_time_train'])
    mean_exec_test = np.mean(grid_result['execution_time_test'])

    mean_accuracy = np.mean(grid_result['metrics']['accuracy'])
    mean_auroc = np.mean(grid_result['metrics']['auroc'])
    mean_f1_score = np.mean(grid_result['metrics']['f1_score'])
    mean_mcc_score = np.mean(grid_result['metrics']['mcc_score'])

    current_grid = {'estimator':model_name, 
                    'execution_time_train':mean_exec_train,
                    'execution_time_test':mean_exec_test,
                    'metrics':{
                        'accuracy':mean_accuracy, 
                        'auroc':mean_auroc,
                        'f1_score':mean_f1_score,
                        'mcc_score':mean_mcc_score
                    }, 
                    'params':grid,
                    'conditions':{
                        'number_of_grids':grid_to_try,
                        'number_of_iter':nb_iteration
                    }
                }

    return current_grid
        

def create_estimator(
        estimator_name: str, 
        params: dict
        ):
    '''Create the wanted estimator with the given hyperparameters. 
    Estimators are created using sklearn, make sure the provided hyperparameters 
    fits the wanted estimator.
    
    :param estimator_name: The name of the estimator to create, can be: 'RandomForest', 
    'ExtraTrees', 'CatBoost', 'GradientBoosting', 'SVM', 'AdaBoost' or 'DecisionTree'
    :type estimator_name: str 

    :param params: The dictionary of hyperparameter to use for the creation of the estimator. 
    Make sure that the provided parameters fits the choosen estimator
    :type params: dict 

    :return: The wanted estimator created using the provided hyperparameters
    :rtype: RandomForestClassifier | ExtraTreesClassifier | 
    CatBoostClassifier | svm.SVC | AdaBoostClassifier | DecisionTreeClassifier
    
    '''
    accepted_estimator_names = [
        'RandomForest', 
        'ExtraTrees', 
        'CatBoost', 
        'GradientBoosting', 
        'SVM', 
        'AdaBoost', 
        'DecisionTree'
        ]
    
    assert estimator_name in accepted_estimator_names, (
        f"Invalid estimator name. Please use one from the list: {accepted_estimator_names}"
        )

    if estimator_name == 'RandomForest':
        estimator = RandomForestClassifier(random_state=0,**params)
    
    elif estimator_name == 'ExtraTrees':
        estimator = ExtraTreesClassifier(random_state=0,**params)

    elif estimator_name == 'CatBoost':
        estimator = CatBoostClassifier(random_state=0,verbose=False, **params)

    elif estimator_name == 'GradientBoosting':
        estimator = GradientBoostingClassifier(random_state=0, **params)

    elif estimator_name == 'SVM':
        estimator = svm.SVC(random_state=0, probability=True, **params)
    
    elif estimator_name == 'AdaBoost':
        estimator = AdaBoostClassifier(random_state=0, algorithm="SAMME", **params)

    elif estimator_name == 'DecisionTree':
        estimator = DecisionTreeClassifier(random_state=0, **params)

    return estimator

        




####################################################################################################
###########################################     MAIN     ###########################################
####################################################################################################


if __name__ == "__main__":
    #### Parameters:
    # Choose model from list:
    # RandomForest, ExtraTrees, CatBoost, GradientBoosting, SVM, AdaBoost, DecisionTree
    model_name = "RandomForest"

    # Work on multiprocesses
    # Split the job on the different available cpu. 
    multiprocess = False

    # Number Cross-Validation
    # Must stay the same during the all process of optimizing.
    # Cross-Beta predictor used 500 iteration for each tested grid
    number_iter = 5

    # Number of random grid to test
    # Can vary depending on your need and on the computation time needed for the different models
    grid_to_try = 2

    # Number of cpu
    # Only used if multiprocess is True
    process_count = multiprocessing.cpu_count()

    # Metric used for the result comparison
    # Can be 'accuracy', 'auroc', 'recall', 'precision', 'f1_score', or 'mcc_score'
    metric = 'f1_score'





    #### Paths ----------> May need modification depending on your usage. 
    # If you have difficulties with the paths, you can replace them by your actual absolute paths

    # Get current dir
    current_directory = path.dirname(__file__)

    # path to the positive and negative dataset (as a csv with spe = ';')
    relative_path_dataset = "data/set_ml_CSV_80_0.csv"
    path_dataset = path.join(current_directory, relative_path_dataset)

    relative_stored_report_path = "result/optimisation/opti_" + model_name + "_report.json"
    stored_report_path = path.join(current_directory, relative_stored_report_path)





    #### Feature selected
    # Final set (Cross-Beta DB and predictor publication)
    non_selected_AA_compo = ['R', 'H', 'K', 'E', 'S', 'G', 'P', 'A', 'V', 'X']
    non_selected_GRPcompo = ['grp_D', 'grp_P']
    non_selected_TRANSIcompo = ['A_to_B', 'A_to_D', 'A_to_P', 'A_to_G', 
                                'B_to_C', 'B_to_P', 
                                'C_to_D', 
                                'D_to_A', 'D_to_D', 'D_to_P', 
                                'P_to_A', 
                                'G_to_C', 'G_to_D', 'G_to_P', 'G_to_G']
    non_selected_features = non_selected_AA_compo + non_selected_GRPcompo + non_selected_TRANSIcompo





    #### Loading data
    data_all_csv = pd.read_csv(path_dataset,sep=';',header=0).fillna("NaN")
    filtered_df_amy = data_all_csv[data_all_csv['LABEL'] == 'AMYLOID'][['LABEL', 'Sequence']]
    filtered_df_amy['LABEL'] = filtered_df_amy['LABEL'].str.lower()
    filtered_df_amy = filtered_df_amy.rename(columns={'LABEL': 'label', 'Sequence': 'sequence'})

    filtered_df_idr = data_all_csv[data_all_csv['LABEL'] != 'AMYLOID'][['LABEL', 'Sequence']]
    filtered_df_idr['LABEL'] = filtered_df_idr['LABEL'].str.lower()
    filtered_df_idr = filtered_df_idr.rename(columns={'LABEL': 'label', 'Sequence': 'sequence'})

    # Convert the filtered DataFrame to a dictionary
    data_dict_positive = filtered_df_amy.to_dict(orient='records')
    data_dict_negative = filtered_df_idr.to_dict(orient='records')

    # Creating commun dataset
    all_data = [data_dict_positive, data_dict_negative]




    #### Optimisation procedure start

    print(f"Optimizing {model_name} for {grid_to_try} grids with {number_iter} iterations")
    #### Generating features
    print("Generating feature values...", end='\r', flush=True)
    time_start_features = time.time()
    for dataset in all_data:
        i = 0
        for entry in dataset:
            entry = ttp.get_features(entry, "charac_3", non_selected_features)
            if i == 0:
                feature_list = ttp.get_feature_list(entry)
            i += 1

    time_stop_features = time.time()
    total_time_features = convert_sec_to_hh_mm_ss(time_stop_features - time_start_features)
    print(f"features generated after {total_time_features} seconds")

    print("Training/Testing of the model...", end='\r', flush=True)


    #### Get previous stored result
    best_grid = read_json_file(stored_report_path)
    if best_grid:
        previous_nb_grid = best_grid['conditions']['number_of_grids']
    else:
        previous_nb_grid = 0


    #### Generated hyperparam grids
    random_grid_dict = get_random_params_range()

    # for model_name in model_list:
    grids_list = get_grids_list(random_grid_dict, model_name, grid_to_try)

    time_start_training = time.time()
    
    #### Run optimization with multiprocess TRUE
    if multiprocess:
        chunks = get_grid_split(grids_list, 100)
        print("\nNumber chunks: ",len(chunks))
        arguments_lists = []
        for i in range(len(chunks)):
            if i < 10:
                print(f"Number of grid in chunk {i}: {len(chunks[i])}")
            arguments_lists.append((
                chunks[i], 
                len(grids_list), 
                model_name, 
                all_data, 
                number_iter, 
                metric, 
                non_selected_features, 
                multiprocess
                ))
        
        with multiprocessing.Pool(processes=process_count) as pool:
            results = pool.starmap(
                get_result_grid_list, 
                tqdm.tqdm(
                    arguments_lists,
                    total=(len(arguments_lists))
                    )
                )
    
    #### Run optimization with multiprocess FALSE
    else:
        results = get_result_grid_list(
            grids_list, 
            len(grids_list), 
            model_name, 
            all_data, 
            number_iter, 
            metric, 
            non_selected_features, 
            multiprocess
            )

    time_stop_training = time.time()
    total_time_training = convert_sec_to_hh_mm_ss(time_stop_training - time_start_training)

    print(f"\nOptimisation finished after {total_time_training} seconds")

    if isinstance(results, dict):
        best_grid = get_best_grid(best_grid, results, metric)
    else:
        for result_grid in results:
            best_grid = get_best_grid(best_grid, result_grid, metric)
    
    best_grid['conditions']['number_of_grids'] = previous_nb_grid + grid_to_try

    with open(stored_report_path, 'w') as file:
        json.dump(best_grid, file, indent=4)

    
