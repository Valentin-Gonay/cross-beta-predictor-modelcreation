# Train and Test a model with the given dataset

from feature_creation import aa_compo as compo ### Get AA compo, Group compo and transition compo
from get_sets import create_set ### create all sets (testing and training)
from stats import eval_metrics as metrics ### Get evaluation score (recall, precision, F1 score, MCC)


import feature_creation.iupred3.iupred3_lib as iupred ### get structural state of a sequence


import os.path as path
import numpy as np
import pandas as pd
import time
import pickle


from sklearn.metrics import roc_auc_score, accuracy_score

from sklearn.ensemble import ExtraTreesClassifier





def get_features(
        entry: dict, 
        classification_mode: str, 
        ignored_features: list = []
        ):
    '''Compute all features for the given entry
    
    :param entry: A dataset entry containing a 'sequence' key with a sequence composed of 
    the 20 essential amino acids
    :type entry: dict 

    :param classification_mode: The classification mode used to create an alternative sequence 
    based on amino acid caracteristics (can be "charac_1" or "charac_2")
    :type classification_mode: str 

    :param ignored_features: The list of feature that will not be use for the training of the model.
    If not provided, default to []
    :type ignored_features: list 

    :return: The updated entry with a 'features' key containing the list of features
    :rtype: dict
    '''

    sequence = entry['sequence']
    grp_sequence = compo.aa_classification_seq(sequence, classification_mode)

    entry['features'] = {}

    ### Composition features
    aa_composition = compo.get_aa_compo(sequence) # dict
    grp_composition = compo.get_group_compo(grp_sequence, classification_mode) # dict
    transi_composition = compo.get_group_transition(grp_sequence, classification_mode) # dict
    
    ### Structural states features
    if "iupred_score" not in ignored_features:
        iupred_score = np.mean(iupred.iupred(sequence, mode='short', smoothing='strong')[0])
        entry['features']['iupred_score'] = iupred_score

    ### Import only features linked to composition if they are not in the "ignored_features" list
    for aa_compo in aa_composition.keys():
        if aa_compo not in ignored_features:
            entry['features'][aa_compo] = aa_composition[aa_compo]
    for grp_compo in grp_composition.keys():
        if grp_compo not in ignored_features:
            entry['features'][grp_compo] = grp_composition[grp_compo]
    for transi_compo in transi_composition.keys():
        if transi_compo not in ignored_features:
            entry['features'][transi_compo] = transi_composition[transi_compo]
    
    return entry


def get_feature_list(entry: dict):
    '''Get the feature name list
    
    :param entry: The entry from which to extract the feature names
    :type entry: dict  

    :return: The list of features present in the entry
    :rtype: list
    '''

    res = []
    for feature in entry['features'].keys():
        res.append(feature)
    return res



def get_best_estimator(
        best_estimator: dict, 
        challenger_estimator: dict, 
        metric: str
        ):
    '''Compare the metric of a given model result to a reference model result. 
    Get the best model based on the given metric
    
    :param best_estimator: The reference "best" estimator containing a 'model' and a 'metric'.
    It also contains the two sets used for training and testing 
    :type best_estimator: dict 

    :param challenger_estimator: The estimator to compare to the reference one. 
    Contains a 'model' and a 'metric'. It also contains the two sets used for training and testing
    :type challenger_estimator: dict 

    :param metric: The metric to use for the comparison of the two models. 
    Can be: 'accuracy', 'auroc', 'recall', 'precision', 'f1_score', or 'mcc_score'
    :type metric: str 

    :return: The best model based on the comparison of the two 'metric' values
    :rtype: dict
    '''

    if best_estimator == None:
        return challenger_estimator
    elif challenger_estimator['metrics'][metric] > best_estimator['metrics'][metric]:
        return challenger_estimator
    else:
        return best_estimator


def train_test_model(
        estimator: ExtraTreesClassifier, 
        data: dict
        ):
    '''Train and test the given estimator (ExtraTreeClassifier) using the dictionary of data 
    containing X and y for train and test. The accepted type for the "estimator" is based on the 
    result of the optimisation process. If you want to change the type of estimator you want to use, 
    please don't forget to change the type of estimator in this function.
    
    :param estimator: The sklearn estimator already configured but not fitted
    :type estimator: ExtraTreesClassifier 

    :param data: The dictionary containing the 'X_train', 'y_train', 'X_test' and 'y_test' datasets
    :type data: dict 

    :return: The complet result of the estimator training, containing the model, the train and test
    time, the different metrics result and the 2 used set for training and testing.
    :rtype: dict
    '''

    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test'] 
    y_test = data['y_test']

    start_train = time.time()
    estimator.fit(X_train, y_train)
    end_train = time.time()
    
    start_test = time.time()
    y_pred = estimator.predict(X_test)
    end_test = time.time()
    
    
    
    # Get Metrics
    accuracy = accuracy_score(y_test, y_pred)
    auroc_score = roc_auc_score(y_test, estimator.predict_proba(X_test)[:,1])
    recall, precision, f1_score, mcc_score = metrics.get_metrics(y_pred, y_test)

    # Get execution time:
    exec_train = end_train - start_train
    exec_test = end_test - start_test

    results = {
        'model':estimator,
        'execution_time_train':exec_train, 
        'execution_time_test':exec_test,
        'metrics':{
            'accuracy':accuracy, 
            'auroc':auroc_score,
            'recall':recall,
            'precision':precision,
            'f1_score':f1_score,
            'mcc_score':mcc_score
        },
        'sets':{
            'training':[X_train,y_train],
            'testing':[X_test, y_test]
        }
    }

    return results


def store_metrics(
        training_result: dict, 
        all_lists: dict
        ):
    '''Store a model result into a list from a dictionary
    
    :param training_result: The result obtained by the test of a model and store as a dictionray 
    with train and test execution times as well as accuracy, AUROC, F1 score and MCC score
    :type training_result: dict 

    :param all_lists: The dictionray containing lists for each values 
    (execution time and metrics score)
    :type all_lists: dict 

    :return: The dictionary of list where the new scores have been added
    :rtype: dict
    '''

    all_lists['execution_time_train'].append(training_result['execution_time_train'])
    all_lists['execution_time_test'].append(training_result['execution_time_test'])

    all_lists['metrics']['accuracy'].append(training_result['metrics']['accuracy'])
    all_lists['metrics']['auroc'].append(training_result['metrics']['auroc'])
    all_lists['metrics']['f1_score'].append(training_result['metrics']['f1_score'])   
    all_lists['metrics']['mcc_score'].append(training_result['metrics']['mcc_score'])

    return all_lists




####################################################################################################
###########################################     MAIN     ###########################################
####################################################################################################


if __name__ == "__main__":

    #### Parameters:
    # Feature generation parameters
    classification_mode = "charac_3"

    # Number Cross-Validation
    number_iter = 500

    # Hyper parameter grid
    param_grid_extratree = {
        'n_estimators': 1788,
        'max_features': 'sqrt', 
        'max_depth': 102, 
        'min_samples_split': 4, 
        'min_samples_leaf': 2, 
        'bootstrap': False
    }





    #### Paths ----------> May need modification depending on your usage. 
    # If you have difficulties with the paths, you can replace them by your actual absolute paths

    # Get current dir
    current_directory = path.dirname(__file__)

    # path to the positive and negative dataset (as a csv with spe = ';')
    relative_path_dataset = "data/set_ml_CSV_80_0.csv"
    path_dataset = path.join(current_directory, relative_path_dataset)

    # path to save to model
    relative_model_name = 'result/Cross_Beta_pred_model_ExtraTree.pickle'
    model_name = path.join(current_directory, relative_model_name)



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

    # Features to ignore
    ignored_features = non_selected_features


    #### Load data CSV
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

    #### Generating features
    print("Generating feature values...")
    for dataset in all_data:
        i = 0
        for entry in dataset:
            entry = get_features(entry, classification_mode, ignored_features)
            if i == 0:
                feature_list = get_feature_list(entry)
            i += 1


    print(f"Used features: {feature_list}")

    #### Initialize result arrays
    all_lists = {
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

    best_forest = None


    for i in range(number_iter):
        # Merge pos and neg in 1:1 ratio
        print(f"iteration: {i}/{number_iter}", end="\r")
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

        if i == 0:
            print(f"Size train set: {len(y_train)}")
            print(f"Size test set: {len(y_test)}")
        
        # Train and test model
        forest = ExtraTreesClassifier(random_state=0, 
                                      n_estimators=param_grid_extratree['n_estimators'], 
                                      max_features=param_grid_extratree['max_features'], 
                                      max_depth=param_grid_extratree['max_depth'], 
                                      min_samples_split=param_grid_extratree['min_samples_split'], 
                                      min_samples_leaf=param_grid_extratree['min_samples_leaf'], 
                                      bootstrap=param_grid_extratree['bootstrap'])
        
        results = train_test_model(forest, data)        
        best_forest = get_best_estimator(best_forest, results, 'f1_score')

        all_lists = store_metrics(results, all_lists)

    # Get mean values from result lists
    mean_exec_train = np.mean(all_lists['execution_time_train'])
    mean_exec_test = np.mean(all_lists['execution_time_test'])

    mean_accuracy = np.mean(all_lists['metrics']['accuracy'])
    mean_auroc = np.mean(all_lists['metrics']['auroc'])
    mean_f1_score = np.mean(all_lists['metrics']['f1_score'])
    mean_mcc_score = np.mean(all_lists['metrics']['mcc_score'])

    # Display results
    print(f"Result for training and testing {forest.__class__.__name__} \
          for {number_iter} cross-validation and classification = {classification_mode}:")
    print(f"\t- Average execution time for training: {mean_exec_train}")
    print(f"\t- Average execution time for testing: {mean_exec_test}")
    print(f"\t- Average accuracy: {mean_accuracy}")
    print(f"\t- Average AUROC: {mean_auroc}")
    print(f"\t- Average F1 score: {mean_f1_score}")
    print(f"\t- Best F1 score: {best_forest['metrics']['f1_score']}")
    print(f"\t- Average MCC score: {mean_mcc_score}")

    # save the best model
    pickle.dump(best_forest['model'], open(model_name,'wb'))

