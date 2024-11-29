
import random
import math
from sklearn import preprocessing




def create_dataset(
        dataset_list: list, 
        positive_index: int, 
        negative_index: int, 
        ratio: str = '1:1'
        ):
    '''Create a positive and a negative set based on the given ratio. Select data by random sampling
    
    :param dataset_list: The list containing both the positive dataset and the negative one
    :type dataset_list: list 

    :param positive_index: The index where to find the positive data in the datasets_list
    :type positive_index: int 

    :param negative_index: The index where to find the negative data in the datasets_list
    :type negative_index: int 

    :param ratio: The ratio between positive and negative as positive_ratio:negative_ratio. If not
    provided, Defaultto  "1:1"
    :type ratio: str 
    
    :return: The randomly sampled positive data of a size depending on the given ratio, 
    The randomly sampled negative data of a size depending on the given ratio
    :rtype: (list, list)
    '''

    factor_pos = float(ratio.split(':')[0])
    factor_neg = float(ratio.split(':')[1])

    nb_pos = len(dataset_list[positive_index])
    nb_neg = len(dataset_list[negative_index])

    final_pos_nb = int(nb_pos * factor_pos)
    final_neg_nb = int(final_pos_nb * factor_neg)

    # Add security if one number is bigger than the size of its corresponding dataset
    if final_pos_nb > nb_pos:
        print("Error, ratio ", ratio, " invalid. Maximal positive number = ", 
              nb_pos, " and asked for ", final_pos_nb)
        return None
    if final_neg_nb > nb_neg:
        print("Error, ratio ", ratio, " invalid. Maximal negative number = ", 
              nb_neg, " and asked for ", final_neg_nb)
        return None
    
    new_pos_data = random.sample(dataset_list[positive_index], final_pos_nb)
    new_neg_data = random.sample(dataset_list[negative_index], final_neg_nb)

    return new_pos_data, new_neg_data



def split_data_train_test(
        data : list, 
        train_ratio: float
        ):
    '''Split the given data into a train set and a test set based on the wanted train ratio
    
    :param data: The array which will be randomly split into 2 sets
    :type data: list 

    :param train_ratio: The proportion of the training set (from 0.1 to 0.9)
    :type train_ratio: float 

    :return: The train set containing randomly choosed entries from the provided data, 
    The test set containing randomly choosed entries from the provided data
    :rtype: (list, list)
    '''

    if train_ratio < 0.1 or train_ratio > 0.9:
        print(f"Error: train_ratio of {train_ratio} is invalid.",
              "Please select a value between 0.1 and 0.9")
        return None
    nb_train = math.floor(len(data)*train_ratio)
    index_train = random.sample(range(0, len(data)), nb_train)

    train = []
    test = []
    
    for i in index_train:
        train.append(data[i])
    for i in range(len(data)):
        if i not in index_train:
            test.append(data[i])

    return train, test



def get_features_value(
        entry: dict, 
        ignored_features: list = []
        ):
    '''Extract feature values from the given entry. 
    Ignore the feature provided in the ignored_feature list.
    
    :param entry: A entry dictionary containing a 'features' key leading to a list of features 
    with their values
    :type entry: dict 

    :param ignored_features: A list of feature name which will be ignored during the feature 
    extraction. If not provided, Defaults to []
    :type ignored_features: list 

    :return: The list of feature values without unwanted features
    :rtype: list
    '''

    res = []
    for feature in entry['features'].keys():
        if feature not in ignored_features:
            res.append(entry['features'][feature])
    return res



def convert_set_X_y(
        data: list, 
        ignored_features: list = []
        ):
    '''Use the 'label' of entries from the given dataset to split 
    feature values and the labels in two arrays
    
    :param data: The list of entry where every entries have a 'features' and a 'label' keys
    :type data: list 

    :param ignored_features: The list of feature to ignore when creating the sets
    :type ignored_features: list 

    :return: The list of entry features values, 
    The list of label where 1 is LLPS (positive) and 0 all the other (negative)
    :rtype: (list, list)
    '''

    X_list = []
    y_list = []

    for entry in data:
        X_list.append(get_features_value(entry, ignored_features))
        if entry['label'] == 'amyloid':
            y_list.append(1)
        else:
            y_list.append(0)

    return X_list, y_list



def get_train_test_sets(
        dataset_list: list, 
        positive_index: int, 
        negative_index: int, 
        pos_neg_ratio: str = '1:1', 
        train_ratio: float = 0.7, 
        ignored_features: list = [], 
        standardize: bool = False
        ):
    '''From an array containing the positive entries and the negative entries, 
    randomly sample them according to the given positive:negative ratio. 
    Then randomly split them into a training and a testing set according to the train proportion
    
    :param dataset_list: The dataset containing an array of positive entries and an array of 
    negative entries
    :type dataset_list: list 

    :param positive_index: The index where to find the positive entries in the dataset_list
    :type positive_index: int 

    :param negative_index: The index where to find the negative entries in the dataset_list
    :type negative_index: int 

    :param pos_neg_ratio: The positive:negative ration to apply for the creation of the random 
    dataset. If not provided, defaults to '1:1'
    :type pos_neg_ratio: str 

    :param train_ratio: The train set proportion compared to the test set.  
    The value must be between 0.1 and 0.9. If not provided, defaults to 0.7 
    :type train_ratio: float 

    :param ignored_features: The list of feature not to include in the set. If not provided, 
    defaults to []
    :type ignored_features: list 

    :param standardize: Add or not an extra step of feature standardization based on the training 
    set. If not provided, defaults to False
    :type standardize: bool 

    :return: The list of entries in the train set, The list of labels in the train set, 
    The list of entries in the test set, The list of lables in the test set
    :rtype: (list, list, list, list)
    '''
    new_pos_data, new_neg_data = create_dataset(
                                        dataset_list, 
                                        positive_index, 
                                        negative_index, 
                                        pos_neg_ratio
                                        )
    pos_train, pos_test = split_data_train_test(new_pos_data, train_ratio)
    neg_train, neg_test = split_data_train_test(new_neg_data, train_ratio)



    merged_train = pos_train + neg_train
    random.shuffle(merged_train)

    merged_test = pos_test + neg_test
    random.shuffle(merged_test)

    X_train, y_train = convert_set_X_y(merged_train, ignored_features)
    X_test, y_test = convert_set_X_y(merged_test, ignored_features)

    # Standardize entries
    if standardize:
        scaler = preprocessing.StandardScaler().fit(X_train)

        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test