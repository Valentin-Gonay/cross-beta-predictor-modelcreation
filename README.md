# Cross-Beta predictor: model creation

## Introduction

Welcome to the source code used for the creation of our cross-beta-forming amyloid predictor [Cross-Beta predictor](https://bioinfo.crbm.cnrs.fr/index.php?route=tools&tool=35) and [local version](https://github.com/Valentin-Gonay/cross-beta-predictor). This source code is the one used for the optimization of the different tested models:
* [RandomForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
* [ExtraTrees](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)
* [CatBoost](https://catboost.ai/)
* [GradientBoosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
* [SVM classifier](https://scikit-learn.org/1.5/modules/generated/sklearn.svm.SVC.html)
* [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
* [DecisionTree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)

The scripts are working in two steps. First, you will need to use the `optimize_models_multiprocess.py` script to identify the best hyperparameters for the different models. Then you will be able to choose the best one (in our case the [ExtraTrees](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html) model) and generate a model using the hyperparameters available in the `/result/optimisation/` folder and the `generate_model.py` script.

## Requirements

This work has been done using `python3.12.2` in association with other libraries as:
* `scikit-learn` version 1.3.1 `$pip install scikit-learn==1.3.1`
* `numpy` version 1.26.4 `$pip install numpy==1.26.4`
* `pandas` version 2.2.1 `$pip install pandas==2.2.1`
* `tqdm` version 4.66.4 `$pip install tqdm==4.66.4`
* `catboost` version 1.2.5 `$pip install catboost==1.2.5`
 
This work also uses a protein disorder predictor [IUPred3](https://iupred3.elte.hu/) which you can download by requesting their [website](https://iupred3.elte.hu/download_new). 

Note that IUPred3 is under academic license.

Once you have downloaded IUPRed3, you can put it in the `feature_creation` folder as:

```
- Cross-Beta_predictor_modelCreation
|
| - feature_creation
|  |
|  | - iupred3 <------
|  |  |
|  |  | ...
|  |
|  | aa_compo.py
...
```




## How to generate your model

All data used to train and test the different models during the optimization process and to generate your final model can be found in the `.csv` file stored in the `data` folder.

### Optimisation of the models
The first step is to optimize the different models to select one showing the best performance and to identify the best hyperparameters for each of them.

In the `optimize_models_multiprocess.py` file, edit your preference options:

```
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
number_iter = 500

# Number of random grid to test
# Can vary depending on your need and on the computation time needed for the different models
grid_to_try = 2

# Number of cpu
# Only used if multiprocess is True
process_count = multiprocessing.cpu_count()

# Metric used for the result comparison
# Can be 'accuracy', 'auroc', 'recall', 'precision', 'f1_score', or 'mcc_score'
metric = 'f1_score'
```

You can then run the script. In a terminal, set your location to the `Cross-Beta_predictor_modelCreation` folder

```$cd PATH/TO/YOUR/INSTALL/OF/Cross-Beta_predictor_modelCreation```

And then execute the script

```$python3 optimize_models_multiprocess.py```


Results will be stored in different `.json` files in the `/result/optimisation/` folder, as an example:
```
{
    "estimator": "RandomForest",
    "execution_time_train": 1.9349546432495117,
    "execution_time_test": 0.07115755081176758,
    "metrics": {
        "accuracy": 0.83125,
        "auroc": 0.9046875,
        "f1_score": 0.8420008912655973,
        "mcc_score": 0.6721540229895849
    },
    "params": {
        "n_estimators": 4855,
        "max_features": "log2",
        "max_depth": 5,
        "min_samples_split": 8,
        "min_samples_leaf": 2,
        "bootstrap": true
    },
    "conditions": {
        "number_of_grids": 2,
        "number_of_iter": 5
    }
}
```

You will find in the different `.json` files:
* `"estimator"`: the name of the optimized model
* `"execution_time_train"`: the average time needed for the model to train (based on "number_of_iter")
* `"execution_time_test"`: the average time needed for the model to test (based on "number_of_iter")
* `"metrics"`: the different measured metrics
* `"params"`: the best grid of parameters found for this model and for the given conditions
* `"conditions"`: the number of grid tested and the number of iterations (different train/test split and subsampling of negative data)


### Creation of the model

Now that you have the model you want to use (here the "ExtraTrees" model) with the best hyperparameters for it, you can generate the final model.

For this, you will need to use the `generate_model.py` script.

You will have to start by adjusting the parameters of the script as:

```
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
```

The classification method "charac_3" is the amino acid classification method used to create Cross-Beta predictor. The number of iterations (=500 for the creation of the Cross-Beta predictor) is the number of different train/test set splits and different negative set subsampling to test have an average performance of the model.

The Hyperparameter grid is the one you will have to change depending on your optimization results.

Once your parameters are updated, you can run the script. For this, in a terminal, set your location to the `Cross-Beta_predictor_modelCreation` folder

```$cd PATH/TO/YOUR/INSTALL/OF/Cross-Beta_predictor_modelCreation```

And then execute the script

```$python3 generate_model.py```

The model will be saved in `/result/Cross_Beta_pred_model_ExtraTree.pickle`

The model can then be loaded using the `pickle` library using the [documentation](https://docs.python.org/3/library/pickle.html)



## Citations
### Author
Valentin Gonay

[GitHub](https://github.com/Valentin-Gonay)

### Citing Cross-Beta predictor

Valentin Gonay, Michael P. Dunne, Javier Caceres-Delpiano, & Andrey V. Kajava. (2024). Developing machine-learning-based amyloid predictors with Cross-Beta DB. bioRxiv, 2024.02.12.579644. https://doi.org/10.1101/2024.02.12.579644

### Citing IUPred3

Gábor Erdős, Mátyás Pajkos, Zsuzsanna Dosztányi IUPred3 - improved prediction of protein disorder with a focus on specific user applications Nucleic Acids Research 2021, Submitted Bálint Mészáros, Gábor Erdős, Zsuzsanna Dosztányi (2018) IUPred2A: context-dependent prediction of protein disorder as a function of redox state and protein binding. Nucleic Acids Research;46(W1):W329-W337.
