
import pdb
import warnings

import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.linear_model
import sklearn.neural_network
import sklearn.preprocessing
from sklearn.base import BaseEstimator, is_regressor
from sklearn.ensemble import (GradientBoostingClassifier,
                              GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import (ElasticNetCV,
                                  LogisticRegression,
                                  LogisticRegressionCV,)
from sklearn.model_selection import (BaseCrossValidator, GridSearchCV,
                                     RandomizedSearchCV,
                                     check_cv)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (PolynomialFeatures,
                                   StandardScaler)
from sklearn.svm import SVC, LinearSVC
import inspect
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.exceptions import NotFittedError

models_regression = [
    ElasticNetCV(),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    MLPRegressor()
]


models_classification = [
    LogisticRegressionCV(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    MLPClassifier()
]

model_list = ['linear', 'forest', 'gbf', 'nnet', 'poly', 'automl']


def scale_pipeline(model):
    """
    Returns a pipeline that scales the input data using StandardScaler and applies the given model.

    Parameters
    ----------
        model : estimator object
            A model object that implements the scikit-learn estimator interface.

    Returns
    ----------
        pipe : Pipeline object
            A pipeline that scales the input data using StandardScaler and applies the given model.
    """
    pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
    return pipe


def flatten_list(lst):
    """
    Flatten a list that may contain nested lists.

    Parameters
    ----------
        lst (list): The list to flatten.

    Returns
    ----------
        list: The flattened list.
    """
    flattened = []
    for item in lst:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened


def is_polynomial_pipeline(estimator):
    if not isinstance(estimator, Pipeline):
        return False
    steps = estimator.steps
    if len(steps) != 2:
        return False
    poly_step = steps[0]
    if not isinstance(poly_step[1], PolynomialFeatures):
        return False
    return True


def check_list_type(lst):
    """
    Checks if a list only contains strings, sklearn model objects, and sklearn model selection objects.

    Parameters
    ----------
        lst (list): A list to check.

    Returns
    ----------
        bool: True if the list only contains valid objects, False otherwise.

    Raises:
        TypeError: If the list contains objects other than strings, sklearn model objects, or sklearn model selection objects.

    Examples:
        >>> check_list_type(['linear', RandomForestRegressor(), KFold()])
        True
        >>> check_list_type([1, 'linear'])
        TypeError: The list must contain only strings, sklearn model objects, and sklearn model selection objects.
    """
    if len(lst) == 0:
        raise ValueError("Estimator list is empty. Please add some models or use some of the defaults provided.")

    for element in lst:
        if not isinstance(element, (str, BaseEstimator, BaseCrossValidator)):
            raise TypeError(
                "The list must contain only strings, sklearn model objects, and sklearn model selection objects.")
    return True


def select_continuous_estimator(estimator_type):
    """
    Returns a continuous estimator object for the specified estimator type.

    Parameters
    ----------
        estimator_type (str): The type of estimator to use, one of: 'linear', 'forest', 'gbf', 'nnet', 'poly'.

    Returns
    ----------
        object: An instance of the selected estimator class.

    Raises:
        ValueError: If the estimator type is unsupported.
    """
    if estimator_type == 'linear':
        return (ElasticNetCV())
    elif estimator_type == 'forest':
        return RandomForestRegressor()
    elif estimator_type == 'gbf':
        return GradientBoostingRegressor()
    elif estimator_type == 'nnet':
        return (MLPRegressor())
    elif estimator_type == 'poly':
        poly = sklearn.preprocessing.PolynomialFeatures()
        linear = sklearn.linear_model.ElasticNetCV(cv=3)  # Play around with precompute and tolerance
        return (Pipeline([('poly', poly), ('linear', linear)]))
    else:
        raise ValueError(f"Unsupported estimator type: {estimator_type}")


def select_discrete_estimator(estimator_type):
    """
    Returns a discrete estimator object for the specified estimator type.

    Parameters
    ----------
        estimator_type (str): The type of estimator to use, one of: 'linear', 'forest', 'gbf', 'nnet', 'poly'.

    Returns
    ----------
        object: An instance of the selected estimator class.

    Raises:
        ValueError: If the estimator type is unsupported.
    """
    if estimator_type == 'linear':
        return (LogisticRegressionCV(multi_class='auto'))
    elif estimator_type == 'forest':
        return RandomForestClassifier()
    elif estimator_type == 'gbf':
        return GradientBoostingClassifier()
    elif estimator_type == 'nnet':
        return (MLPClassifier())
    elif estimator_type == 'poly':
        poly = PolynomialFeatures()
        linear = LogisticRegressionCV(multi_class='auto')
        return (Pipeline([('poly', poly), ('linear', linear)]))
    else:
        raise ValueError(f"Unsupported estimator type: {estimator_type}")


def select_estimator(estimator_type, is_discrete):
    """
    Returns an estimator object for the specified estimator and target types.

    Parameters
    ----------
        estimator_type (str): The type of estimator to use, one of: 'linear', 'forest', 'gbf', 'nnet', 'poly', 'automl', 'all'.
        is_discrete (bool): The type of target variable, if true then it's discrete.

    Returns
    ----------
        object: An instance of the selected estimator class.

    Raises:
        ValueError: If the estimator or target types are unsupported.
    """
    if not isinstance(is_discrete, bool):
        raise ValueError(f"Unsupported target type: {is_discrete}")
    elif is_discrete:
        return select_discrete_estimator(estimator_type=estimator_type)
    else:
        return select_continuous_estimator(estimator_type=estimator_type)


def get_complete_estimator_list(estimator_list, is_discrete):
    '''
    Returns a list of sklearn objects from an input list of str's, and sklearn objects.

    Parameters
    ----------
        estimator_list : List of estimators; can be sklearn object or str: 'linear', 'forest', 'gbf', 'nnet', 'poly', 'auto', 'all'.
        is_discrete (bool): if target type is discrete or continuous.

    Returns
    ----------
        object: A list of sklearn objects

    Raises:
        ValueError: If the estimator is not supported.

    '''
    # pdb.set_trace()
    if isinstance(estimator_list, str):
        if 'all' == estimator_list:
            estimator_list = ['linear', 'forest', 'gbf', 'nnet', 'poly']
        elif 'auto' == estimator_list:
            estimator_list = ['linear']
        elif estimator_list in ['linear', 'forest', 'gbf', 'nnet', 'poly']:
            estimator_list = [estimator_list]
        else:
            raise ValueError(
                "Invalid estimator_list value. Please provide a valid value from the list of available estimators: ['linear', 'forest', 'gbf', 'nnet', 'poly', 'automl']")

    if isinstance(estimator_list, BaseEstimator):
        estimator_list = [estimator_list]

    if not isinstance(estimator_list, list):
        if 'auto' in estimator_list:
            for estimator in ['linear']:
                if estimator not in estimator_list:
                    estimator_list.append(estimator)
        if 'all' in estimator_list:
            for estimator in ['linear', 'forest', 'gbf', 'nnet', 'poly']:
                if estimator not in estimator_list:
                    estimator_list.append(estimator)
        raise ValueError(f"estimator_list should be of type list not: {type(estimator_list)}")

    check_list_type(estimator_list)
    temp_est_list = []

    # Set to remove duplicates
    for estimator in set(estimator_list):
        # if sklearn object: add to list, else turn str into corresponding sklearn object and add to list
        if isinstance(estimator, (BaseEstimator, BaseCrossValidator)):
            temp_est_list.append(estimator)
        else:
            temp_est_list.append(select_estimator(estimator, is_discrete))
    temp_est_list = flatten_list(temp_est_list)

    # Check that all types of models are matched towards the problem. 
    for estimator in temp_est_list:
        if not is_regressor_or_classifier(estimator, is_discrete=is_discrete):
            raise TypeError("Invalid estimator type: {} - must be a regressor or classifier".format(type(estimator)))
    return temp_est_list


def select_classification_hyperparameters(estimator):
    """
    Returns a hyperparameter grid for the specified classification model type.

    Parameters
    ----------
        model_type (str): The type of model to be used. Valid values are 'linear', 'forest', 'nnet', and 'poly'.

    Returns
    ----------
        A dictionary representing the hyperparameter grid to search over.
    """

    if isinstance(estimator, LogisticRegressionCV):
        return {
            'Cs': [0.01, 0.1, 1],
            'cv': [3],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['lbfgs', 'liblinear', 'saga']
        }
    elif isinstance(estimator, RandomForestClassifier):
        return {
            'n_estimators': [100, 500],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    elif isinstance(estimator, GradientBoostingClassifier):
        return {
            'n_estimators': [100, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],

        }
    elif isinstance(estimator, MLPClassifier):
        return {
            'hidden_layer_sizes': [(10,), (50,), (100,)],
            'activation': ['relu'],
            'solver': ['adam'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
    elif is_polynomial_pipeline(estimator=estimator):
        return {
            'poly__degree': [2, 3, 4],
            'linear__Cs': [1, 10, 20],
            'linear__max_iter': [100, 200],
            'linear__penalty': ['l2'],
            'linear__solver': ['saga', 'liblinear', 'lbfgs']
        }
    else:
        warnings.warn("No hyperparameters for this type of model. There are default hyperparameters for LogisticRegressionCV, RandomForestClassifier, MLPClassifier, and the polynomial pipleine", category=UserWarning)
        return {}
        # raise ValueError("Invalid model type. Valid values are 'linear', 'forest', 'nnet', and 'poly'.")


def select_regression_hyperparameters(estimator):
    """
    Returns a dictionary of hyperparameters to be searched over for a regression model.

    Parameters
    ----------
        model_type (str): The type of model to be used. Valid values are 'linear', 'forest', 'nnet', and 'poly'.

    Returns
    ----------
        A dictionary of hyperparameters to be searched over using a grid search.
    """
    if isinstance(estimator, ElasticNetCV):
        return {
            'l1_ratio': [0.1, 0.5, 0.9],
            'cv': [3],
            'max_iter': [1000],
        }
    elif isinstance(estimator, RandomForestRegressor):
        return {
            'n_estimators': [100],
            'max_depth': [None, 10, 50],
            'min_samples_split': [2, 5, 10],
        }
    elif isinstance(estimator, MLPRegressor):
        return {
            'hidden_layer_sizes': [(10,), (50,), (100,)],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
    elif isinstance(estimator, GradientBoostingRegressor):
        return {
            'n_estimators': [100, 500],
            'learning_rate': [0.01, 0.1, 0.05],
            'max_depth': [3, 5],
        }
    elif is_polynomial_pipeline(estimator=estimator):
        return {
            'linear__l1_ratio': [0.1, 0.5, 0.9],
            'linear__max_iter': [1000],
            'poly__degree': [2, 3, 4]
        }
    else:
        warnings.warn("No hyperparameters for this type of model. There are default hyperparameters for ElasticNetCV, RandomForestRegressor, MLPRegressor, and the polynomial pipeline.", category=UserWarning)
        return {}


def is_linear_model(estimator):
    """
    Check whether an estimator is a polynomial regression, logistic regression, linear SVM, or any other type of
    linear model.

    Parameters
    ----------
    estimator (scikit-learn estimator): The estimator to check.

    Returns
    ----------
    is_linear (bool): True if the estimator is a linear model, False otherwise.
    """

    if isinstance(estimator, Pipeline):
        has_poly_feature_step = any(isinstance(step[1], PolynomialFeatures) for step in estimator.steps)
        if has_poly_feature_step:
            return True

    if hasattr(estimator, 'fit_intercept') and hasattr(estimator, 'coef_'):
        return True

    if isinstance(estimator, (LogisticRegression, LinearSVC, SVC)):
        return True

    return False


def is_data_scaled(X):
    """
    Check if the input data is already centered and scaled using StandardScaler.

    Parameters
    ----------
        X array-like of shape (n_samples, n_features): The input data.

    Returns
    ----------
        is_scaled (bool): Whether the input data is already centered and scaled using StandardScaler or not.

    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    is_scaled = np.allclose(mean, 0.0) and np.allclose(std, 1.0)

    return is_scaled


def auto_hyperparameters(estimator_list, is_discrete=True):
    """
    Selects hyperparameters for a list of estimators.

    Parameters
    ----------
    - estimator_list: list of scikit-learn estimators
    - is_discrete: boolean indicating whether the problem is classification or regression

    Returns
    ----------
    - param_list: list of parameter grids for the estimators
    """
    param_list = []
    for estimator in estimator_list:
        if is_discrete:
            param_list.append(select_classification_hyperparameters(estimator=estimator))
        else:
            param_list.append(select_regression_hyperparameters(estimator=estimator))
    return param_list


def set_search_hyperparameters(search_object, hyperparameters):
    if isinstance(search_object, (RandomizedSearchCV, GridSearchCV)):
        search_object.set_params(**hyperparameters)
    else:
        raise ValueError("Invalid search object")


def is_mlp(estimator):
    return isinstance(estimator, (MLPClassifier, MLPRegressor))


def has_random_state(model):
    if is_polynomial_pipeline(model):
        signature = inspect.signature(type(model['linear']))
    else:
        signature = inspect.signature(type(model))
    return ("random_state" in signature.parameters)


def supports_sample_weight(estimator):
    fit_signature = inspect.signature(estimator.fit)
    return 'sample_weight' in fit_signature.parameters


def just_one_model_no_params(estimator_list, param_list):
    return (len(estimator_list) == 1) and (len(param_list) == 1) and (len(param_list[0]) == 0)

def is_regressor_or_classifier(model, is_discrete):
    if is_discrete:
        if is_polynomial_pipeline(model):
            return isinstance(model[1], ClassifierMixin)    
        else:
            return isinstance(model, ClassifierMixin)
    else:
        if is_polynomial_pipeline(model):
            return isinstance(model[1], RegressorMixin)    
        else:
            return isinstance(model, RegressorMixin)
