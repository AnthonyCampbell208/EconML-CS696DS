import unittest

import numpy as np
from econml.sklearn_extensions.model_selection import *
from econml.sklearn_extensions.model_selection_utils import *
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class TestSearchEstimatorList(unittest.TestCase):
    def test_initialization(self):
        with self.assertRaises(ValueError):
            SearchEstimatorList(search='invalid_search')

    def test_auto_param_grid_discrete(self):
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)
        search_estimator_list = SearchEstimatorList(is_discrete=True)
        search_estimator_list.select(X_train, y_train)
        self.assertIsNotNone(search_estimator_list.best_estimator_)
        self.assertIsNotNone(search_estimator_list.best_score_)
        self.assertIsNotNone(search_estimator_list.best_params_)

    def test_auto_param_grid_continuous(self):
        X, y = fetch_california_housing(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)
        search_estimator_list = SearchEstimatorList(is_discrete=False)
        search_estimator_list.select(X_train, y_train)
        self.assertIsNotNone(search_estimator_list.best_estimator_)
        self.assertIsNotNone(search_estimator_list.best_score_)
        self.assertIsNotNone(search_estimator_list.best_params_)
        print("Best estimator: ", search_estimator_list.best_estimator_)
        print("Best score: ", search_estimator_list.best_score_)
        print("Best parameters: ", search_estimator_list.best_params_)

    def test_random_forest_discrete(self):
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)
        estimator_list = [RandomForestClassifier()]
        param_grid_list = [{'n_estimators': [10, 50, 100], 'max_depth': [3, 5, None]}]
        search_estimator_list = SearchEstimatorList(
            estimator_list=estimator_list, param_grid_list=param_grid_list, is_discrete=True)
        search_estimator_list.select(X_train, y_train)
        self.assertIsNotNone(search_estimator_list.best_estimator_)
        self.assertIsNotNone(search_estimator_list.best_score_)
        self.assertIsNotNone(search_estimator_list.best_params_)
        print("Best estimator: ", search_estimator_list.best_estimator_)
        print("Best score: ", search_estimator_list.best_score_)
        print("Best parameters: ", search_estimator_list.best_params_)

    def test_random_forest_continuous(self):
        X, y = fetch_california_housing(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)
        estimator_list = [RandomForestRegressor()]
        param_grid_list = [{'n_estimators': [10, 50, 100], 'max_depth': [3, 5, None]}]
        search_estimator_list = SearchEstimatorList(
            estimator_list=estimator_list, param_grid_list=param_grid_list, is_discrete=False)
        search_estimator_list.select(X_train, y_train)
        self.assertIsNotNone(search_estimator_list.best_estimator_)
        self.assertIsNotNone(search_estimator_list.best_score_)
        self.assertIsNotNone(search_estimator_list.best_params_)

    def test_warning(self):
        X, y = fetch_california_housing(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)
        estimator_list = [RandomForestRegressor()]
        param_grid_list = [{'n_estimators': [10, 50, 100], 'max_depth': [3, 5, None]}]
        with self.assertWarns(UserWarning):
            search_estimator_list = SearchEstimatorList(
                estimator_list=estimator_list, param_grid_list=param_grid_list, is_discrete=False)


if __name__ == '__main__':
    unittest.main()
