import unittest

import numpy as np
from econml.sklearn_extensions.model_selection import *
from econml.sklearn_extensions.model_selection_utils import *
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


class TestSearchEstimatorListClassifier(unittest.TestCase):
    def setUp(self):
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def test_initialization(self):
        with self.assertRaises(ValueError):
            SearchEstimatorList(search='invalid_search')

    def test_auto_param_grid_discrete(self):

        search_estimator_list = SearchEstimatorList(is_discrete=True)
        search_estimator_list.select(self.X_train, self.y_train)
        self.assertIsNotNone(search_estimator_list.best_estimator_)
        self.assertIsNotNone(search_estimator_list.best_score_)
        self.assertIsNotNone(search_estimator_list.best_params_)

    def test_random_forest_discrete(self):
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)
        estimator_list = [RandomForestClassifier()]
        param_grid_list = [{'n_estimators': [10, 50, 100], 'max_depth': [3, 5, None]}]
        search_estimator_list = SearchEstimatorList(
            estimator_list=estimator_list, param_grid_list=param_grid_list, is_discrete=True)
        search_estimator_list.select(self.X_train, self.y_train)
        self.assertIsNotNone(search_estimator_list.best_estimator_)
        self.assertIsNotNone(search_estimator_list.best_score_)
        self.assertIsNotNone(search_estimator_list.best_params_)
        print("Best estimator: ", search_estimator_list.best_estimator_)
        print("Best score: ", search_estimator_list.best_score_)
        print("Best parameters: ", search_estimator_list.best_params_)

    def test_linear_estimator(self):
        search = SearchEstimatorList(estimator_list='linear', is_discrete=True)
        search.select(self.X_train, self.y_train)
        y_pred = search.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average=None)
        self.assertAlmostEqual(acc, your_expected_accuracy, delta=your_accuracy_tolerance)
        self.assertAlmostEqual(f1, your_expected_f1_score, delta=your_f1_score_tolerance)

    def test_poly_estimator(self):
        search = SearchEstimatorList(estimator_list='poly', is_discrete=True)
        search.select(self.X_train, self.y_train)
        y_pred = search.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average=None)
        self.assertAlmostEqual(acc, your_expected_accuracy, delta=your_accuracy_tolerance)
        self.assertAlmostEqual(f1, your_expected_f1_score, delta=your_f1_score_tolerance)

    def test_gbf_estimator(self):
        search = SearchEstimatorList(estimator_list='gbf', is_discrete=True)
        search.select(self.X_train, self.y_train)
        y_pred = search.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average=None)
        self.assertAlmostEqual(acc, your_expected_accuracy, delta=your_accuracy_tolerance)
        self.assertAlmostEqual(f1, your_expected_f1_score, delta=your_f1_score_tolerance)

    def test_nnet_estimator(self):
        search = SearchEstimatorList(estimator_list='nnet', is_discrete=True)
        search.select(self.X_train, self.y_train)
        y_pred = search.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average=None)
        self.assertAlmostEqual(acc, your_expected_accuracy, delta=your_accuracy_tolerance)
        self.assertAlmostEqual(f1, your_expected_f1_score, delta=your_f1_score_tolerance)

    def test_linear_and_forest_estimators(self):
        search = SearchEstimatorList(estimator_list=['linear', 'forest'], is_discrete=True)
        search.select(self.X_train, self.y_train)
        y_pred = search.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average=None)
        self.assertAlmostEqual(acc, your_expected_accuracy, delta=your_accuracy_tolerance)
        self.assertAlmostEqual(f1, your_expected_f1_score, delta=your_f1_score_tolerance)

    def test_all_estimators(self):
        search = SearchEstimatorList(estimator_list=['linear', 'forest', 'gbf', 'nnet', 'poly'], is_discrete=True)
        search.select(self.X_train, self.y_train)
        y_pred = search.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average=None)
        self.assertAlmostEqual(acc, your_expected_accuracy, delta=your_accuracy_tolerance)
        self.assertAlmostEqual(f1, your_expected_f1_score, delta=your_f1_score_tolerance)

    def test_logistic_regression_estimator(self):
        search = SearchEstimatorList(estimator_list=LogisticRegression(), is_discrete=True)
        search.select(self.X_train, self.y_train)
        y_pred = search.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average=None)
        self.assertAlmostEqual(acc, your_expected_accuracy, delta=your_accuracy_tolerance)
        self.assertAlmostEqual(f1, your_expected_f1_score, delta=your_f1_score_tolerance)

    def test_logistic_regression_cv_estimator(self):
        search = SearchEstimatorList(estimator_list=LogisticRegressionCV(), is_discrete=True)
        search.select(self.X_train, self.y_train)
        y_pred = search.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average=None)
        self.assertAlmostEqual(acc, your_expected_accuracy, delta=your_accuracy_tolerance)
        self.assertAlmostEqual(f1, your_expected_f1_score, delta=your_f1_score_tolerance)

    def test_empty_estimator_list(self):
        search = SearchEstimatorList(estimator_list=[], is_discrete=True)
        search.select(self.X_train, self.y_train)
        y_pred = search.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        self.assertAlmostEqual(acc, your_expected_accuracy, delta=your_accuracy_tolerance)

if __name__ == '__main__':
    unittest.main()
