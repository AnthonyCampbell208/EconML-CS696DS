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
        search_estimator_list.fit(self.X_train, self.y_train)
        self.assertIsNotNone(search_estimator_list.best_estimator_)
        self.assertIsNotNone(search_estimator_list.best_score_)
        self.assertIsNotNone(search_estimator_list.best_params_)

    def test_linear_estimator(self):
        search = SearchEstimatorList(estimator_list='linear', is_discrete=True)
        search.fit(self.X_train, self.y_train)
        y_pred = search.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average=None)

        self.assertEqual(len(search.complete_estimator_list), 1)
        self.assertEqual(len(search.param_grid_list), 1)
        self.assertEqual(type(search.complete_estimator_list[0]), LogisticRegressionCV)

        self.assertAlmostEqual(acc, expected_accuracy, delta=accuracy_tolerance)
        self.assertAlmostEqual(f1, expected_f1_score, delta=f1_score_tolerance)

    def test_poly_estimator(self):
        search = SearchEstimatorList(estimator_list='poly', is_discrete=True)
        search.fit(self.X_train, self.y_train)
        y_pred = search.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average=None)

        self.assertEqual(len(search.complete_estimator_list), 1)
        self.assertTrue(is_polynomial_pipeline(search.complete_estimator_list[0]))

        self.assertAlmostEqual(acc, expected_accuracy, delta=accuracy_tolerance)
        self.assertAlmostEqual(f1, expected_f1_score, delta=f1_score_tolerance)

    def test_forest_estimator(self):
        search = SearchEstimatorList(estimator_list='forest', is_discrete=True)
        search.fit(self.X_train, self.y_train)
        y_pred = search.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average=None)

        self.assertEqual(len(search.complete_estimator_list), 1)
        self.assertEqual(len(search.param_grid_list), 1)
        self.assertEqual(type(search.complete_estimator_list[0]), RandomForestClassifier)

        self.assertAlmostEqual(acc, expected_accuracy, delta=accuracy_tolerance)
        self.assertAlmostEqual(f1, expected_f1_score, delta=f1_score_tolerance)

    def test_gbf_estimator(self):
        search = SearchEstimatorList(estimator_list='gbf', is_discrete=True)
        search.fit(self.X_train, self.y_train)
        y_pred = search.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average=None)

        self.assertEqual(len(search.complete_estimator_list), 1)
        self.assertEqual(len(search.param_grid_list), 1)
        self.assertEqual(type(search.complete_estimator_list[0]), GradientBoostingClassifier)

        self.assertAlmostEqual(acc, expected_accuracy, delta=accuracy_tolerance)
        self.assertAlmostEqual(f1, expected_f1_score, delta=f1_score_tolerance)

    def test_nnet_estimator(self):
        search = SearchEstimatorList(estimator_list='nnet', is_discrete=True)
        search.fit(self.X_train, self.y_train)
        y_pred = search.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average=None)

        self.assertEqual(len(search.complete_estimator_list), 1)
        self.assertEqual(len(search.param_grid_list), 1)
        self.assertEqual(type(search.complete_estimator_list[0]), MLPClassifier)

        self.assertAlmostEqual(acc, expected_accuracy, delta=accuracy_tolerance)
        self.assertAlmostEqual(f1, expected_f1_score, delta=f1_score_tolerance)

    def test_linear_and_forest_estimators(self):
        search = SearchEstimatorList(estimator_list=['linear', 'forest'], is_discrete=True)
        search.fit(self.X_train, self.y_train)
        y_pred = search.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average=None)

        self.assertEqual(len(search.complete_estimator_list), 2)
        self.assertEqual(len(search.param_grid_list), 2)
        self.assertEqual(type(search.complete_estimator_list[0]), LogisticRegressionCV)
        self.assertEqual(type(search.complete_estimator_list[1]), RandomForestClassifier)

        self.assertAlmostEqual(acc, expected_accuracy, delta=accuracy_tolerance)
        self.assertAlmostEqual(f1, expected_f1_score, delta=f1_score_tolerance)

    def test_all_estimators(self):
        search = SearchEstimatorList(estimator_list=['linear', 'forest', 'gbf', 'nnet', 'poly'], is_discrete=True)
        search.fit(self.X_train, self.y_train)
        y_pred = search.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average=None)

        self.assertEqual(len(search.complete_estimator_list), 5)
        self.assertEqual(len(search.param_grid_list), 5)

        self.assertAlmostEqual(acc, expected_accuracy, delta=accuracy_tolerance)
        self.assertAlmostEqual(f1, expected_f1_score, delta=f1_score_tolerance)

    def test_logistic_regression_estimator(self):
        search = SearchEstimatorList(estimator_list=LogisticRegression(), is_discrete=True)
        search.fit(self.X_train, self.y_train)
        y_pred = search.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average=None)
        self.assertAlmostEqual(acc, expected_accuracy, delta=accuracy_tolerance)
        self.assertAlmostEqual(f1, expected_f1_score, delta=f1_score_tolerance)

    def test_logistic_regression_cv_estimator(self):
        search = SearchEstimatorList(estimator_list=LogisticRegressionCV(), is_discrete=True)
        search.fit(self.X_train, self.y_train)
        y_pred = search.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average=None)
        self.assertAlmostEqual(acc, expected_accuracy, delta=accuracy_tolerance)
        self.assertAlmostEqual(f1, expected_f1_score, delta=f1_score_tolerance)

    def test_empty_estimator_list(self):
        search = SearchEstimatorList(estimator_list=[], is_discrete=True)
        search.fit(self.X_train, self.y_train)
        y_pred = search.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        self.assertAlmostEqual(acc, expected_accuracy, delta=accuracy_tolerance)

    def test_random_forest_discrete(self):
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)
        estimator_list = [RandomForestClassifier()]
        param_grid_list = [{'n_estimators': [10, 50, 100], 'max_depth': [3, 5, None]}]

        search = SearchEstimatorList(
            estimator_list=estimator_list, param_grid_list=param_grid_list, is_discrete=True)
        search.fit(self.X_train, self.y_train)

        self.assertEqual(len(search.complete_estimator_list), 1)
        self.assertEqual(len(search.param_grid_list), 1)

        self.assertIsNotNone(search.best_estimator_)
        self.assertIsNotNone(search.best_score_)
        self.assertIsNotNone(search.best_params_)


if __name__ == '__main__':
    unittest.main()
