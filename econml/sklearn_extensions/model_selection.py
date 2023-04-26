# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.
"""Collection of scikit-learn extensions for model selection techniques."""

import numbers
import pdb
import warnings

import numpy as np
import scipy.sparse as sp
import sklearn
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection import (BaseCrossValidator, GridSearchCV, KFold,
                                     RandomizedSearchCV, StratifiedKFold,
                                     check_cv)
# TODO: conisder working around relying on sklearn implementation details
from sklearn.model_selection._validation import (_check_is_permutation,
                                                 _fit_and_predict)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state, indexable
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _num_samples

from econml.sklearn_extensions.model_selection_utils import *


def _split_weighted_sample(self, X, y, sample_weight, is_stratified=False):
    random_state = self.random_state if self.shuffle else None
    if is_stratified:
        kfold_model = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle,
                                      random_state=random_state)
    else:
        kfold_model = KFold(n_splits=self.n_splits, shuffle=self.shuffle,
                            random_state=random_state)

    if sample_weight is None:
        return kfold_model.split(X, y)
    else:
        random_state = self.random_state
        kfold_model.shuffle = True
        kfold_model.random_state = random_state

    weights_sum = np.sum(sample_weight)
    max_deviations = []
    all_splits = []
    for _ in range(self.n_trials + 1):
        splits = [test for (train, test) in list(kfold_model.split(X, y))]
        weight_fracs = np.array([np.sum(sample_weight[split]) / weights_sum for split in splits])
        if np.all(weight_fracs > .95 / self.n_splits):
            # Found a good split, return.
            return self._get_folds_from_splits(splits, X.shape[0])
        # Record all splits in case the stratification by weight yeilds a worse partition
        all_splits.append(splits)
        max_deviation = np.max(np.abs(weight_fracs - 1 / self.n_splits))
        max_deviations.append(max_deviation)
        # Reseed random generator and try again
        if isinstance(kfold_model.random_state, numbers.Integral):
            kfold_model.random_state = kfold_model.random_state + 1
        elif kfold_model.random_state is not None:
            kfold_model.random_state = np.random.RandomState(kfold_model.random_state.randint(np.iinfo(np.int32).max))

    # If KFold fails after n_trials, we try the next best thing: stratifying by weight groups
    warnings.warn("The KFold algorithm failed to find a weight-balanced partition after " +
                  "{n_trials} trials. Falling back on a weight stratification algorithm.".format(
                      n_trials=self.n_trials), UserWarning)
    if is_stratified:
        stratified_weight_splits = [[]] * self.n_splits
        for y_unique in np.unique(y.flatten()):
            class_inds = np.argwhere(y == y_unique).flatten()
            class_splits = self._get_splits_from_weight_stratification(sample_weight[class_inds])
            stratified_weight_splits = [split + list(class_inds[class_split]) for split, class_split in zip(
                stratified_weight_splits, class_splits)]
    else:
        stratified_weight_splits = self._get_splits_from_weight_stratification(sample_weight)
    weight_fracs = np.array([np.sum(sample_weight[split]) / weights_sum for split in stratified_weight_splits])

    if np.all(weight_fracs > .95 / self.n_splits):
        # Found a good split, return.
        return self._get_folds_from_splits(stratified_weight_splits, X.shape[0])
    else:
        # Did not find a good split
        # Record the devaiation for the weight-stratified split to compare with KFold splits
        all_splits.append(stratified_weight_splits)
        max_deviation = np.max(np.abs(weight_fracs - 1 / self.n_splits))
        max_deviations.append(max_deviation)
    # Return most weight-balanced partition
    min_deviation_index = np.argmin(max_deviations)
    return self._get_folds_from_splits(all_splits[min_deviation_index], X.shape[0])


class WeightedKFold:
    """K-Folds cross-validator for weighted data.

    Provides train/test indices to split data in train/test sets.
    Split dataset into k folds of roughly equal size and equal total weight.

    The default is to try sklearn.model_selection.KFold a number of trials to find
    a weight-balanced k-way split. If it cannot find such a split, it will fall back
    onto a more rigorous weight stratification algorithm.

    Parameters
    ----------
    n_splits : int, default 3
        Number of folds. Must be at least 2.

    n_trials : int, default 10
        Number of times to try sklearn.model_selection.KFold before falling back to another
        weight stratification algorithm.

    shuffle : bool, optional
        Whether to shuffle the data before splitting into batches.

    random_state : int, RandomState instance, or None, default None
            If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by :mod:`np.random<numpy.random>`. Used when ``shuffle`` == True.
    """

    def __init__(self, n_splits=3, n_trials=10, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.n_trials = n_trials
        self.random_state = random_state

    def split(self, X, y, sample_weight=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array_like, shape (n_samples,)
            The target variable for supervised learning problems.

        sample_weight : array_like, shape (n_samples,)
            Weights associated with the training data.
        """
        return _split_weighted_sample(self, X, y, sample_weight, is_stratified=False)

    def get_n_splits(self, X, y, groups=None):
        """Return the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.

        y : object
            Always ignored, exists for compatibility.

        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits

    def _get_folds_from_splits(self, splits, sample_size):
        folds = []
        sample_indices = np.arange(sample_size)
        for it in range(self.n_splits):
            folds.append([np.setdiff1d(sample_indices, splits[it], assume_unique=True), splits[it]])
        return folds

    def _get_splits_from_weight_stratification(self, sample_weight):
        # Weight stratification algorithm
        # Sort weights for weight strata search
        random_state = check_random_state(self.random_state)
        sorted_inds = np.argsort(sample_weight)
        sorted_weights = sample_weight[sorted_inds]
        max_split_size = sorted_weights.shape[0] // self.n_splits
        max_divisible_length = max_split_size * self.n_splits
        sorted_inds_subset = np.reshape(sorted_inds[:max_divisible_length], (max_split_size, self.n_splits))
        shuffled_sorted_inds_subset = np.apply_along_axis(random_state.permutation, axis=1, arr=sorted_inds_subset)
        splits = [list(shuffled_sorted_inds_subset[:, i]) for i in range(self.n_splits)]
        if max_divisible_length != sorted_weights.shape[0]:
            # There are some leftover indices that have yet to be assigned
            subsample = sorted_inds[max_divisible_length:]
            if self.shuffle:
                random_state.shuffle(subsample)
            new_splits = np.array_split(subsample, self.n_splits)
            random_state.shuffle(new_splits)
            # Append stratum splits to overall splits
            splits = [split + list(new_split) for split, new_split in zip(splits, new_splits)]
        return splits


class WeightedStratifiedKFold(WeightedKFold):
    """Stratified K-Folds cross-validator for weighted data.

    Provides train/test indices to split data in train/test sets.
    Split dataset into k folds of roughly equal size and equal total weight.

    The default is to try sklearn.model_selection.StratifiedKFold a number of trials to find
    a weight-balanced k-way split. If it cannot find such a split, it will fall back
    onto a more rigorous weight stratification algorithm.

    Parameters
    ----------
    n_splits : int, default 3
        Number of folds. Must be at least 2.

    n_trials : int, default 10
        Number of times to try sklearn.model_selection.StratifiedKFold before falling back to another
        weight stratification algorithm.

    shuffle : bool, optional
        Whether to shuffle the data before splitting into batches.

    random_state : int, RandomState instance, or None, default None
            If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by :mod:`np.random<numpy.random>`. Used when ``shuffle`` == True.
    """

    def split(self, X, y, sample_weight=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array_like, shape (n_samples,)
            The target variable for supervised learning problems.

        sample_weight : array_like, shape (n_samples,)
            Weights associated with the training data.
        """
        return _split_weighted_sample(self, X, y, sample_weight, is_stratified=True)

    def get_n_splits(self, X, y, groups=None):
        """Return the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.

        y : object
            Always ignored, exists for compatibility.

        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits

class SearchEstimatorList(BaseEstimator):
    """
    Parameters
    ----------
        estimator_list: 
            List of estimator names to be used for searching. Default is ['linear', 'forest'].
        
        param_grid_list: 
            List of dictionaries or 'auto' for hyperparameters of each estimator in estimator_list. 
            If 'auto', it will automatically generate hyperparameters for the estimators. Default is 'auto'.
        
        scaling: 
            Boolean, whether to scale the input data using StandardScaler. Default is True.

        is_discrete: Boolean, whether the models are discrete or not. Default is False.
        scoring: Scoring metric to be used for selecting the best estimator. Default is None.
        n_jobs: Number of CPU cores to use for parallel processing. Default is None.
        refit: Refit the best estimator with the entire dataset. Default is True.
        grid_folds: Number of folds for cross-validation. Default is 3.
        verbose: Verbosity level for the output messages. Default is 2.
        pre_dispatch: Controls the number of jobs that get dispatched during parallel execution. Default is '2*n_jobs'.
        random_state: Seed for random number generation. Default is None.
        error_score: Value to assign to the score if an error occurs in estimator fitting. Default is np.nan.
        return_train_score: Whether to include training scores in the cv_results attribute. Default is False.
    """
    def __init__(self, estimator_list = ['linear', 'forest'], param_grid_list = None, scaling=True, is_discrete=False, scoring=None,
                 n_jobs=None, refit=True, grid_folds=3, verbose=2, pre_dispatch='2*n_jobs', random_state=None,
                 error_score=np.nan, return_train_score=False):
        
        self.estimator_list = estimator_list 
        self.complete_estimator_list = get_complete_estimator_list(clone(estimator_list, safe=False), 'discrete' if is_discrete else 'continuous')

        # TODO Add in more functionality by checking if it's an empty list. If it's just 1 dictionary then we're going to need to turn it into a list
        # Just do more cases
        if param_grid_list == 'auto':
            self.param_grid_list = auto_hyperparameters(estimator_list=self.complete_estimator_list, is_discrete=is_discrete)
        elif (param_grid_list == None):
            self.param_grid_list = len(self.complete_estimator_list) * [{}]
        else:
            self.param_grid_list = param_grid_list
        # self.categorical_indices = categorical_indices
        self.scoring = scoring
        if scoring == None:
            if is_discrete:
                self.scoring = 'f1_macro'
            else:
                self.scoring = 'neg_mean_squared_error'
            warnings.warn(f"No scoring value was given. Using default score method {self.scoring}.")
        self.scaling=scaling
        self.n_jobs = n_jobs
        self.refit = refit
        self.grid_folds = grid_folds
        self.verbose = verbose
        self.random_state = random_state
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.return_train_score = return_train_score
        self.is_discrete = is_discrete

    def fit(self, X, y, *, sample_weight=None, groups=None):
        """
        Perform cross-validation on the estimator list.
        """
        self._search_list = []
        
        if self.scaling:
            if not is_data_scaled(X):
                self.scaler = StandardScaler()
                self.scaler.fit(X)
                scaled_X = self.scaler.transform(X)

        for estimator, param_grid in zip(self.complete_estimator_list, self.param_grid_list):
            try:
                if self.random_state != None:
                    if has_random_state(model=estimator):
                        # For a polynomial pipeline, you have to set the random state of the linear part, the polynomial part doesn't have random state
                        if is_polynomial_pipeline(estimator):
                            estimator = estimator.set_params(linear__random_state=42)
                        else:
                            estimator.set_params(random_state=42)
                print(estimator)
                print(param_grid)
                
                temp_search = GridSearchCV(estimator, param_grid, scoring=self.scoring,
                                       n_jobs=self.n_jobs, refit=self.refit, cv=self.grid_folds, verbose=self.verbose,
                                       pre_dispatch=self.pre_dispatch, error_score=self.error_score,
                                       return_train_score=self.return_train_score)
                if self.scaling: # is_linear_model(estimator) and
                    if is_polynomial_pipeline(estimator=estimator):
                        temp_search.fit(scaled_X, y, groups=groups, linear__sample_weight=sample_weight)
                    elif is_mlp(estimator=estimator):
                        temp_search.fit(scaled_X, y, groups=groups)
                    else:
                        temp_search.fit(scaled_X, y, groups=groups, sample_weight=sample_weight) # , groups=groups, sample_weight=sample_weight
                    self._search_list.append(temp_search)
                else:
                    if is_polynomial_pipeline(estimator=estimator):
                        temp_search.fit(X, y, groups=groups, linear__sample_weight=sample_weight)
                    elif is_mlp(estimator=estimator):
                        temp_search.fit(X, y, groups=groups)
                    elif not supports_sample_weight(estimator=estimator):
                        temp_search.fit(X, y, groups=groups)
                    else:
                        temp_search.fit(X, y,  groups=groups, sample_weight=sample_weight)
                    self._search_list.append(temp_search)
            except (ValueError, TypeError, FitFailedWarning) as e:
                warning_msg = f"Warning: {e} for estimator {estimator} and param_grid {param_grid}"
                warnings.warn(warning_msg, category=UserWarning)
            if not hasattr(temp_search, 'cv_results_'):
                warning_msg = f"Warning: estimator {estimator} and param_grid {param_grid} failed"
                warnings.warn(warning_msg, category=FitFailedWarning)
        self.best_ind_ = np.argmax([search.best_score_ for search in self._search_list])
        self.best_estimator_ = self._search_list[self.best_ind_].best_estimator_
        self.best_score_ = self._search_list[self.best_ind_].best_score_
        self.best_params_ = self._search_list[self.best_ind_].best_params_
        print(f'Best estimator {self.best_estimator_} and best score {self.best_score_} and best params {self.best_params_}')
        return self
    
    def scaler_transform(self, X):
        if self.scaling:    
            return self.scaler.transform(X)
    def best_model(self):
        return self.best_estimator_
    def predict(self, X):
        if self.scaling:    
            return self.best_estimator_.predict(self.scaler.transform(X))
        return self.best_estimator_.predict(X)
    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)
    
class GridSearchCVList(BaseEstimator):
    """ An extension of GridSearchCV that allows for passing a list of estimators each with their own
    parameter grid and returns the best among all estimators in the list and hyperparameter in their
    corresponding grid. We are only changing the estimator parameter to estimator_list and the param_grid
    parameter to be a list of parameter grids. The rest of the parameters are the same as in
    :meth:`~sklearn.model_selection.GridSearchCV`. See the documentation of that class
    for explanation of the remaining parameters.

    Parameters
    ----------
    estimator_list : list of estimator object.
        Each estimator in th list is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_grid : list of dict or list of list of dictionaries
        For each estimator, the dictionary with parameters names (`str`) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.
    """

    def __init__(self, estimator_list = ['linear', 'forest'], param_grid_list = 'auto', scoring=None,
                 n_jobs=None, refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs',
                 error_score=np.nan, return_train_score=False, is_discrete= False):
        # 'discrete' if is_discrete else 'continuous'
        self.estimator_list = get_complete_estimator_list(estimator_list, 'continuous')
        if param_grid_list == 'auto':
            self.param_grid_list = auto_hyperparameters(estimator_list=self.estimator_list, is_discrete=is_discrete)
        elif (param_grid_list == None):
            self.param_grid_list = len(self.estimator_list) * [{}]
        else:
            self.param_grid_list = param_grid_list
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.cv = cv
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.return_train_score = return_train_score
        return

    def fit(self, X, y=None, **fit_params):
        self._gcv_list = [GridSearchCV(estimator, param_grid, scoring=self.scoring,
                                       n_jobs=self.n_jobs, refit=self.refit, cv=self.cv, verbose=self.verbose,
                                       pre_dispatch=self.pre_dispatch, error_score=self.error_score,
                                       return_train_score=self.return_train_score)
                          for estimator, param_grid in zip(self.estimator_list, self.param_grid_list)]
        self.best_ind_ = np.argmax([gcv.fit(X, y, **fit_params).best_score_ for gcv in self._gcv_list])
        self.best_estimator_ = self._gcv_list[self.best_ind_].best_estimator_
        self.best_score_ = self._gcv_list[self.best_ind_].best_score_
        self.best_params_ = self._gcv_list[self.best_ind_].best_params_
        return self
    
    def best_model(self):
        return self.best_estimator_

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)


def _cross_val_predict(estimator, X, y=None, *, groups=None, cv=3,
                       n_jobs=None, verbose=0, fit_params=None,
                       pre_dispatch='2*n_jobs', method='predict', safe=True):
    """This is a fork from :meth:`~sklearn.model_selection.cross_val_predict` to allow for
    non-safe cloning of the models for each fold.

    Parameters
    ----------
    estimator : estimator object implementing 'fit' and 'predict'
        The object to use to fit the data.

    X : array_like of shape (n_samples, n_features)
        The data to fit. Can be, for example a list, or an array at least 2d.

    y : array_like of shape (n_samples,) or (n_samples, n_outputs), \
            default=None
        The target variable to try to predict in the case of
        supervised learning.

    groups : array_like of shape (n_samples,), optional
        Group labels for the samples used while splitting the dataset into
        train/test set. Only used in conjunction with a "Group" :term:`cv`
        instance (e.g., :class:`GroupKFold`).

    cv : int, cross-validation generator or an iterable, default None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - CV splitter,
        - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

    n_jobs : int, default None
        The number of CPUs to use to do the computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : int, default 0
        The verbosity level.

    fit_params : dict, defualt=None
        Parameters to pass to the fit method of the estimator.

    pre_dispatch : int or str, default '2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    method : str, default 'predict'
        Invokes the passed method name of the passed estimator. For
        method='predict_proba', the columns correspond to the classes
        in sorted order.

    safe : bool, default True
        Whether to clone with safe option.

    Returns
    -------
    predictions : ndarray
        This is the result of calling ``method``
    """
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    splits = list(cv.split(X, y, groups))

    test_indices = np.concatenate([test for _, test in splits])
    if not _check_is_permutation(test_indices, _num_samples(X)):
        raise ValueError('cross_val_predict only works for partitions')

    # If classification methods produce multiple columns of output,
    # we need to manually encode classes to ensure consistent column ordering.
    encode = method in ['decision_function', 'predict_proba',
                        'predict_log_proba'] and y is not None
    if encode:
        y = np.asarray(y)
        if y.ndim == 1:
            le = LabelEncoder()
            y = le.fit_transform(y)
        elif y.ndim == 2:
            y_enc = np.zeros_like(y, dtype=int)
            for i_label in range(y.shape[1]):
                y_enc[:, i_label] = LabelEncoder().fit_transform(y[:, i_label])
            y = y_enc

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    predictions = parallel(delayed(_fit_and_predict)(
        clone(estimator, safe=safe), X, y, train, test, verbose, fit_params, method)
        for train, test in splits)
    from pkg_resources import parse_version
    if parse_version(sklearn.__version__) < parse_version("0.24.0"):
        # Prior to 0.24.0, this private scikit-learn method returned a tuple of two values
        predictions = [p[0] for p in predictions]

    inv_test_indices = np.empty(len(test_indices), dtype=int)
    inv_test_indices[test_indices] = np.arange(len(test_indices))

    if sp.issparse(predictions[0]):
        predictions = sp.vstack(predictions, format=predictions[0].format)
    elif encode and isinstance(predictions[0], list):
        # `predictions` is a list of method outputs from each fold.
        # If each of those is also a list, then treat this as a
        # multioutput-multiclass task. We need to separately concatenate
        # the method outputs for each label into an `n_labels` long list.
        n_labels = y.shape[1]
        concat_pred = []
        for i_label in range(n_labels):
            label_preds = np.concatenate([p[i_label] for p in predictions])
            concat_pred.append(label_preds)
        predictions = concat_pred
    else:
        predictions = np.concatenate(predictions)

    if isinstance(predictions, list):
        return [p[inv_test_indices] for p in predictions]
    else:
        return predictions[inv_test_indices]



    
        


