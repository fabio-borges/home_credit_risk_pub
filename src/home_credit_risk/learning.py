from collections import defaultdict
from typing import Any
import numpy as np
import pandas as pd
from numpy import floating
from sklearn.model_selection import cross_validate, StratifiedKFold
import xgboost as xgb


def cross_validate_xgboost_model(model: xgb.XGBClassifier,
                                 X: pd.DataFrame,
                                 y: pd.Series,
                                 random_state: int,
                                 n_folds: int = 5,
                                 n_jobs: int = -1) -> dict:
    """
    This is a wrapper for the sklearn's model_selection.cross_validation function. It cross validates the XGClassifier model
    and returns the result dictionary gotten from sklearn's cross_validation function, with an added entry for the
    feature importance map. The importance values are the average importances computed over the models fitted for
    each CV fold. The importance metric is 'total_gain'.
    The cross validation is stratified on the target.
    :param model: The XGBClassifier model to fit.
    :param X: The features.
    :param y: The target.
    :param random_state:
    :param n_folds:
    :param n_jobs:
    :return: The sklearn.model_selection.cross_validation result dictionary plus an entry with key='xgb_total_gain' and
        the value being the feature importance ('total_gain') map.
    """

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    result = cross_validate(model, X, y, cv=cv, scoring='roc_auc',
                            n_jobs=n_jobs, return_train_score=True,
                            return_estimator=True)

    estimators = result['estimator']
    result['xgb_total_gain'] = compute_mean_total_gain(estimators)
    return result

def compute_mean_total_gain(estimators) -> dict[str, floating[Any]]:
    importances = defaultdict(list)
    for estimator in estimators:
        importance_dict = estimator.get_booster().get_score(importance_type='total_gain')
        for feature, importance in importance_dict.items():
            importances[feature].append(importance)
    return {feature: np.mean(importance) for feature, importance in importances.items()}