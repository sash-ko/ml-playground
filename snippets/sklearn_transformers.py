from operator import lshift
from typing import List

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline, make_union, make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def get_date_feature(df: pd.DataFrame, op: str, date_col: str) -> np.ndarray:
    """Extracts date related features from date column"""
    return getattr(df[date_col].dt, op).values.reshape(-1, 1)


def create_simple_pipeline(
    cat_features: List,
    passthrough: List,
    date_col: str,
    **fit_params
) -> Pipeline:
    """Combines transformers and builds a prediction pipeline"""

    # one hot encoded date features
    transform_encoded_date_features = make_pipeline(
        make_union(
            FunctionTransformer(get_date_feature, kw_args={
                                'op': 'weekofyear', 'date_col': date_col}),
            FunctionTransformer(get_date_feature, kw_args={
                                'op': 'year', 'date_col': date_col}),
            FunctionTransformer(get_date_feature, kw_args={
                                'op': 'month', 'date_col': date_col}),
        ),
        OneHotEncoder(handle_unknown='ignore')
    )

    # numerical date features
    transform_date_features = make_pipeline(
        FunctionTransformer(get_date_feature, kw_args={'op': 'day', 'date_col': date_col}))

    # pass selected columns without any changes
    transform_passthrough = make_column_transformer(
        ('passthrough', passthrough))

    # create a union of date, passthrough and categorical features
    transform_features = make_union(
        transform_date_features,
        transform_encoded_date_features,
        transform_passthrough,
        make_column_transformer(
            (OneHotEncoder(handle_unknown='ignore'), cat_features)
        )
    )

    regressor = lgb.LGBMRegressor(**fit_params)
    regressor = TransformedTargetRegressor(
        regressor, func=np.log1p, inverse_func=np.exp
    )
    return make_pipeline(transform_features, regressor)


def make_prediction(
        df: pd.DataFrame,
        y: pd.Series,
        cat_features: List,
        passthrough: List,
        date_col: str,
        sample_weight_col: str = None
) -> np.array:

    X_train, X_test, y_train, y_test = train_test_split(df, y)

    fit_params = {}
    if sample_weight_col:
        fit_params['transformedtargetregressor__sample_weight'] = X_train[sample_weight_col]

    pipeline = create_simple_pipeline(cat_features, passthrough, date_col)
    pipeline.fit(X_train, y_train, **fit_params)

    y_pred = pipeline.predict(X_test)

    return y_pred


# create random dataset
df_data = pd.util.testing.makeMixedDataFrame()
cat_features = ['C']
passthrough = ['B']
date_col = 'D'
sample_weight_col = 'A'

# create random target
y = np.random.rand(len(df_data))


make_prediction(df_data, y, cat_features, passthrough, date_col, sample_weight_col)
