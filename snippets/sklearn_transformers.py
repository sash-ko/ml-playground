"""
The example shows how to create a prediction pipeline using building
blocks provided by sklearn - transformers and pipelines
"""

from typing import List

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.compose import TransformedTargetRegressor
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def get_date_feature(df: pd.DataFrame, dt_prop: str, date_col: str) -> np.ndarray:
    """Extracts date related features from a datetime column

    Parameters
    ----------

    df: pd.DataFrame
        Data set with a datetime column

    dt_prop: str
        Name of a date field property, e.g. "weekofyear", "month"

    date_col: str
        Name of a datetime column

    Returns
    -------

    feature: np.ndarray
    """
    feature = getattr(df[date_col].dt, dt_prop).values.reshape(-1, 1)
    return feature


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
        FunctionTransformer(get_date_feature, kw_args={
                            'op': 'day', 'date_col': date_col}))

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

    # apply log transformation to the target variable
    regressor = TransformedTargetRegressor(
        regressor, func=np.log1p, inverse_func=np.exp
    )

    # combine feature transformers and regressor in one pipeline
    return make_pipeline(transform_features, regressor)


def get_feature_names(pipeline: Pipeline) -> List:
    """Extract feature names from a pipeline"""
    
    names = []
    
    if hasattr(pipeline, 'steps'):
        for step in pipeline.steps:
            names.extend(get_feature_names(step[1]))
            
    elif hasattr(pipeline, 'transformer_list'):
        for tr in pipeline.transformer_list:
            names.extend(get_feature_names(tr[1]))
            
    elif hasattr(pipeline, 'get_feature_names'):
        names.extend(pipeline.get_feature_names())
        
    return names


def make_prediction(
        df: pd.DataFrame,
        y: pd.Series,
        cat_features: List,
        passthrough: List,
        date_col: str,
        sample_weight_col: str = None
) -> np.array:

    X_train, X_test, y_train, y_test = train_test_split(df, y)

    # pass additional parameters to model fit
    fit_params = {}
    if sample_weight_col:
        # 'transformedtargetregressor' is the name
        # of the last step of the pipeline generated automatically
        fit_params['transformedtargetregressor__sample_weight'] = X_train[sample_weight_col]

    pipeline = create_simple_pipeline(cat_features, passthrough, date_col)

    pipeline.fit(X_train, y_train, **fit_params)
    
    print(f'Features: {get_feature_names(pipeline)}')
    
    y_pred = pipeline.predict(X_test)
    print('MSE:', mean_squared_error(y_test, y_pred))


if __name__ == "__main__":

    # create a fake dataset
    df_data = pd.util.testing.makeMixedDataFrame()

    cat_features = ['C']
    passthrough = ['B']
    date_col = 'D'
    sample_weight_col = 'A'

    # create a random target
    y = np.random.rand(len(df_data))

    make_prediction(df_data, y, cat_features, passthrough, date_col, sample_weight_col)
