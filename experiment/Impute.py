from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from hyperimpute.plugins.imputers import Imputers

import pandas as pd
import numpy as np
import pickle


def filled_df_except_miss_col(df_with_null, miss_col):
    knn_imputer = KNNImputer()
    df_filled = knn_imputer.fit_transform(df_with_null.copy())
    df_filled_except_miss_col = pd.DataFrame(df_filled, columns=df_with_null.columns)
    df_filled_except_miss_col[miss_col] = df_with_null[miss_col]
    return df_filled_except_miss_col


def fill_with_model(df_with_null, miss_col, model):
    """
    Fill the missing values with the model
    :param df_with_null: the dataframe with missing values
    :param miss_col: the column with missing values
    :return: the dataframe with missing values filled
    """
    # print(df_with_null.isnull().sum(), miss_col)
    df_filled_except_miss_col = filled_df_except_miss_col(df_with_null, miss_col)
    # print(df_filled_except_miss_col.isnull().sum(), miss_col)

    train_df = df_filled_except_miss_col.dropna()
    X_train = train_df.drop(miss_col, axis=1)
    y_train = train_df[miss_col]

    model.fit(X_train, y_train)

    miss_index = df_with_null[df_with_null[miss_col].isnull()].index

    predict_X = df_filled_except_miss_col.loc[miss_index].drop(miss_col, axis=1)

    predict_y = model.predict(predict_X)

    df_filled = df_with_null.copy()
    df_filled.loc[miss_index, miss_col] = predict_y
    return df_filled


def fill_with_vote(df_with_null, miss_col):
    et = ExtraTreesRegressor()
    ride = Ridge()
    vote = VotingRegressor(estimators=[('et', et), ('ride', ride)])
    return fill_with_model(df_with_null, miss_col, vote)


def fill_with_stack(df_with_null, miss_col):
    et = ExtraTreesRegressor()
    ride = Ridge()
    vote = VotingRegressor(estimators=[('et', et), ('ride', ride)])
    return fill_with_model(df_with_null, miss_col, vote)


def fill_with_dt(df_with_null, miss_col):
    """
    Fill the missing values with decision tree
    :param df_with_null: the dataframe with missing values
    :param miss_col: the column with missing values
    :return: the dataframe with missing values filled
    """
    dt = DecisionTreeRegressor()
    return fill_with_model(df_with_null, miss_col, dt)


def fill_with_rf(df_with_null, miss_col):
    """
    Fill the missing values with random forest
    :param df_with_null: the dataframe with missing values
    :param miss_col: the column with missing values
    :return: the dataframe with missing values filled
    """
    rf = RandomForestRegressor()
    return fill_with_model(df_with_null, miss_col, rf)


def fill_with_ridge(df_with_null, miss_col):
    """
    Fill the missing values with ridge regression
    :param df_with_null: the dataframe with missing values
    :param miss_col: the column with missing values
    :return: the dataframe with missing values filled
    """
    ridge = Ridge()
    return fill_with_model(df_with_null, miss_col, ridge)


def fill_with_et(df_with_null, miss_col):
    """
    Fill the missing values with extra tree
    :param df_with_null: the dataframe with missing values
    :param miss_col: the column with missing values
    :return: the dataframe with missing values filled
    """
    et = ExtraTreesRegressor()
    return fill_with_model(df_with_null, miss_col, et)


def fill_with_gbr(df_with_null, miss_col):
    """
    Fill the missing values with gradient boosting regression
    :param df_with_null: the dataframe with missing values
    :param miss_col: the column with missing values
    :return: the dataframe with missing values filled
    """
    gbr = GradientBoostingRegressor()
    return fill_with_model(df_with_null, miss_col, gbr)


def fill_with_xgb(df_with_null, miss_col):
    """
    Fill the missing values with xgboost regression
    :param df_with_null: the dataframe with missing values
    :param miss_col: the column with missing values
    :return: the dataframe with missing values filled
    """

    xgb = XGBRegressor()
    return fill_with_model(df_with_null, miss_col, xgb)


def fill_with_lgb(df_with_null, miss_col):
    """
    Fill the missing values with lightgbm regression
    :param df_with_null: the dataframe with missing values
    :param miss_col: the column with missing values
    :return: the dataframe with missing values filled
    """
    lgb = LGBMRegressor()
    return fill_with_model(df_with_null, miss_col, lgb)


def fill_with_hyperimpute(df_with_null, _):
    """
    Fill the missing values with hyperimpute
    :param df_with_null: the dataframe with missing values
    :param miss_col: the column with missing values
    :return: the dataframe with missing values filled
    """
    plugin = Imputers().get('hyperimpute')
    df_filled = plugin.fit_transform(df_with_null)
    return df_filled


def fill_with_mice(df_with_null, _):
    """
    Fill the missing values with hyperimpute
    :param df_with_null: the dataframe with missing values
    :param miss_col: the column with missing values
    :return: the dataframe with missing values filled
    """
    plugin = Imputers().get('missforest')
    df_filled = plugin.fit_transform(df_with_null)
    return df_filled


methods = [fill_with_dt, fill_with_et, fill_with_gbr, fill_with_hyperimpute, fill_with_lgb, fill_with_mice,
           fill_with_rf, fill_with_ridge, fill_with_xgb]


def fill_with_meta(df_with_null, miss_col):
    """
    Fill the missing values with meta model
    :param df_with_null: the dataframe with missing values
    :param miss_col: the column with missing values
    :return: the dataframe with missing values filled
    """
    with open('model_choice.pkl', 'rb') as f:
        meta_model = pickle.load(f)
    df_filled_except_miss_col = filled_df_except_miss_col(df_with_null, miss_col)

    col_miss_ratio = df_filled_except_miss_col[miss_col].isnull().sum() / len(df_filled_except_miss_col[miss_col])
    # print(col_miss_ratio)
    col_median = df_filled_except_miss_col[miss_col].median()
    col_mode = df_filled_except_miss_col[miss_col].mode()[0]
    col_lens = len(df_filled_except_miss_col[miss_col])
    col_skew = df_filled_except_miss_col[miss_col].skew()
    col_kurt = df_filled_except_miss_col[miss_col].kurt()

    corr = df_filled_except_miss_col.corr()[miss_col]
    corr_less_than_0 = len(corr[corr < 0]) / len(corr)
    corr_greater_than_0 = len(corr[corr >= 0]) / len(corr)

    corr_mean = corr.mean()
    corr_std = corr.std()

    col_desc = df_filled_except_miss_col[miss_col].describe()
    col_cv = col_desc['std'] / col_desc['mean']
    col_range = col_desc['max'] - col_desc['min']
    col_iqr = col_desc['75%'] - col_desc['25%']
    col_desc = col_desc.to_list()

    col_info = [col_miss_ratio] + col_desc + [col_median, col_mode, col_lens, col_skew, col_kurt, corr_less_than_0,
                                              corr_greater_than_0, corr_mean, corr_std, col_cv, col_range, col_iqr]

    choice_model_id = meta_model.predict([col_info])[0]
    choice_model = methods[choice_model_id]

    df_col_filled = choice_model(df_filled_except_miss_col, miss_col)
    df_filled = df_with_null.copy()
    df_filled[miss_col] = df_col_filled[miss_col]

    return df_filled


if __name__ == '__main__':
    from Impute import fill_with_meta, fill_with_et
    import pandas as pd
    import numpy as np
    from hyperimpute.utils.benchmarks import compare_models
    from hyperimpute.plugins.imputers import Imputers, ImputerPlugin


    class MetaImputer(ImputerPlugin):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # self._model = fill_with_meta

        @staticmethod
        def name():
            return "meta"

        @staticmethod
        def hyperparameter_space():
            return []

        def _fit(self, *args, **kwargs):
            return self

        def _transform(self, df):
            # 按照缺失值的比例进行排序
            miss_rate = df.isnull().sum() / df.shape[0]
            cols = miss_rate.sort_values().index.tolist()
            for col in cols:
                df = fill_with_et(df, col)
            return df


    df = pd.read_csv('./dataset/BMDS_data.csv')
    df = df.select_dtypes(include=[np.number])
    # 随机生成缺失值
    df_with_null = df.copy()
    df_with_null = df_with_null.mask(np.random.random(df_with_null.shape) < 0.2)
    # print(df_with_null)

    imputers = Imputers()
    imputers.add("meta", MetaImputer)
    imputer = imputers.get("meta")
    df_filled = imputer.fit_transform(df_with_null)
    # print(df_filled)
    # compare_models(
    #     name="example",
    #     evaluated_model=imputer,
    #     X_raw=df.copy(),
    #     ref_methods=["mean", "missforest"],
    #     scenarios=["MAR"],
    #     miss_pct=[0.1, 0.2, 0.3, 0.4, 0.5],
    #     n_iter=2)
    # print("RMSE: ", compare_models(df, df_with_null, imputer, "RMSE"))
