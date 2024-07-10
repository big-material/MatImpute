import glob
import logging
import os
import time

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from scipy.stats import wasserstein_distance

from Impute import fill_with_dt, fill_with_rf, fill_with_gbr, fill_with_lgb, fill_with_xgb, fill_with_hyperimpute, \
    fill_with_mice, fill_with_ridge, fill_with_et

sns.set_theme(style="darkgrid")

from Utils import check_dir

if __name__ == '__main__':
    # a = pd.DataFrame([[1, 2, 3], [4, np.NAN, np.NAN], [7, 8, 9]], columns=['a', 'b', 'c'])
    # miss_index = a[a['b'].isnull()].index.tolist()
    #
    # # print(a['b', miss_index])
    # print(a.loc[miss_index, 'b'].values)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # datasets = glob.glob("dataset/*.csv")
    datasets = ["dataset/Crystal_structure.csv"]
    check_dir("dataset/filled_results")
    for dataset in tqdm(datasets):
        dataset_name = os.path.basename(dataset).split(".")[0]
        df_full = pd.read_csv(dataset)
        miss_dataset_root_dir = os.path.join("dataset/miss_datasets", dataset_name)
        numeric_columns = df_full.select_dtypes(include=['float64', 'int64']).columns.tolist()
        df_full = df_full[numeric_columns]
        result_cols = ['dataset', 'column', 'miss_ratio', 'method', 'RMSE', 'Wasserstein', 'time']
        dataset_filled_result = pd.DataFrame(columns=result_cols)

        for col in numeric_columns:
            for i in range(1, 10):
                miss_ratio = i * 0.1
                if i == 3:
                    miss_ratio = 0.30000000000000004
                elif i == 6:
                    miss_ratio = 0.6000000000000001
                elif i == 7:
                    miss_ratio = 0.7000000000000001
                if "/" in col:
                    col_save_name = col.replace("/", "_")
                else:
                    col_save_name = col
                df_miss = pd.read_csv(
                    os.path.join(miss_dataset_root_dir, "{}_{}.csv".format(col_save_name, miss_ratio)))
                df_miss = df_miss[numeric_columns]
                miss_index = df_miss[df_miss[col].isnull()].index.tolist()

                methods = [fill_with_dt, fill_with_rf, fill_with_gbr, fill_with_lgb, fill_with_xgb, fill_with_ridge,
                           fill_with_et]
                for method in methods:
                    start_time = time.time()
                    df_filled = method(df_miss, col)
                    end_time = time.time()
                    time_elapsed = end_time - start_time
                    y_ture = df_full.loc[miss_index, col].values
                    y_pred = df_filled.loc[miss_index, col].values

                    rmse = np.sqrt(mean_squared_error(y_ture, y_pred))
                    wasserstein = wasserstein_distance(y_ture, y_pred)
                    method_result = pd.DataFrame([[dataset_name, col, miss_ratio, method.__name__,
                                                   rmse, wasserstein, time_elapsed]], columns=result_cols)
                    dataset_filled_result = pd.concat([method_result, dataset_filled_result], ignore_index=True)
        dataset_filled_result.to_csv(os.path.join("dataset/filled_results", "{}.csv".format(dataset_name)),
                                     index=False)
        logging.info("Dataset: {} finished".format(dataset_name))
