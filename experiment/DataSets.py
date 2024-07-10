import os
import glob
import logging

import pandas as pd
from Utils import make_col_null_with_ratio, check_dir

if __name__ == '__main__':
    # 生成缺失值数据集，不同缺失率
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # datasets = glob.glob("dataset/*.csv")
    datasets = ["dataset/Crystal_structure.csv"]
    check_dir("dataset/miss_datasets")
    for dataset in datasets:
        df_full = pd.read_csv(dataset)
        df_full.dropna(how='any', inplace=True)
        numeric_columns = df_full.select_dtypes(include=['float64', 'int64']).columns.tolist()
        logging.info("Dataset: {}, numeric columns: {}".format(dataset, numeric_columns))

        data_root_dir = os.path.join("dataset/miss_datasets", os.path.basename(dataset).split(".")[0])
        check_dir(data_root_dir)
        for col in numeric_columns:
            for i in range(1, 10):
                miss_ratio = i * 0.1
                df_miss = make_col_null_with_ratio(df_full, col, miss_ratio)

                # check col exist '/' or not
                if '/' in col:
                    col_save_name = col.replace('/', '_')
                else:
                    col_save_name = col
                df_miss.to_csv(os.path.join(data_root_dir, "{}_{}.csv".format(col_save_name, miss_ratio)), index=False)
                logging.info("Dataset: {}, column: {}, miss ratio: {}".format(dataset, col_save_name, miss_ratio))
