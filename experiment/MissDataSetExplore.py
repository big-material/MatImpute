import glob
import math
import os

import pandas as pd
from matplotlib import pyplot as plt
from ydata_profiling import ProfileReport
import seaborn as sns
import scienceplots

plt.style.use(['nature', 'no-latex', "grid"])

from Utils import check_dir

if __name__ == '__main__':
    # datasets = glob.glob("dataset/*.csv")
    datasets = ["dataset/CCPM.csv"]
    check_dir("dataset/data_report")
    for dataset in datasets:
        dataset_name = os.path.basename(dataset).split(".")[0]
        df_full = pd.read_csv(dataset)
        # check profile, if exist then skip
        if not os.path.exists(os.path.join("dataset/data_report", "{}.html".format(dataset_name))):
            profile = ProfileReport(df_full, title=dataset_name)
            profile.to_file(os.path.join("dataset/data_report", "{}.html".format(dataset_name)))
        miss_dataset_root_dir = os.path.join("dataset/miss_datasets", dataset_name)
        numeric_columns = df_full.select_dtypes(include=['float64', 'int64']).columns.tolist()
        fig, axes = plt.subplots(math.ceil(len(numeric_columns) / 4), 4, figsize=(10, 10))
        axes = axes.flatten()
        for i, col in enumerate(numeric_columns):
            ax = axes[i]
            sns.kdeplot(df_full[col].dropna(), ax=ax)
            for i in range(1, 10):
                miss_ratio = i * 0.1
                col_save_name = col
                if "/" in col:
                    col_save_name = col.replace("/", "_")

                df_miss = pd.read_csv(
                    os.path.join(miss_dataset_root_dir, "{}_{}.csv".format(col_save_name, miss_ratio)))
                sns.kdeplot(df_miss[col].dropna(), ax=ax)
            ax.set_xlabel(col, fontsize=14)
            ax.set_ylabel("")
            ax.tick_params(axis='both', which='major', labelsize=14)

        # figure remove not used axes
        for i in range(len(numeric_columns), len(axes)):
            fig.delaxes(axes[i])
        fig.legend(['full', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%'], loc='lower center', ncol=5,
                   fontsize=15, bbox_to_anchor=(0.5, 0.01))
        # set common y label
        fig.text(0.01, 0.5, 'Density', va='center', rotation='vertical', fontsize=20)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.15, wspace=0.36, hspace=0.3)
        fig.savefig(os.path.join("dataset/data_report", "{}.png".format(dataset_name)), dpi=300)
        plt.show()
