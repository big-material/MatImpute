import glob
import math
from pathlib import Path
import scienceplots
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

plt.style.use(['nature', "no-latex", 'grid'])


# plt.rcParams['font.family'] = 'sans-serif'


def list_replace(l, old, new):
    for i, v in enumerate(l):
        if v == old:
            l[i] = new
    return l


if __name__ == '__main__':
    # datasets = glob.glob("./dataset/filled_results/*.csv")
    datasets = ["./dataset/filled_results/CCPM.csv"]
    for dataset in datasets:
        # get the filename
        dataset_name = Path(dataset).stem

        fill_result_df = pd.read_csv("./dataset/filled_results/{}.csv".format(dataset_name))
        cols = fill_result_df['column'].unique().tolist()
        methods = fill_result_df['method'].unique().tolist()
        # filter the method missforest and hyperimpute
        methods = list(filter(lambda x: x != 'fill_with_mice', methods))
        methods = list(filter(lambda x: x != 'fill_with_hyperimpute', methods))

        fig, axes = plt.subplots(nrows=math.ceil(len(cols) / 3), ncols=3, figsize=(10, 10), sharex=True)
        axes = axes.flatten()
        markers = ['o', 's', 'D', '^', 'v', 'p', 'P', '*', 'X', 'd']
        for col in cols:
            col_fill_result = fill_result_df[fill_result_df['column'] == col]
            ax = axes[cols.index(col)]
            for method in methods:
                method_fill_result = col_fill_result[col_fill_result['method'] == method]
                sns.lineplot(x='miss_ratio', y='RMSE', data=method_fill_result, label=method, ax=ax,
                             marker=markers[methods.index(method)], markersize=5,
                             markeredgecolor='black')
                # rmove x label
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_title(col, fontsize=16)
                # remove legend
                ax.get_legend().remove()
                ax.tick_params(labelsize=16)

        # fig set legend
        handles, labels = ax.get_legend_handles_labels()
        labels = [x[10:] for x in labels]
        labels = list_replace(labels, 'et', 'Extra Trees')
        labels = list_replace(labels, 'gbr', 'Gradient Boosting')
        labels = list_replace(labels, 'lgb', 'LightGBM')
        labels = list_replace(labels, 'rf', 'Random Forest')
        labels = list_replace(labels, 'xgb', 'XGBoost')
        labels = list_replace(labels, 'dt', 'Decision Tree')
        labels = list_replace(labels, 'ridge', 'Ridge')

        fig.legend(handles, labels, loc='lower right', fontsize=18, bbox_to_anchor=(0.98, 0.04), ncol=2)

        # remove the unused axes
        for i in range(len(cols), len(axes)):
            fig.delaxes(axes[i])

        fig.tight_layout()
        fig.text(0.008, 0.5, 'RMSE', va='center', rotation='vertical', fontsize=18)
        fig.text(0.5, 0.01, 'Missing Ratio', ha='center', fontsize=18)
        fig.subplots_adjust(left=0.07, bottom=0.07, wspace=0.2)

        plt.savefig("./dataset/filled_results/{}_RMSE.png".format(dataset_name), dpi=300)
        plt.show()
