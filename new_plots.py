import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_each_criteria():
    df = pd.read_excel(file_path)

    criteria_list = df['Criteria'].unique()
    bar_width = 0.5

    colors = ['#00008B', '#FF8C00', '#8B0000', '#9932CC', '#8B4513', '#FF0005']
    for ind, criterion in enumerate(criteria_list):
        fig, ax = plt.subplots(figsize=(8, 6))
        subset = df[df['Criteria'] == criterion]
        print(subset)

        plt.bar(subset['Method'], subset['Values'].values, bar_width, color=colors[ind])

        plt.xlabel('Method')
        plt.ylabel('Values')
        plt.title(f'{criterion}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def plot_all_criteria():
    df = pd.read_excel(file_path)

    methods = df['Method'].unique()
    criteria = df['Criteria'].unique()
    num_methods = len(methods)
    num_criteria = len(criteria)

    data = {crit: [] for crit in criteria}

    for method in methods:
        method_data = df[df['Method'] == method]
        for crit in criteria:
            value_series = method_data[method_data['Criteria'] == crit]['Values']
            if value_series.empty:
                data[crit].append(0)
            else:
                data[crit].append(float(value_series.iloc[0]))

    fig, ax = plt.subplots(figsize=(12, 8))
    bar_width = 0.15
    index = np.arange(num_methods)

    colors_bars = ['#00008B', '#FF8C00', '#8B0000', '#9932CC', '#8B4513', '#FF0005']

    for i, crit in enumerate(criteria):
        ax.bar(index + i * bar_width, data[crit], bar_width, label=crit, color=colors_bars[i % len(colors_bars)])

    ax.set_xlabel('Methods')
    ax.set_ylabel('Values')
    ax.set_title('Comparison of Methods by Various Criteria')
    ax.set_xticks(index + bar_width * (num_criteria - 1) / 2)
    ax.set_xticklabels(methods, rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    file_path = 'results/result_RealTimeTestRandomGoal_20240628_000352.xlsx'
    plot_each_criteria()
    plot_all_criteria()
