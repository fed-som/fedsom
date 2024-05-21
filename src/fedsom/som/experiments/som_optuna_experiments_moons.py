import warnings
from itertools import product

import matplotlib.pyplot as plt
import optuna
import pandas as pd
import plotly.express as px
import plotly.io as pio
import sklearn
import torch
from optuna.visualization import plot_optimization_history
from sklearn.cluster import KMeans

from deepclustering.som.experiments.two_moons import gen_moons
from deepclustering.som.som import SelfOrganizingMap
from deepclustering.som.utils.som_utils import bmu_to_string, convert_bmu_labels

warnings.filterwarnings("ignore")


def dict_2_string(dict_):
    string = ""
    for k, v in dict_.items():
        if isinstance(v, int):
            string += f"{k}_{v}__"
        else:
            string += f"{k}_{v:.6}__"
    return string[:-2]


def create_datasets(n_samples_values, centers_values):
    parameter_combinations = product(n_samples_values, centers_values)

    datasets = []
    for params in parameter_combinations:
        n_samples, centers = params

        X, labels = gen_moons(n_samples=n_samples, centers=centers, random_state=42)

        dataset = {"parameters": {"n_samples": n_samples, "centers": centers}, "X": X, "labels": labels}

        yield dataset


def objective(trial, X, labels_true):
    grid_edge_length = trial.suggest_int("grid_edge_length", 3, 4)
    grid_dim = trial.suggest_int("grid_dim", 2, 2)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1.0)
    sigma = trial.suggest_loguniform("sigma", 1e-5, 10.0)
    num_epochs = trial.suggest_int("num_epochs", 1, 30, log=False)
    grid_size = tuple([grid_edge_length for _ in range(grid_dim)])

    som = SelfOrganizingMap(grid_size, input_size=2, learning_rate=learning_rate, sigma=sigma)
    som.train(torch.tensor(X), num_epochs=num_epochs)

    bmu_labels = {}
    for n, x in enumerate(X):
        units = som.find_best_matching_unit(torch.tensor(x))
        bmu_labels[n] = bmu_to_string(units)

    labels = convert_bmu_labels(bmu_labels)

    if len(set(labels)) == 1:
        return 0
    else:
        return sklearn.metrics.calinski_harabasz_score(X, labels)
        # return sklearn.metrics.adjusted_mutual_info_score(labels_true,labels)


# Create a wrapper function to pass additional arguments
def wrapped_objective(arg1, arg2):
    return lambda trial: objective(trial, arg1, arg2)


def som_cluster(X, params):
    grid_edge_length = params["grid_edge_length"]
    grid_dim = params["grid_dim"]
    learning_rate = params["learning_rate"]
    sigma = params["sigma"]
    num_epochs = params["num_epochs"]

    grid_size = tuple([grid_edge_length for _ in range(grid_dim)])
    som = SelfOrganizingMap(grid_size, input_size=2, learning_rate=learning_rate, sigma=sigma)
    som.train(torch.tensor(X), num_epochs=num_epochs)

    bmu_labels = {}
    for n, x in enumerate(X):
        units = som.find_best_matching_unit(torch.tensor(x))
        bmu_labels[n] = bmu_to_string(units)

    return convert_bmu_labels(bmu_labels)


def scatter_plot(X, labels, best_params_string, filepath):
    fig = plt.figure(figsize=(14, 8))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab20")  #'viridis')
    plt.title(best_params_string, fontsize=14, fontweight="bold")
    plt.colorbar(label="Cluster")
    plt.show(block=False)
    plt.savefig(filepath)
    plt.close()


if __name__ == "__main__":
    n_samples_values = [1000]
    centers_values = [10]
    n_trials = 100
    results = []
    datasets = create_datasets(n_samples_values, centers_values)
    results_dir = "../../sandbox/images/optuna/temp/"

    for dataset in datasets:
        parameters = dataset["parameters"]
        X = dataset["X"]
        labels_true = dataset["labels"]
        run_name = dict_2_string(parameters)
        result = {"run_name": run_name}

        # Create an Optuna study
        study = optuna.create_study(direction="maximize")

        # Optimize the objective function
        study.optimize(wrapped_objective(X, labels_true), n_trials=n_trials)

        # Print the best parameters and objective value
        best_params = study.best_params
        best_value = study.best_value
        print(f"Best parameters: {best_params}")
        print(f"Best value: {best_value}")

        # Plot labeled data
        best_labels = som_cluster(X, best_params)
        best_params_string = dict_2_string(best_params)
        filepath = f"{results_dir}{run_name}___{best_params_string}_scatter.png"
        scatter_plot(X, best_labels, best_params_string, filepath)

        # record run
        result = {"run_name": run_name}
        result.update(parameters)
        result.update(best_params)
        result["best_value"] = best_value
        results.append(result)

        # Create a color scale using the 'Viridis' color scale
        fig = optuna.visualization.plot_parallel_coordinate(
            study, params=["grid_edge_length", "grid_dim", "learning_rate", "sigma", "num_epochs"]
        )
        fig.update_layout(
            title="Optuna Study: Parallel Coordinate Plot",
            font=dict(size=18),
            height=1200,  # Set the height in pixels
            width=2000,  # Set the width in pixels
        )
        color_scale = px.colors.diverging.Tealrose  # px.colors.sequential.Viridis
        fig.data[0]["line"]["colorscale"] = color_scale
        pio.write_image(fig, f"{results_dir}{run_name}__{best_params_string}.png")

    num_significant_figures = 6
    format_str = f"%.{num_significant_figures}g"
    results_frame = pd.DataFrame(results)
    results_frame.to_csv(f"{results_dir}results.csv", index=None, float_format=format_str)  # type: ignore
