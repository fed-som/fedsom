import copy
import warnings
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import plotly.express as px
import plotly.io as pio
import sklearn
import torch
from optuna.visualization import plot_optimization_history
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

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


def create_datasets(n_samples_values, centers_values, custer_std_values, evaluations):
    parameter_combinations = product(n_samples_values, centers_values, cluster_std_values, evaluations)

    datasets = []
    for params in parameter_combinations:
        n_samples, centers, cluster_std, evaluation = params

        X, labels = make_blobs(n_samples=n_samples, centers=centers, random_state=42, cluster_std=cluster_std)

        dataset = {
            "parameters": {
                "n_samples": n_samples,
                "centers": centers,
                "cluster_std": cluster_std,
            },
            "X": X,
            "labels": labels,
            "evaluation": evaluation,
        }

        yield dataset


def select_evaluation(evaluation):
    if evaluation == "silhouette":
        return {"method": sklearn.metrics.silhouette_score, "direction": "maximize"}
    elif evaluation == "calinski_harabazs":
        return {"method": sklearn.metrics.calinski_harabasz_score, "direction": "maximize"}
    elif evaluation == "davies_bouldin":
        return {"method": sklearn.metrics.davies_bouldin_score, "direction": "minimize"}


def objective(trial, X, labels_true, som_parameters, evaluation):
    grid_edge_length = trial.suggest_int("grid_edge_length", *som_parameters["grid_edge_length"])
    grid_dim = trial.suggest_int("grid_dim", *som_parameters["grid_dim"])
    learning_rate = trial.suggest_loguniform("learning_rate", *som_parameters["learning_rate"])
    sigma = trial.suggest_loguniform("sigma", *som_parameters["sigma"])
    num_epochs = trial.suggest_int("num_epochs", *som_parameters["num_epochs"], log=False)
    grid_size = tuple([grid_edge_length for _ in range(grid_dim)])

    som = SelfOrganizingMap(grid_size, input_size=2, learning_rate=learning_rate, sigma=sigma)
    som.train(torch.tensor(X), num_epochs=num_epochs)

    bmu_labels = {}
    for n, x in enumerate(X):
        units = som.find_best_matching_unit(torch.tensor(x))
        bmu_labels[n] = bmu_to_string(units)

    labels = convert_bmu_labels(bmu_labels)





    
    direction = evaluation["direction"]
    if len(set(labels)) == 1:
        if direction == "maximize":
            return -np.inf
        if direction == "minimize":
            return np.inf
    else:
        return evaluation["method"](X, labels)


# Create a wrapper function to pass additional arguments
def wrapped_objective(arg1, arg2, arg3, arg4):
    return lambda trial: objective(trial, arg1, arg2, arg3, arg4)


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
    plt.title(best_params_string, fontsize=12, fontweight="bold")
    plt.colorbar(label="Cluster")
    plt.show(block=False)
    plt.savefig(filepath)
    plt.close()
    del fig


if __name__ == "__main__":
    n_samples_values = [1000]
    centers_values = [100]
    cluster_std_values = [0.1]
    n_trials = 10
    evaluations = ["calinski_harabazs", "silhouette", "davies_bouldin"]
    results = []
    datasets = create_datasets(n_samples_values, centers_values, cluster_std_values, evaluations)
    results_dir = "../../../../sandbox/images/optuna/temp/"

    som_parameters = {
        "grid_edge_length": [3, 4],
        "grid_dim": [2, 3],
        "learning_rate": [1.0e-5, 10.0],
        "sigma": [1.0e-5, 10.0],
        "num_epochs": [1, 30],
    }

    for dataset in datasets:
        parameters = dataset["parameters"]
        X = dataset["X"]
        labels_true = dataset["labels"]
        evaluation = select_evaluation(dataset["evaluation"])

        experiment_parameters = copy.deepcopy(parameters)
        experiment_parameters.update({"evaluation": dataset["evaluation"]})
        run_name = dict_2_string(experiment_parameters)

        # Create an Optuna study
        study = optuna.create_study(direction=evaluation["direction"])

        # Optimize the objective function
        study.optimize(wrapped_objective(X, labels_true, som_parameters, evaluation), n_trials=n_trials)

        # Print the best parameters and objective value
        best_params = study.best_params
        best_value = study.best_value
        print(f"Best parameters: {best_params}")
        print(f"Best value: {best_value}")

        # Plot labeled data
        best_labels = som_cluster(X, best_params)
        best_params_string = dict_2_string(best_params) + f'_{dataset["evaluation"]}'
        filepath = f"{results_dir}{run_name}___{best_params_string}_scatter.png"
        scatter_plot(X, best_labels, best_params_string, filepath)

        # Create a color scale using the 'Viridis' color scale
        fig = optuna.visualization.plot_parallel_coordinate(
            study, params=["grid_edge_length", "grid_dim", "learning_rate", "sigma", "num_epochs"]
        )
        fig.update_layout(
            title=best_params_string,
            font=dict(size=18),
            height=1200,  # Set the height in pixels
            width=2000,  # Set the width in pixels
        )
        color_scale = px.colors.diverging.Tealrose  # px.colors.sequential.Viridis
        fig.data[0]["line"]["colorscale"] = color_scale
        pio.write_image(fig, f"{results_dir}{run_name}__{best_params_string}.png")
        del fig

        fig = optuna.visualization.plot_param_importances(study)
        fig.update_layout(
            title=best_params_string,
            font=dict(size=18),
            height=1200,  # Set the height in pixels
            width=2000,  # Set the width in pixels
        )
        pio.write_image(fig, f"{results_dir}best_params_{run_name}__{best_params_string}.png")
        del fig

        # record run
        result = {"run_name": run_name}
        result.update(parameters)
        result.update(best_params)
        result["best_value"] = best_value
        results.append(result)

    num_significant_figures = 6
    format_str = f"%.{num_significant_figures}g"
    results_frame = pd.DataFrame(results)
    results_frame.to_csv(f"{results_dir}results.csv", index=None, float_format=format_str)  # type: ignore
