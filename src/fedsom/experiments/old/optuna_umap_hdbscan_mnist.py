from sklearn.metrics import normalized_mutual_info_score
import optuna 
import numpy as np   
import hdbscan
from deepclustering.encoder.tabular_perturbations import DataTrimmer
from multiprocessing import cpu_count
import torch  
from optuna.samplers import TPESampler
import torch.optim as optim
import joblib 
import plotly.express as px
import plotly.io as pio
import umap 
import scipy.sparse as sp 
from deepclustering.utils.general_utils import dict_2_string
from pathlib import Path    
from deepclustering.utils.graphics_utils import scatter_plot
import pandas as pd    
from deepclustering.encoder.utils.preprocessing_utils import (
    encode_categoricals,
    train_categorical_encodings,
)
import optuna
from joblib import Parallel, delayed
import warnings
import os 
import argparse
warnings.filterwarnings("ignore")



def mnist_to_sparse(data):

    data = (1./255)*data[[c for c in data.columns if c!='class']].values.astype('float32')
    num_samples, num_features = data.shape   
    sparse_mnist_data = sp.lil_matrix((num_samples,num_features), dtype=np.float32)
    for row_idx in range(num_samples):

        nonzero_indices = data[row_idx]!=0
        sparse_mnist_data[row_idx,nonzero_indices] = data[row_idx,nonzero_indices]

    return sparse_mnist_data




class Checkpoint(object):
    def __init__(self,data_dir,clear_old=True):

        self.data_dir = data_dir
        self.hdbscan_name = 'hdbscan.joblib'
        self.umap_name = 'mapper.joblib'
        self.data_trimmer_name = 'data_trimmer.pt'  
        self.score_name = 'score.joblib'
        if clear_old:
            self.delete_file(data_dir / self.hdbscan_name)
            self.delete_file(data_dir / self.umap_name)
            self.delete_file(data_dir / self.data_trimmer_name)
            self.delete_file(data_dir / self.score_name)

    def __call__(self,score,umap_mapper,hdbscan_model,data_trimmer):

        best_score = self.load_best_score()
        if score>best_score:
            joblib.dump(umap_mapper,self.data_dir / self.umap_name)
            joblib.dump(hdbscan_model,self.data_dir / self.hdbscan_name)
            data_trimmer.save(self.data_dir / self.data_trimmer_name)
            self.overwrite_best_score(score)

    @staticmethod
    def delete_file(filepath):

        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError as e:
                print(f"Error: {e} - {filepath}")
        else:
            print(f"The file {filepath} does not exist.")

    def seed(self):
        self.overwrite_best_score(0)

    def load_best_score(self):
        return joblib.load(self.data_dir / self.score_name) 

    def overwrite_best_score(self,score):
        joblib.dump(score, self.data_dir / self.score_name)

    def load_data_trimmer(self):
        return DataTrimmer.load(self.data_dir / self.data_trimmer_name)

    def load_hdbscan(self):
        return joblib.load(self.data_dir / self.hdbscan_name)

    def load_mapper(self):
        return joblib.load(self.data_dir / self.umap_name)




def objective(trial, data, dataset_params, encoder_params, umap_params, hdbscan_params, checkpoint):



    np.random.seed(0)

    approx_constant_thresh = trial.suggest_float("approx_constant_thresh",*encoder_params["approx_constant_thresh"],log=True)

    umap_params = {
    'n_neighbors':trial.suggest_int("n_neighbors", *umap_params["n_neighbors"]),
    'n_components':trial.suggest_int("n_components",*umap_params["n_components"]),
    'min_dist':trial.suggest_float("min_dist",*umap_params["min_dist"],log=True),
    'init':trial.suggest_categorical("init",umap_params["init"]),
    'metric':trial.suggest_categorical("metric",umap_params["metric"]),
    }
    alt_algo = trial.suggest_categorical("alt_algo",hdbscan_params["alt_algo"])
    hdbscan_params = {
    'min_cluster_size':trial.suggest_int("min_cluster_size",*hdbscan_params["min_cluster_size"]),
    'min_samples':trial.suggest_int("min_samples",*hdbscan_params["min_samples"]),
    'cluster_selection_method':trial.suggest_categorical("cluster_selection_method",hdbscan_params["cluster_selection_method"]),
    'cluster_selection_epsilon':trial.suggest_float("cluster_selection_epsilon",*hdbscan_params["cluster_selection_epsilon"]),
    'algorithm':trial.suggest_categorical("algorithm",hdbscan_params["algorithm"]),
    'leaf_size':trial.suggest_int("leaf_size",*hdbscan_params["leaf_size"]),
    'metric':trial.suggest_categorical("hdbscan_metric",hdbscan_params["hdbscan_metric"]),
    'p':trial.suggest_float("p",*hdbscan_params["p"],log=True)
    } 
    if hdbscan_params['metric'] in ["canberra","braycurtis"]:
        hdbscan_params['algorithm'] = alt_algo


    num_samples_train = dataset_params['num_samples_train']
    num_samples_test = dataset_params['num_samples_test']

    num_samples_train = min(num_samples_train,54000)
    num_samples_test = min(num_samples_test,13500)

    data_train = data.iloc[:num_samples_train,:] 
    data_test = data.iloc[num_samples_train:num_samples_train+num_samples_test,:]

    data_test_vectors = data_test[[c for c in data.columns if c!='class']]
    train_vectors = data_train[[c for c in data.columns if c!='class']]

    data_trimmer = DataTrimmer(approx_constant_thresh=approx_constant_thresh)
    data_trimmer.learn_sparsities(train_vectors)
    data_trimmed = data_trimmer.trim_constant_columns(train_vectors)
    sparse_mnist_data = mnist_to_sparse(data_trimmed)

    umap_mapper = umap.UMAP(**umap_params,random_state=42, low_memory=True).fit(sparse_mnist_data)
    hdbscan_model = hdbscan.HDBSCAN(**hdbscan_params,core_dist_n_jobs=cpu_count(),prediction_data=True)
    hdbscan_model.fit(umap_mapper.embedding_)

    data_test_trimmed = data_trimmer.trim_constant_columns(data_test_vectors)
    embedding = umap_mapper.transform(data_test_trimmed)
    labels_learned,_ = hdbscan.approximate_predict(hdbscan_model,embedding)
    labels_true = data_test['class']

    score = normalized_mutual_info_score(labels_true,labels_learned)
    checkpoint(score,umap_mapper,hdbscan_model,data_trimmer)

    return score


# Create a wrapper function to pass additional arguments
def wrapped_objective(arg1, arg2, arg3, arg4, arg5, arg6):
    return lambda trial: objective(trial, arg1, arg2, arg3, arg4, arg5, arg6)



def encode(test_data,dataset_params,checkpoint):

    np.random.seed(0)
    data_trimmer = checkpoint.load_data_trimmer()
    hdbscan_model = checkpoint.load_hdbscan()
    mapper = checkpoint.load_mapper()
    test_data_vectors = test_data[[c for c in test_data.columns if c!='class']]
    labels_true = test_data['class'].values
    data_trimmed = data_trimmer.trim_constant_columns(test_data_vectors)
    embedding = mapper.transform(data_trimmed)
    labels_learned,_ = hdbscan.approximate_predict(hdbscan_model,embedding)

    return embedding,labels_true,labels_learned



if __name__=='__main__':

    # python optuna_encoder_som.py --dataset mnist --n_trials 3

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str,default='mnist')
    parser.add_argument('--n_trials',type=int,default=200)
    parser.add_argument('--num_samples_train',type=int,default=100000)
    parser.add_argument('--num_samples_test',type=int,default=100000)
    parser.add_argument('--num_samples_cluster',type=int,default=2500)

    args = parser.parse_args()
    dataset = args.dataset 
    n_trials = args.n_trials
    num_samples_train = args.num_samples_train
    num_samples_test = args.num_samples_test
    num_samples_cluster = args.num_samples_cluster

    results_dir = Path(f'../../../sandbox/results/UMAP/{dataset}/')
    if not os.path.exists(results_dir):
        results_dir.mkdir(parents=True)  

    checkpoint_dir = Path(f'../../../sandbox/tmp_experiment_data/{dataset}/')
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    if dataset=='mnist':
        data_path = Path('../../../sandbox/data/mnist/mnist.csv')
        data = pd.read_csv(data_path)

    run_name = dataset
    dataset_params = {}
    dataset_params['num_samples_train'] = num_samples_train
    dataset_params['num_samples_test'] = num_samples_test
    dataset_params['num_samples_cluster'] = num_samples_cluster


    encoder_params = {
                    "approx_constant_thresh" : [0.5,0.999],

    }

    umap_params = {
        "n_neighbors": [2, 100],
        "n_components": [2, 200],
        "min_dist": [0.000001, 0.1],
        "init": ["spectral","random","tswspectral","pca"],
        "metric": ["cosine"],
    }

    hdbscan_params = {
    "min_cluster_size": [2,100],
    "min_samples": [2,100],
    "cluster_selection_method": ["eom"],
    "cluster_selection_epsilon": [0,1],
    "algorithm": ["best","prims_kdtree","prims_balltree","boruvka_kdtree"],
    "alt_algo":["best","prims_balltree"],
    "leaf_size": [5,100],
    "p":[0.001,2],
    "hdbscan_metric": ["canberra","braycurtis","chebyshev","cityblock",
                "euclidean","minkowski"],
        }


    checkpoint = Checkpoint(checkpoint_dir)
    checkpoint.seed()

    num_samples_cluster = min(num_samples_cluster,2500)
    data_cluster = data.iloc[(data.shape[0]-num_samples_cluster):data.shape[0],:]

    sampler = TPESampler(seed=10)  # Make the sampler behave in a deterministic way.
    study = optuna.create_study(sampler=sampler,direction='maximize')
    study.optimize(wrapped_objective(data, dataset_params, encoder_params, umap_params, hdbscan_params, checkpoint), n_trials=n_trials)

    # Print the best parameters and objective value
    best_params = study.best_params
    best_value = study.best_value
    print(f"Best parameters: {best_params}")
    print(f"Best value: {best_value}")
    best_params_df = {}
    best_params_df['best_value'] = study.best_value
    best_params_df = pd.DataFrame.from_dict(best_params,orient='index')
    best_params_df.to_csv(results_dir / 'best_params.csv')


    best_params = study.best_params
    embeddings,labels_true, labels_learned = encode(data_cluster,dataset_params,checkpoint)
    best_params_string = dict_2_string(best_params)
    scatter_path = results_dir / f"{run_name}__scatter_true_labels.png"
    scatter_plot(embeddings, labels_true, best_params_string, scatter_path)
    scatter_path = results_dir / f"{run_name}__scatter_learned_labels.png"
    scatter_plot(embeddings, labels_learned, best_params_string, scatter_path)

    fig = optuna.visualization.plot_parallel_coordinate(study, params=[
                    "n_neighbors",
                    "n_components",
                    "min_dist",
                    "init",
                    "metric",
                    "min_cluster_size",
                    "min_samples",
                    "cluster_selection_method",
                    "cluster_selection_epsilon",
                    "algorithm",
                    "leaf_size",
                    "p",
                    "hdbscan_metric",
                    "approx_constant_thresh",
                    ]
    )
    idx = len(best_params_string)//2
    fig.update_layout(
        title=best_params_string[:idx] + '\n' + best_params_string[idx:],
        font=dict(size=14),
        height=1200,  # Set the height in pixels
        width=2000,  # Set the width in pixels
    )
    color_scale = px.colors.diverging.Tealrose  # px.colors.sequential.Viridis
    fig.data[0]["line"]["colorscale"] = color_scale
    pio.write_image(fig, results_dir / f"{run_name}__parallel.png")
    del fig

    idx = len(best_params_string)//2
    fig = optuna.visualization.plot_param_importances(study)
    fig.update_layout(
        title=best_params_string[:idx] + '\n' + best_params_string[idx:],
        font=dict(size=14),
        height=1200,  # Set the height in pixels
        width=2000,  # Set the width in pixels
    )
    pio.write_image(fig, results_dir / f"{run_name}__bar.png")
    del fig



















