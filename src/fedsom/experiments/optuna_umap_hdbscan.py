from sklearn.metrics import normalized_mutual_info_score
import optuna 
import numpy as np   
import hdbscan
from deepclustering.encoder.tabular_perturbations import TabularPerturbation,DataTrimmer
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
from deepclustering.encoder.utils.data_utils import CategoricalOneHotEncoder   
from pathlib import Path    
from deepclustering.utils.graphics_utils import scatter_plot
import pandas as pd    
from deepclustering.encoder.utils.preprocessing_utils import (
    encode_categoricals,
    train_categorical_encodings,
    parse_columns,
    replace_nans,
    replace_nans_np_array,
    train_power_transformer,
)
from deepclustering.datasets.datasets import *
import optuna
from joblib import Parallel, delayed
import warnings
import os 
import argparse
warnings.filterwarnings("ignore")



def data_to_sparse(data):

    if isinstance(data,pd.DataFrame):
        data = data.values

    num_samples, num_features = data.shape   
    sparse_data = sp.lil_matrix((num_samples,num_features), dtype=np.float32)
    for row_idx in range(num_samples):

        nonzero_indices = data[row_idx]!=0
        sparse_data[row_idx,nonzero_indices] = data[row_idx,nonzero_indices]

    return sparse_data


def embed_via_umap_dict(vectors,umap_mapper_dict,data_trimmer,tabular_perturbation,categorical_onehot_encoder,power_transformer):

    trimmed = data_trimmer.trim_constant_columns(vectors)
    categorical_indices = tabular_perturbation.get_categorical_indices()
    categorical,numerical = parse_columns(trimmed,categorical_indices)

    if not categorical.empty:
        categorical = categorical_onehot_encoder.transform(categorical)
    if not categorical.empty:
        sparse_categorical = data_to_sparse(categorical)
        cat_embedding = umap_mapper_dict['umap_mapper_categorical'].transform(sparse_categorical)
        cat_embedding = replace_nans_np_array(cat_embedding)
    if not numerical.empty:
        numerical = replace_nans(numerical)
        if power_transformer:
            numerical = pd.DataFrame(power_transformer.transform(numerical),columns=numerical.columns,index=numerical.index)
        sparse_numerical = data_to_sparse(numerical)
        num_embedding = umap_mapper_dict['umap_mapper_numerical'].transform(sparse_numerical)
        num_embedding = replace_nans_np_array(num_embedding)
    if (not categorical.empty) and (not numerical.empty):
        embedding = umap_mapper_dict['umap_mapper_combo'].transform(np.concatenate((cat_embedding,num_embedding),axis=1))
    elif (not categorical.empty) and (numerical.empty):
        embedding = cat_embedding   
    elif (categorical.empty) and (not numerical.empty):
        embedding = num_embedding 

    return embedding 


def generate_params_list(encoder_params,umap_params,hdbscan_params):

    total_keys = []
    for key in umap_params:
        if key=='numerical':
            for subkey in umap_params[key].keys():
                total_keys.append(f'{subkey}__num')
        elif key=='categorical':
            for subkey in umap_params[key].keys():
                total_keys.append(f'{subkey}__cat')
        elif key=='combo':
            for subkey in umap_params[key].keys():
                total_keys.append(f'{subkey}__combo')

    total_keys += list(encoder_params.keys()) + list(hdbscan_params.keys())
    return total_keys


class Checkpoint(object):
    def __init__(self,data_dir,clear_old=True):

        self.data_dir = data_dir
        self.hdbscan_name = 'hdbscan.joblib'
        self.power_transformer_name = 'power_transformer.joblib'
        self.tabular_perturbation_name = 'tabular_perturbation.pt'
        self.categorical_onehot_encoder_name = 'categorical_onehot_encoder.joblib'
        self.umap_names = ['umap_mapper_categorical','umap_mapper_numerical','umap_mapper_combo']
        self.data_trimmer_name = 'data_trimmer.pt'  
        self.score_name = 'score.joblib'
        if clear_old:
            self.delete_file(data_dir / self.power_transformer_name)
            self.delete_file(data_dir / self.hdbscan_name)
            self.delete_file(data_dir / self.data_trimmer_name)
            self.delete_file(data_dir / self.score_name)
            self.delete_file(data_dir / self.categorical_onehot_encoder_name)
            for umap_name in self.umap_names:
                self.delete_file(data_dir / f'{umap_name}.joblib')

    def __call__(self,score,umap_mapper_dict,hdbscan_model,data_trimmer,tabular_perturbation,categorical_onehot_encoder,power_transformer):

        best_score = self.load_best_score()
        if score>best_score:
            self.save_umap(umap_mapper_dict)
            joblib.dump(hdbscan_model,self.data_dir / self.hdbscan_name)
            joblib.dump(power_transformer,self.data_dir / self.power_transformer_name)
            data_trimmer.save(self.data_dir / self.data_trimmer_name)
            tabular_perturbation.save(self.data_dir / self.tabular_perturbation_name)
            categorical_onehot_encoder.save(self.data_dir / self.categorical_onehot_encoder_name)
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


    def save_umap(self,umap_mapper_dict):
        for umap_name in umap_mapper_dict.keys():
            if umap_mapper_dict[umap_name]: 
                joblib.dump(umap_mapper_dict[umap_name],self.data_dir / f'{umap_name}.joblib')

    def seed(self):
        self.overwrite_best_score(0)

    def load_best_score(self):
        return joblib.load(self.data_dir / self.score_name) 

    def overwrite_best_score(self,score):
        joblib.dump(score, self.data_dir / self.score_name)

    def load_power_transformer(self):
        return joblib.load(self.data_dir / self.power_transformer_name)

    def load_data_trimmer(self):
        return DataTrimmer.load(self.data_dir / self.data_trimmer_name)

    def load_categorical_onehot_encoder(self):
        return CategoricalOneHotEncoder.load(self.data_dir / self.categorical_onehot_encoder_name)

    def load_perturbation(self):
        return TabularPerturbation.load(self.data_dir / self.tabular_perturbation_name)

    def load_hdbscan(self):
        return joblib.load(self.data_dir / self.hdbscan_name)

    def load_mappers(self):
        umap_mapper_dict = {}
        for umap_name in self.umap_names:
            filepath = self.data_dir / f'{umap_name}.joblib'
            if os.path.exists(filepath):
                umap_mapper_dict[umap_name] = joblib.load(filepath)
        return umap_mapper_dict  




def objective(trial, data_dict, encoder_params, umap_params, hdbscan_params, checkpoint):

    np.random.seed(0)
    approx_constant_thresh = trial.suggest_float("approx_constant_thresh",*encoder_params["approx_constant_thresh"],log=True)

    train_vectors = data_dict['train_vectors']
    data_trimmer = DataTrimmer(approx_constant_thresh=approx_constant_thresh)
    data_trimmer.learn_sparsities(train_vectors)
    train_trimmed = data_trimmer.trim_constant_columns(train_vectors)
    tabular_perturbation = TabularPerturbation(train_trimmed.shape[1])
    tabular_perturbation.find_categorical_indices(train_trimmed)
    categorical_indices = tabular_perturbation.get_categorical_indices()
    categorical,numerical = parse_columns(train_trimmed,categorical_indices)

    categorical_onehot_encoder = CategoricalOneHotEncoder()
    if not categorical.empty:
        categorical_onehot_encoder.record_high_entropy_cols(categorical)
        categorical = categorical_onehot_encoder.fit_transform(categorical)

    if not numerical.empty:
        umap_params_numerical = {
        'n_neighbors':trial.suggest_int("n_neighbors__num", *umap_params["numerical"]["n_neighbors"]),
        'n_components':trial.suggest_int("n_components__num",*umap_params["numerical"]["n_components"]),
        'min_dist':trial.suggest_float("min_dist__num",*umap_params["numerical"]["min_dist"],log=True),
        'init':trial.suggest_categorical("init__num",umap_params["numerical"]["init"]),
        'metric':trial.suggest_categorical("metric__num",umap_params["numerical"]["metric"]),
        }
    if not categorical.empty:
        umap_params_categorical = {
        'n_neighbors':trial.suggest_int("n_neighbors__cat", *umap_params['categorical']["n_neighbors"]),
        'n_components':trial.suggest_int("n_components__cat",*umap_params['categorical']["n_components"]),
        'min_dist':trial.suggest_float("min_dist__cat",*umap_params['categorical']["min_dist"],log=True),
        'init':trial.suggest_categorical("init__cat",umap_params['categorical']["init"]),
        'metric':trial.suggest_categorical("metric__cat",umap_params['categorical']["metric"]),
        }
    if (not numerical.empty) and (not categorical.empty):
        umap_params_combo = {
        'n_neighbors':trial.suggest_int("n_neighbors__combo", *umap_params['combo']["n_neighbors"]),
        'n_components':trial.suggest_int("n_components__combo",*umap_params['combo']["n_components"]),
        'min_dist':trial.suggest_float("min_dist__combo",*umap_params['combo']["min_dist"],log=True),
        'init':trial.suggest_categorical("init__combo",umap_params['combo']["init"]),
        'metric':trial.suggest_categorical("metric__combo",umap_params['combo']["metric"]),
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

    umap_mapper_categorical = None   
    umap_mapper_numerical = None   
    umap_mapper_combo = None 
    power_transformer = None 
    if not categorical.empty:

        sparse_categorical = data_to_sparse(categorical)
        umap_params_categorical['n_components'] = min(umap_params_categorical['n_components'],sparse_categorical.shape[1])
        umap_mapper_categorical = umap.UMAP(**umap_params_categorical,random_state=42, low_memory=True).fit(sparse_categorical)

    if not numerical.empty:

        numerical = replace_nans(numerical)

        power_transformer = None 
        if not categorical.empty:
            power_transformer = train_power_transformer(numerical)
            numerical = pd.DataFrame(power_transformer.transform(numerical),columns=numerical.columns,index=numerical.index)

        sparse_numerical = data_to_sparse(numerical)
        umap_params_numerical['n_components'] = min(umap_params_numerical['n_components'],sparse_numerical.shape[1])
        umap_mapper_numerical = umap.UMAP(**umap_params_numerical,random_state=42, low_memory=True).fit(sparse_numerical)

    if (not categorical.empty) and (not numerical.empty):
        joined = np.concatenate((umap_mapper_numerical.embedding_,umap_mapper_categorical.embedding_),axis=1)
        umap_params_combo['n_components'] = min(umap_params_combo['n_components'],umap_params_categorical['n_components']+umap_params_numerical['n_components'])
        umap_mapper_combo = umap.UMAP(**umap_params_combo,random_state=42,low_memory=True).fit(joined)

    if (not categorical.empty) and numerical.empty:
        embedding = umap_mapper_categorical.embedding_   
    elif categorical.empty and (not numerical.empty):
        embedding = umap_mapper_numerical.embedding_     
    else:
        embedding = umap_mapper_combo.embedding_

    umap_mapper_dict = {}
    umap_mapper_dict['umap_mapper_categorical'] = umap_mapper_categorical
    umap_mapper_dict['umap_mapper_numerical'] = umap_mapper_numerical
    umap_mapper_dict['umap_mapper_combo'] = umap_mapper_combo

    hdbscan_model = hdbscan.HDBSCAN(**hdbscan_params,core_dist_n_jobs=cpu_count(),prediction_data=True)
    hdbscan_model.fit(embedding)

    validation_vectors = data_dict['validation_vectors']   
    embedding = embed_via_umap_dict(validation_vectors,umap_mapper_dict,data_trimmer,tabular_perturbation,categorical_onehot_encoder,power_transformer)

    labels_learned,_ = hdbscan.approximate_predict(hdbscan_model,embedding)
    labels_true = data_dict['validation_labels']
    score = normalized_mutual_info_score(labels_true,labels_learned)
    checkpoint(score,umap_mapper_dict,hdbscan_model,data_trimmer,tabular_perturbation,categorical_onehot_encoder,power_transformer)

    return score


# Create a wrapper function to pass additional arguments
def wrapped_objective(arg1, arg2, arg3, arg4, arg5):
    return lambda trial: objective(trial, arg1, arg2, arg3, arg4, arg5)



def encode(data_dict,checkpoint):

    np.random.seed(0)
    data_trimmer = checkpoint.load_data_trimmer()
    hdbscan_model = checkpoint.load_hdbscan()
    umap_mapper_dict = checkpoint.load_mappers()
    tabular_perturbation = checkpoint.load_perturbation()
    categorical_onehot_encoder = checkpoint.load_categorical_onehot_encoder()
    power_transformer = checkpoint.load_power_transformer()
    test_vectors = data_dict['test_vectors']
    labels_true = data_dict['test_labels']

    embedding = embed_via_umap_dict(test_vectors,umap_mapper_dict,data_trimmer,tabular_perturbation,categorical_onehot_encoder,power_transformer)
    labels_learned,_ = hdbscan.approximate_predict(hdbscan_model,embedding)

    return embedding,labels_true,labels_learned



if __name__=='__main__':


    # python optuna_encoder_som.py --dataset mnist --n_trials 3

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str,default='malmem')
    parser.add_argument('--n_trials',type=int,default=3)
    parser.add_argument('--num_samples_train',type=int,default=1000)#_000_000)
    parser.add_argument('--num_samples_validate',type=int,default=1000)#_000_000)
    parser.add_argument('--num_samples_test',type=int,default=250)#0)

    args = parser.parse_args()
    dataset = args.dataset 
    n_trials = args.n_trials
    num_samples_train = args.num_samples_train
    num_samples_validate = args.num_samples_validate
    num_samples_test = args.num_samples_test

    np.random.seed(0)

    results_dir = Path(f'../../../sandbox/results/UMAP/{dataset}/')
    if not os.path.exists(results_dir):
        results_dir.mkdir(parents=True)  

    checkpoint_dir = Path(f'../../../sandbox/results/UMAP/checkpoint/{dataset}/')
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

# ============================================================================================================

    if dataset=='mnist':
        data_path = Path('../../../sandbox/data/mnist_data/mnist_train.csv')
        data_class = MNISTDatasetInMem

    elif dataset=='fashionmnist':
        data_path = Path('../../../sandbox/data/fashion_mnist/fashion_mnist_train.csv')
        data_class = FashionMNISTDatasetInMem

    elif dataset=='notmnist':
        data_path = Path('../../../sandbox/data/notmnist/not_mnist_train.csv')
        data_class = NotMNISTDatasetInMem

    elif dataset=='quickdraw':
        data_path = Path('../../../sandbox/data/quickdraw/quickdraw_train.csv')
        data_class = QuickdrawInMem

    elif dataset=='slmnist':
        data_path = Path('../../../sandbox/data/slmnist/sign_mnist_train.csv')
        data_class = SLMNISTInMem

    elif dataset=='kuz':
        data_path = Path('../../../sandbox/data/kuzushiji_data/kuzushiji_mnist_normalized_train.csv')
        data_class = KuzushijiInMem

    elif dataset=='emnist':
        data_path = Path('../../../sandbox/data/emnist_data/emnist_normalized_train.csv')
        data_class = EMNISTInMem

    elif dataset=='chars74k':
        data_path = Path('../../../sandbox/data/chars74k/chars74k_coarse_vectors_normalized_train.csv')
        data_class = Chars74kDatasetInMem

    elif dataset=='cifar10':
        data_path = Path('../../../sandbox/data/cifar10/cifar10_train.csv')
        data_class = CIFAR10DatasetInMem

    elif dataset=='ember':
        data_path = Path('../../../sandbox/data/ember/top_k/ember_top_10_train.csv')
        data_class = EmberDatasetInMem

    elif dataset=='sorel':
        data_path = Path('../../../sandbox/data/sorel/sorel_subset_train.csv')
        data_class = SorelDatasetInMem

    elif dataset=='ccc':
        data_path = Path('../../../sandbox/data/MalwareDatasets/MalwareDatasets/CCCS-CIC-AndMal2020/total/cccs_cic_andmal2020_train.csv')
        data_class = CCCSInMem

    elif dataset=='cicandmal2017':
        data_path = Path('../../../sandbox/data/MalwareDatasets/MalwareDatasets/CICAndMal2017/total/cic_andmal2017_train.csv')
        data_class = CICAndMal2017InMem

    elif dataset=='syscalls':  
        data_path = Path('../../../sandbox/data/MalwareDatasets/MalwareDatasets/CICMalDroid2020/total/feature_vectors_syscalls_frequency_5_Cat_train.csv')
        data_class = SysCallsInMem

    elif dataset=='syscallsbinders':
        data_path = Path('../../../sandbox/data/MalwareDatasets/MalwareDatasets/CICMalDroid2020/total/feature_vectors_syscallsbinders_frequency_5_Cat_train.csv')
        data_class = SysCallsBindersInMem

    elif dataset=='malmem':
        data_path = Path('../../../sandbox/data/MalwareDatasets/MalwareDatasets/MalMem/total/Obfuscated-MalMem2022_train.csv')
        data_class = MalMemInMem

    elif dataset=='pdfmalware': 
        data_path = Path('../../../sandbox/data/MalwareDatasets/MalwareDatasets/PDFMalware/total/pdfmalware_train.csv')
        data_class = PDFMalwareInMem


    data = data_class.load(data_path)
    if dataset=='malmem':
        data_class_object = data_class(class_type='subavclass')
        data_dict = data_class_object.split_data(data,num_samples_train,num_samples_validate,num_samples_test)
    else:
        data_dict = data_class.split_data(data,num_samples_train,num_samples_validate,num_samples_test)
    

    # input('2')

    run_name = dataset
    encoder_params = {
        "approx_constant_thresh" : [0.5,0.999],
    }

    umap_params_numerical = {
        "n_neighbors": [2, 100],
        "n_components": [2, 200],
        "min_dist": [0.000001, 0.1],
        "init": ["spectral","random"],
        "metric": ["cosine","euclidean"],
    }

    umap_params_categorical = {
        "n_neighbors": [2, 100],
        "n_components": [2, 200],
        "min_dist": [0.000001, 0.1],
        "init": ["spectral","random"],
        "metric": ["dice"],
    }

    umap_params_combo = {
        "n_neighbors": [2, 100],
        "n_components": [2, 200],
        "min_dist": [0.000001, 0.1],
        "init": ["spectral","random"],
        "metric": ["euclidean"],
    }

    umap_params = {}
    umap_params['numerical'] = umap_params_numerical
    umap_params['categorical'] = umap_params_categorical 
    umap_params['combo'] = umap_params_combo

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


    # input('3')

    sampler = TPESampler(seed=10)  # Make the sampler behave in a deterministic way.
    study = optuna.create_study(sampler=sampler,direction='maximize')
    study.optimize(wrapped_objective(data_dict, encoder_params, umap_params, hdbscan_params, checkpoint), n_trials=n_trials)

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
    embeddings,labels_true, labels_learned = encode(data_dict,checkpoint)
    best_params_string = dict_2_string(best_params)
    scatter_path = results_dir / f"{run_name}__scatter_true_labels.png"
    scatter_plot(embeddings, labels_true, dataset, scatter_path)
    scatter_path = results_dir / f"{run_name}__scatter_learned_labels.png"
    scatter_plot(embeddings, labels_learned, dataset, scatter_path)

    fig = optuna.visualization.plot_parallel_coordinate(study, params=list(best_params.keys())
    )
    idx = len(best_params_string)//2
    fig.update_layout(
        title=dataset,#best_params_string[:idx] + '\n' + best_params_string[idx:],
        font=dict(size=10),
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
        title=dataset,#best_params_string[:idx] + '\n' + best_params_string[idx:],
        font=dict(size=14),
        height=1200,  # Set the height in pixels
        width=2000,  # Set the width in pixels
    )
    pio.write_image(fig, results_dir / f"{run_name}__bar.png")
    del fig



















