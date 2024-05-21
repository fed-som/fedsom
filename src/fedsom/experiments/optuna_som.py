from sklearn.metrics import normalized_mutual_info_score
import optuna 
import numpy as np   
import torch  
from optuna.samplers import TPESampler
import joblib 
from torch.utils.data import Dataset, DataLoader
import plotly.express as px
import plotly.io as pio
from deepclustering.datasets.datasets import * 
from pathlib import Path    
from deepclustering.utils.graphics_utils import scatter_plot
from deepclustering.som.som_temp import SelfOrganizingMap
from deepclustering.som.federated_som import FederatedSelfOrganizingMap
from deepclustering.utils.general_utils import dict_2_string
from deepclustering.utils.device_utils import assign_device
import pandas as pd    
import optuna
import warnings
import os 
import argparse
warnings.filterwarnings("ignore")



class Checkpoint(object):
    def __init__(self,data_dir,som_name,clear_old=True):

        self.data_dir = data_dir
        self.score_name = 'score.joblib'
        self.som_name = som_name 
        if clear_old:
            self.delete_file(data_dir / self.score_name)
            self.delete_file(data_dir / self.som_name)

    def save(self,score,som):

        best_score = self.load_best_score()
        if score>best_score:
            som.save(self.data_dir / self.som_name)
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

    def load_som(self):
        if self.som_name=='som':
            return SelfOrganizingMap.load(self.data_dir / self.som_name)
        elif self.som_name=='fed_som':
            return FederatedSelfOrganizingMap.load(self.data_dir / self.som_name)


                         
def objective(trial,data,som_params,fedsom_params,optimization_params,dataset_params,checkpoint):

    np.random.seed(0)
    torch.manual_seed(0)

    num_epochs = trial.suggest_int("num_epochs",*optimization_params["num_epochs"])
    batch_size = trial.suggest_int("batch_size",*optimization_params["batch_size"])
    parallel = optimization_params["parallel"]
    batch_size = 2**batch_size

    input_size = data.shape[1]-1
    if dataset_params['som_type']=='som':
        grid_edge_length = trial.suggest_int("grid_edge_length",*som_params["grid_edge_length"])
        grid_dim = trial.suggest_int("grid_dim",*som_params["grid_dim"])
        learning_rate = trial.suggest_float("learning_rate",*som_params["learning_rate"],log=True)
        sigma = trial.suggest_float("sigma",*som_params["sigma"],log=True)
        num_epochs_som = trial.suggest_int("num_epochs_som",*som_params["num_epochs_som"])
        grid_size = tuple([grid_edge_length for _ in range(grid_dim)])
        som = SelfOrganizingMap(grid_size, 
                                input_size=input_size,
                                learning_rate=learning_rate,
                                sigma=sigma
                                )

    elif dataset_params['som_type']=='fed_som':
        grid_edge_length = trial.suggest_int("grid_edge_length",*fedsom_params["grid_edge_length"])
        grid_dim = trial.suggest_int("grid_dim",*fedsom_params["grid_dim"])
        learning_rate = trial.suggest_float("learning_rate",*fedsom_params["learning_rate"],log=True)
        sigma = trial.suggest_float("sigma",*fedsom_params["sigma"],log=True)
        num_epochs_som = trial.suggest_int("num_epochs_som",*fedsom_params["num_epochs_som"])
        grid_size = tuple([grid_edge_length for _ in range(grid_dim)])

        meta_grid_edge_length = trial.suggest_int("meta_grid_edge_length",*fedsom_params["meta_grid_edge_length"])
        meta_grid_dim = trial.suggest_int("meta_grid_dim",*fedsom_params["meta_grid_dim"])
        meta_learning_rate = trial.suggest_float("meta_learning_rate",*fedsom_params["meta_learning_rate"],log=True)
        meta_sigma = trial.suggest_float("meta_sigma",*fedsom_params["meta_sigma"],log=True)
        num_epochs_meta = trial.suggest_int("num_epochs_meta",*fedsom_params["num_epochs_meta"])
        num_soms = trial.suggest_int("num_soms",*fedsom_params["num_soms"])

        grid_size = tuple([grid_edge_length for _ in range(grid_dim)])
        meta_grid_size = tuple([meta_grid_edge_length for _ in range(meta_grid_dim)])

        som = FederatedSelfOrganizingMap(grid_size,
                                        num_soms=num_soms,
                                        input_size=input_size,
                                        learning_rate=learning_rate,
                                        meta_learning_rate=meta_learning_rate,
                                        sigma=sigma,
                                        meta_sigma=meta_sigma,
                                        meta_grid_dim=meta_grid_dim,
                                        parallel=parallel
                                        )    

    data_class = dataset_params['data_class']
    num_samples_train = dataset_params['num_samples_train']
    dataset = data_class(data,num_samples_train)
    if dataset_params['som_type']=='som':
        for epoch in range(num_epochs):
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            for n,(batch,labels,index) in enumerate(dataloader):
                    som.batch_train(batch,num_epochs_som)

    dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
    embeddings = []
    labels_true = []
    for n,(batch,labels,index) in enumerate(dataloader):
        embeddings.append(batch)
        labels_true.append(labels)
    embeddings = torch.tensor(np.vstack(embeddings))

    if isinstance(labels_true[0][0],str):
        labels_true = [x for L in labels_true for x in L]
        labels_dict = dict(zip(sorted(list(set(labels_true))),range(len(set(labels_true)))))
        labels_true = np.array([labels_dict[key] for key in labels_true])
    else:
        labels_true = torch.cat(labels_true)

    # labels_true = torch.concat(labels_true)

    if dataset_params['som_type']=='fed_som':
        for epoch in range(num_epochs):
            som.train(embeddings,num_epochs_som,num_epochs_meta)

    labels_learned = som.get_labels(embeddings)
    score = normalized_mutual_info_score(labels_true,labels_learned)
    checkpoint.save(score,som)

    return score


# Create a wrapper function to pass additional arguments
def wrapped_objective(arg1, arg2, arg3, arg4, arg5, arg6):
    return lambda trial: objective(trial, arg1, arg2, arg3, arg4, arg5, arg6)


def score_som(data,dataset_params,checkpoint):

    np.random.seed(0)
    torch.manual_seed(0)

    data_class = dataset_params['data_class']
    dataset = data_class(data,dataset_params['num_samples_test'])
    som = checkpoint.load_som()

    dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
    total= []
    labels_list = []
    for n,(batch,labels,index) in enumerate(dataloader):
        total.append(batch)
        labels_list.append(labels)
    embeddings = torch.cat(total,dim=0)
    labels_learned = som.get_labels(embeddings)

    if isinstance(labels_list[0][0],str):
        labels_true = [x for L in labels_list for x in L]
    else:
        labels_true = torch.cat(labels_list)

    return embeddings,labels_true,labels_learned




if __name__=='__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str,default='mnist')
    parser.add_argument('--som_type',type=str,default='som')
    parser.add_argument('--embedding_origin',type=str,default='ENCODER')
    parser.add_argument('--n_trials',type=int,default=10)
    parser.add_argument('--num_samples_train',type=int,default=1000)
    parser.add_argument('--num_samples_test',type=int,default=1000)

    args = parser.parse_args()
    dataset = args.dataset 
    n_trials = args.n_trials
    num_samples_train = args.num_samples_train
    num_samples_test = args.num_samples_test
    som_type = args.som_type
    embedding_origin = args.embedding_origin

    results_dir = Path(f'../../../sandbox/results/OPTUNA_SOM_ON_ENCODER_EMBEDDINGS/{dataset}/{embedding_origin}/{som_type}/')
    if not os.path.exists(results_dir):
        results_dir.mkdir(parents=True)  

    checkpoint_dir = Path(f'../../../sandbox/results/OPTUNA_SOM_ON_ENCODER_EMBEDDINGS/{dataset}/{embedding_origin}/{som_type}/checkpoint/')
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    data_class = EmbeddingsDataset

    if dataset=='mnist':
        data_path = Path('../../../sandbox/data/mnist_data/embeddings/mnist_train_embeddings.csv')
        data_path = Path('../../../sandbox/data/mnist_data/embeddings/mnist_train_encoder_embeddings.csv')

    elif dataset=='fashionmnist':
        data_path = Path('../../../sandbox/data/fashion_mnist/embeddings/fashion_mnist_train_embeddings.csv')
        data_path = Path('../../../sandbox/data/fashion_mnist/embeddings/fashionmnist_train_encoder_embeddings.csv')

    elif dataset=='notmnist':
        data_path = Path('../../../sandbox/data/notmnist/embeddings/not_mnist_train_embeddings.csv')
        data_path = Path('../../../sandbox/data/notmnist/embeddings/notmnist_train_encoder_embeddings.csv')

    elif dataset=='quickdraw':
        data_path = Path('../../../sandbox/data/quickdraw/embeddings/quickdraw_train_embeddings.csv')
        data_path = Path('../../../sandbox/data/quickdraw/embeddings/quickdraw_train_encoder_embeddings.csv')

    elif dataset=='slmnist':
        data_path = Path('../../../sandbox/data/slmnist/embeddings/sign_mnist_train_embeddings.csv')
        data_path = Path('../../../sandbox/data/slmnist/embeddings/slmnist_train_encoder_embeddings.csv')

    elif dataset=='kuz':
        data_path = Path('../../../sandbox/data/kuzushiji_data/embeddings/kuzushiji_mnist_normalized_train_embeddings.csv')
        data_path = Path('../../../sandbox/data/kuzushiji_data/embeddings/kuz_train_encoder_embeddings.csv')

    elif dataset=='emnist':
        data_path = Path('../../../sandbox/data/emnist_data/embeddings/emnist_normalized_train_embeddings.csv')
        data_path = Path('../../../sandbox/data/emnist_data/embeddings/emnist_train_encoder_embeddings.csv')

    elif dataset=='chars74k':
        data_path = Path('../../../sandbox/data/chars74k/embeddings/chars74k_coarse_vectors_normalized_train_embeddings.csv')
        data_path = Path('../../../sandbox/data/chars74k/embeddings/chars74k_train_encoder_embeddings.csv')

    elif dataset=='cifar10':
        data_path = Path('../../../sandbox/data/cifar10/embeddings/cifar10_train_embeddings.csv')
        data_path = Path('../../../sandbox/data/cifar10/embeddings/cifar_train_encoder_embeddings.csv')

    elif dataset=='ember':
        data_path = Path('../../../sandbox/data/ember/top_k/embeddings/ember_top_10_train_embeddings.csv')
        data_path = Path('../../../sandbox/data/ember/top_k/embeddings/ember_train_encoder_embeddings.csv')

    elif dataset=='sorel':
        data_path = Path('../../../sandbox/data/sorel/embeddings/sorel_subset_train_embeddings.csv')
        data_path = Path('../../../sandbox/data/sorel/embeddings/sorel_train_encoder_embeddings.csv')

    elif dataset=='ccc':
        data_path = Path('../../../sandbox/data/MalwareDatasets/MalwareDatasets/CCCS-CIC-AndMal2020/total/embeddings/cccs_cic_andmal2020_train_embeddings.csv')
        data_path = Path('../../../sandbox/data/MalwareDatasets/MalwareDatasets/CCCS-CIC-AndMal2020/total/embeddings/ccc_train_encoder_embeddings.csv')

    elif dataset=='cicandmal2017':
        data_path = Path('../../../sandbox/data/MalwareDatasets/MalwareDatasets/CICAndMal2017/total/embeddings/cic_andmal2017_train_embeddings.csv')

    elif dataset=='syscalls':
        data_path = Path('../../../sandbox/data/MalwareDatasets/MalwareDatasets/CICMalDroid2020/total/embeddings/feature_vectors_syscalls_frequency_5_Cat_train_embeddings.csv')
        data_path = Path('../../../sandbox/data/MalwareDatasets/MalwareDatasets/CICMalDroid2020/total/embeddings/syscalls_train_encoder_embeddings.csv')

    elif dataset=='syscallsbinders':
        data_path = Path('../../../sandbox/data/MalwareDatasets/MalwareDatasets/CICMalDroid2020/total/embeddings/feature_vectors_syscallsbinders_frequency_5_Cat_train_embeddings.csv')
        data_path = Path('../../../sandbox/data/MalwareDatasets/MalwareDatasets/CICMalDroid2020/total/embeddings/syscallsbinders_train_encoder_embeddings.csv')

    elif dataset=='malmem':
        data_path = Path('../../../sandbox/data/MalwareDatasets/MalwareDatasets/MalMem/total/embeddings/Obfuscated-MalMem2022_train_embeddings.csv')
        data_path = Path('../../../sandbox/data/MalwareDatasets/MalwareDatasets/MalMem/total/embeddings/malmem_train_encoder_embeddings.csv')

    elif dataset=='pdfmalware':
        data_path = Path('../../../sandbox/data/MalwareDatasets/MalwareDatasets/PDFMalware/total/embeddings/pdfmalware_train_embeddings.csv')
        data_path = Path('../../../sandbox/data/MalwareDatasets/MalwareDatasets/PDFMalware/total/embeddings/pdfmalware_train_encoder_embeddings.csv')


    run_name = dataset
    dataset_params = {}
    dataset_params['data_class'] = data_class
    dataset_params['som_type'] = som_type
    dataset_params['num_samples_train'] = num_samples_train
    dataset_params['num_samples_test'] = num_samples_test

    optimization_params = {
        "num_epochs": [5,5],
        "batch_size": [2,8],
        "parallel": False
    }

    som_params = {
        "grid_edge_length": [2, 8],
        "grid_dim": [2, 3],
        "learning_rate": [1.0e-5, 10.0],
        "sigma": [0.1, 10.0],
        "num_epochs_som": [1, 5],
    }

    fedsom_params = {
        "grid_edge_length": [2, 10],
        "grid_dim": [2, 2],
        "learning_rate": [1.0e-5, 10.0],
        "sigma": [0.1, 10.0],
        "num_epochs_som": [1, 5],
        "num_soms" : [2,5],
        "meta_grid_edge_length": [2, 10],
        "meta_grid_dim": [2, 3],
        "meta_learning_rate": [1.0e-5, 10.0],
        "meta_sigma": [0.1, 10.0],
        "num_epochs_meta": [1, 5],
    }

    checkpoint = Checkpoint(checkpoint_dir,dataset_params['som_type'])
    checkpoint.seed()

    data = data_class.load(data_path)
    sampler = TPESampler(seed=10)  # Make the sampler behave in a deterministic way.
    study = optuna.create_study(sampler=sampler,direction='maximize')
    study.optimize(wrapped_objective(data,som_params,fedsom_params,optimization_params,dataset_params,checkpoint), n_trials=n_trials)

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
    embeddings,labels_true, labels_learned = score_som(data,dataset_params,checkpoint)
    best_params_string = dict_2_string(best_params)
    scatter_path = results_dir / f"{run_name}__scatter_true_labels.png"
    scatter_plot(embeddings, labels_true, best_params_string, scatter_path)
    scatter_path = results_dir / f"{run_name}__scatter_learned_labels.png"
    scatter_plot(embeddings, labels_learned, best_params_string, scatter_path)

    fig = optuna.visualization.plot_parallel_coordinate(study, params=list(best_params.keys()))
    idx = len(best_params_string)//2
    fig.update_layout(
        title=best_params_string[:idx] + '\n' + best_params_string[idx:],
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
        title=best_params_string[:idx] + '\n' + best_params_string[idx:],
        font=dict(size=14),
        height=1200,  # Set the height in pixels
        width=2000,  # Set the width in pixels
    )
    pio.write_image(fig, results_dir / f"{run_name}__bar.png")
    del fig











