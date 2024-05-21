from sklearn.metrics import normalized_mutual_info_score
from hdbscan.hdbscan_ import HDBSCAN
import optuna 
import numpy as np   
from deepclustering.encoder.tabular_perturbations import TabularPerturbation, DataTrimmer
import torch  
from deepclustering.encoder.lenet_encoder import LeNet5
from optuna.samplers import TPESampler
import torch.optim as optim
import joblib 
from torch.utils.data import Dataset, DataLoader
import plotly.express as px
from deepclustering.loss.c3_loss import C3Loss
import plotly.io as pio
from deepclustering.loss.contrastive_loss import contrastive_loss,ContrastiveAttention 
from deepclustering.loss.som_loss import som_loss,SOMLoss 
from deepclustering.datasets.datasets import EmberDataset,MNISTDataset,SorelDataset#,CIFAR10Dataset
from deepclustering.loss.vicreg import VICRegCluster,VICReg
from pathlib import Path    
from deepclustering.encoder.custom_metric import Categorical,Metric 
from deepclustering.utils.graphics_utils import scatter_plot
from deepclustering.som.som import SelfOrganizingMap
from deepclustering.utils.general_utils import dict_2_string
from deepclustering.utils.device_utils import assign_device
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


class Checkpoint(object):
    def __init__(self,data_dir,clear_old=True):

        self.data_dir = data_dir
        self.model_name = 'model.pt'
        self.tabular_perturbation_name = 'tabular_perturbation.pt'
        self.data_trimmer_name = 'data_trimmer.pt'  
        self.score_name = 'score.joblib'
        self.som_name = 'som.pt' 
        if clear_old:
            self.delete_file(data_dir / self.model_name)
            self.delete_file(data_dir / self.tabular_perturbation_name)
            self.delete_file(data_dir / self.data_trimmer_name)
            self.delete_file(data_dir / self.score_name)
            self.delete_file(data_dir / self.som_name)

    def __call__(self,score,model,tabular_perturbation,data_trimmer):

        best_score = self.load_best_score()
        if score>best_score:
            model.save(self.data_dir / self.model_name)
            tabular_perturbation.save(self.data_dir / self.tabular_perturbation_name)
            data_trimmer.save(self.data_dir / self.data_trimmer_name)
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

    def load_data_trimmer(self):
        return DataTrimmer.load(self.data_dir / self.data_trimmer_name)

    def load_tabular_perturbation(self):
        return TabularPerturbation.load(self.data_dir / self.tabular_perturbation_name)

    def load_model(self):
        return TabularNet.load(self.data_dir / self.model_name)

    def load_som(self):
        return SelfOrganizingMap.load(self.data_dir / self.som_name)



def objective(trial, data, encoder_params, som_parameters, dataset_params, optimization_params, checkpoint):

    np.random.seed(0)
    torch.manual_seed(0)

    # embedding_dim = trial.suggest_int("embedding_dim",*encoder_params["embedding_dim"])
    representation_dim = trial.suggest_int("representation_dim",*encoder_params["representation_dim"])
    final_dim = trial.suggest_int("final_dim",*encoder_params["final_dim"])
    # temp = trial.suggest_float("temp",*encoder_params["temp"],log=True)
    sparsity_thresh = trial.suggest_float("sparsity_thresh",*encoder_params["sparsity_thresh"],log=True)
    # approx_constant_thresh = trial.suggest_float("approx_constant_thresh",*encoder_params["approx_constant_thresh"],log=True)
    corruption_factor = trial.suggest_float("corruption_factor",*encoder_params["corruption_factor"],log=True)
    sample_prior = trial.suggest_float("sample_prior",*encoder_params["sample_prior"],log=True)
    learning_rate_enc = trial.suggest_float("learning_rate_enc",*encoder_params["learning_rate_enc"],log=True)
    num_epochs = trial.suggest_int("num_epochs",*encoder_params["num_epochs"])
    batch_size = trial.suggest_int("batch_size",*encoder_params["batch_size"])
    # coarseness = trial.suggest_categorical("coarseness", encoder_params["coarseness"])
    # vicreg_weight = trial.suggest_int("vicreg_weight",*encoder_params["vicreg_weight"])
    zeta = trial.suggest_float("zeta",*encoder_params["zeta"],log=True)
    perturb_both = trial.suggest_categorical("perturb_both",encoder_params["perturb_both"])
    loss_algo = trial.suggest_categorical("loss_algo",optimization_params["loss_algo"])
    loss_weight = trial.suggest_float("loss_weight",*optimization_params["loss_weight"],log=True)
    batch_size = 2**batch_size
    null_char = 0
  
    grid_edge_length = trial.suggest_int("grid_edge_length",*som_parameters["grid_edge_length"])
    grid_dim = trial.suggest_int("grid_dim",*som_parameters["grid_dim"])
    learning_rate_som = trial.suggest_float("learning_rate_som",*som_parameters["learning_rate_som"],log=True)
    sigma = trial.suggest_float("sigma",*som_parameters["sigma"],log=True)
    num_epochs_som = trial.suggest_int("num_epochs_som",*som_parameters["num_epochs_som"])
    cosine = trial.suggest_categorical("cosine",som_parameters["cosine"])
    grid_size = tuple([grid_edge_length for _ in range(grid_dim)])

    data_path = dataset_params['data_path']
    data_class = dataset_params['data_class']
    num_samples_dist = dataset_params['num_samples_dist']
    num_samples_train = dataset_params['num_samples_train']
    num_samples_cluster = dataset_params['num_samples_cluster']

    dataset = data_class(data,num_samples_dist,'distribution')
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)
    data_trimmer = DataTrimmer(approx_constant_thresh=1.1)#approx_constant_thresh)
    for n,(batch,labels) in enumerate(dataloader):
        batch = pd.DataFrame(batch)
        data_trimmer.learn_sparsities(batch)

    x_dim = 784# = data_trimmer.get_data_dim()
    tabular_perturbation = TabularPerturbation(
                                x_dim=x_dim, 
                                sparsity_thresh=sparsity_thresh, 
                                corruption_factor=corruption_factor,
                                sample_prior=sample_prior,
                                null_char=null_char
          )

    big_batch = []
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)
    for n,(batch,labels) in enumerate(dataloader):
        batch = pd.DataFrame(batch)
        batch = data_trimmer.trim_constant_columns(batch)
        big_batch.append(batch)
    big_batch = pd.concat(big_batch)
    tabular_perturbation.find_categorical_indices(big_batch)



    dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)
    for n,(batch,labels) in enumerate(dataloader):
        batch = pd.DataFrame(batch)
        tabular_perturbation.update(batch)

    categorical_indices = tabular_perturbation.get_categorical_indices()
    encodings = train_categorical_encodings(big_batch.iloc[:,categorical_indices])
    cat_bool_index = tabular_perturbation.get_cat_bool_index()
    # metric = Metric(cat_bool_index=cat_bool_index)
    # categorical = Categorical(~cat_bool_index,coarseness=coarseness,min_max=False)
    gpu_count = torch.cuda.device_count()
    device = assign_device(trial.number, gpu_count)
    # contrastive_attention = ContrastiveAttention(categorical,metric,device=device)
    # vic_reg_cluster_loss = VICRegCluster()
    # vic_reg = VICReg(sim_coeff=vicreg_weight,std_coeff=vicreg_weight)
    c3_loss = C3Loss(zeta=zeta).to(device)

    model = LeNet5(representation_dim, final_dim)
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate_enc,
    )

    som = SelfOrganizingMap(grid_size, input_size=representation_dim, learning_rate=learning_rate_som, sigma=sigma)
    som_loss = SOMLoss(som,cosine=cosine)
    som.eval()
    dataset = data_class(data,num_samples_train,'train')
    for epoch in range(num_epochs):

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for n,(batch,labels) in enumerate(dataloader):

            batch = pd.DataFrame(batch)
            if batch.shape[0]>1:

                if perturb_both:
                    perturbed_batch = tabular_perturbation.perturb_batch(batch)
                    representations = model.create_representation(perturbed_batch)
                else:
                    representations = model.create_representation(batch) 

                perturbed_batch = tabular_perturbation.perturb_batch(batch)
                representations_perturbed = model.create_representation(perturbed_batch)

                embeddings = model.final_layer(representations)
                # print(embeddings)
                embeddings_perturbed = model.final_layer(representations_perturbed)

                representations_total = torch.vstack([representations,representations_perturbed])
                som.batch_train(representations,num_epochs_som)

                optimizer.zero_grad()
                if loss_algo=='contrastive_only':
                    encoder_loss = contrastive_loss(embeddings,embeddings_perturbed,temp=temp)
                elif loss_algo=='contrastive_attention_only':
                    encoder_loss = contrastive_attention(embeddings,embeddings_perturbed,batch,perturbed_batch,temp=temp)
                elif loss_algo=='contrastive_vicreg':
                    cont_loss = contrastive_loss(embeddings,embeddings_perturbed,temp=temp) 
                    representations_total = torch.vstack([representations,representations_perturbed])
                    vicreg_loss = vic_reg_cluster_loss(representations_total)
                    encoder_loss = cont_loss + vicreg_loss
                elif loss_algo=='contrastive_attention_vicreg':
                    cont_atten_loss = contrastive_attention(embeddings,embeddings_perturbed,batch,perturbed_batch,temp=temp)
                    representations_total = torch.vstack([representations,representations_perturbed])
                    vicreg_loss = vic_reg_cluster_loss(representations_total)
                    encoder_loss = cont_atten_loss + vicreg_loss 
                elif loss_algo=='vicreg_only':
                    encoder_loss = vic_reg(embeddings,embeddings_perturbed)
                elif loss_algo=='c3_loss':
                    encoder_loss = c3_loss(embeddings,embeddings_perturbed)

                som_loss_value = som_loss(representations,representations_perturbed)
                loss = loss_weight*encoder_loss + (1-loss_weight)*som_loss_value
                loss.backward()
                optimizer.step()


    model.eval()
    dataset = data_class(data,num_samples_cluster,'test')
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
    total = []
    labels_list = []
    for n,(batch, labels) in enumerate(dataloader):
        embeddings = model.create_representation(data_trimmer.trim_constant_columns(pd.DataFrame(batch)))
        total.append(embeddings)
        labels_list.append(labels)
    embeddings = torch.cat(total,dim=0)

    if isinstance(labels_list[0][0],str):
        labels_list = [x for L in labels_list for x in L]
        labels_dict = dict(zip(sorted(list(set(labels_list))),range(len(set(labels_list)))))
        labels_true = np.array([labels_dict[key] for key in labels_list])
    else:
        labels_true = torch.cat(labels_list)
    try:
        labels_learned = som.get_labels(embeddings)
    except Exception:
        return 0


    score = normalized_mutual_info_score(labels_true,labels_learned)
    checkpoint(score,model,tabular_perturbation,data_trimmer,som)

    return score


# Create a wrapper function to pass additional arguments
def wrapped_objective(arg1, arg2, arg3, arg4, arg5, arg6):
    return lambda trial: objective(trial, arg1, arg2, arg3, arg4, arg5, arg6)


def encode(data,dataset_params,checkpoint):

    np.random.seed(0)
    torch.manual_seed(0)

    gpu_count = torch.cuda.device_count()
    device = assign_device(0, gpu_count)

    data_trimmer = checkpoint.load_data_trimmer()
    tabular_perturbation = checkpoint.load_tabular_perturbation()
    model = checkpoint.load_model()
    model.to(device)
    model.eval()
    data_class = dataset_params['data_class']
    dataset = data_class(data,dataset_params['num_samples_cluster'],'test')

    dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
    total= []
    labels_list = []
    for n,(batch, labels) in enumerate(dataloader):
        embeddings = model.create_representation(data_trimmer.trim_constant_columns(pd.DataFrame(batch)))
        total.append(embeddings)
        labels_list.append(labels)
    embeddings = torch.cat(total,dim=0)

    labels_learned = som.get_labels(embeddings)

    if isinstance(labels_list[0][0],str):
        labels_true = [x for L in labels_list for x in L]
    else:
        labels_true = torch.cat(labels_list)

    return embeddings,labels_true,labels_learned



if __name__=='__main__':

    # python optuna_encoder_som_lenet.py --dataset mnist --n_trials 3

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str,default='mnist')
    parser.add_argument('--n_trials',type=int,default=30)
    parser.add_argument('--num_samples_dist',type=int,default=1000)
    parser.add_argument('--num_samples_train',type=int,default=1000)
    parser.add_argument('--num_samples_cluster',type=int,default=250)

    args = parser.parse_args()
    dataset = args.dataset 
    n_trials = args.n_trials
    num_samples_dist = args.num_samples_dist
    num_samples_train = args.num_samples_train
    num_samples_cluster = args.num_samples_cluster

    results_dir = Path(f'../../../sandbox/results/TMP/OTHER/{dataset}/')
    if not os.path.exists(results_dir):
        results_dir.mkdir(parents=True)  

    checkpoint_dir = Path(f'../../../sandbox/tmp_experiment_data/{dataset}/')
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)


    if dataset=='mnist':
        data_path = Path('../../../sandbox/data/mnist/mnist.csv')
        data_class = MNISTDataset

    elif dataset=='cifar10':
        data_path = Path('../../../sandbox/data/cifar10/cifar10_test.csv')
        data_class = CIFAR10Dataset

    elif dataset=='ember':
        data_path = Path('../../../sandbox/data/ember/top_k/ember_top_5.csv')
        data_class = EmberDataset

    elif dataset=='sorel':
        data_path = Path('../../../sandbox/data/sorel/sorel_subset.csv')
        data_class = SorelDataset


    run_name = dataset
    dataset_params = {}
    dataset_params['data_class'] = data_class
    dataset_params['data_path'] = data_path  
    dataset_params['num_samples_dist'] = num_samples_dist
    dataset_params['num_samples_train'] = num_samples_train
    dataset_params['num_samples_cluster'] = num_samples_cluster


    optimization_params = {
    'loss_algo' : [#'contrastive_only',
                    # 'contrastive_attention_only',
                    # 'contrastive_vicreg',
                    # 'contrastive_attention_vicreg',
                    # 'vicreg_only',
                    'c3_loss'],
    'loss_weight': [0.0001,0.999]}

    encoder_params = {
                    # "embedding_dim" : [2,5],
                    "representation_dim" : [5,400],
                    "final_dim" : [2,40],
                    # "temp" : [0.1,10],
                    "sparsity_thresh" : [0.001,0.95],
                    # "approx_constant_thresh" : [0.5,0.999],
                    "corruption_factor" : [0.01,0.99],
                    "sample_prior" : [0.01,0.99],
                    "learning_rate_enc": [0.00001,0.1],
                    "num_epochs" : [5,5],
                    "batch_size" :[4,10],
                    # "coarseness" : [1,25,50,75,100],
                    # "vicreg_weight": [1,100],
                    "zeta": [0.1,10],
                    "perturb_both": [False,True]
    }

    som_params = {
        "grid_edge_length": [2,6],
        "grid_dim": [2, 4],
        "learning_rate_som": [1.0e-5, 10.0],
        "sigma": [0.1, 10.0],
        "num_epochs_som": [1, 5],
        "cosine": [True,False]
    }

    checkpoint = Checkpoint(checkpoint_dir,LeNet5)
    checkpoint.seed()

    data = data_class.load(data_path)
    sampler = TPESampler(seed=10)  # Make the sampler behave in a deterministic way.
    study = optuna.create_study(sampler=sampler,direction='maximize')
    study.optimize(wrapped_objective(data, encoder_params, som_params, dataset_params, optimization_params, checkpoint), n_trials=n_trials)

    # Print the best parameters and objective value
    best_params = study.best_params
    best_value = study.best_value
    print(f"Best parameters: {best_params}")
    print(f"Best value: {best_value}")
    best_params_df = pd.DataFrame.from_dict(best_params,orient='index')
    best_params_df.to_csv(results_dir / 'best_params.csv')


    best_params = study.best_params
    embeddings,labels_true,labels_learned = encode(data,dataset_params,checkpoint)
    best_params_string = dict_2_string(best_params)
    scatter_path = results_dir / f"{run_name}__scatter_true_labels.png"
    scatter_plot(embeddings, labels_true, best_params_string, scatter_path)
    scatter_path = results_dir / f"{run_name}__scatter_learned_labels.png"
    scatter_plot(embeddings, labels_learned, best_params_string, scatter_path)

    fig = optuna.visualization.plot_parallel_coordinate(study, params=[
                    # "embedding_dim",
                    "representation_dim",
                    "final_dim",
                    # "temp",
                    "sparsity_thresh",
                    # "approx_constant_thresh",
                    "corruption_factor",
                    "sample_prior",
                    "learning_rate_enc",
                    "num_epochs",
                    "batch_size",
                    # "coarseness",
                    # "vicreg_weight",
                    "loss_algo",
                    "loss_weight",
                    "grid_edge_length",
                    "grid_dim",
                    "learning_rate_som",
                    "sigma",
                    "num_epochs_som",
                    "cosine",
                    "zeta",
                    "perturb_both"
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



















