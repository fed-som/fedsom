from sklearn.metrics import normalized_mutual_info_score
from hdbscan.hdbscan_ import HDBSCAN
import optuna 
import numpy as np   
from deepclustering.encoder.tabular_perturbations import TabularPerturbation, DataTrimmer
import torch  
from deepclustering.encoder.tabular_encoder import TabularNet
from optuna.samplers import TPESampler
import torch.optim as optim
import joblib 
from torch.utils.data import Dataset, DataLoader
import plotly.express as px
import plotly.io as pio
from deepclustering.loss.contrastive_loss import contrastive_loss,ContrastiveAttention   
from deepclustering.datasets.datasets import EmberDataset,MNISTDataset
from deepclustering.loss.vicreg import VICRegCluster,VICReg
from pathlib import Path    
from deepclustering.encoder.custom_metric import Categorical,Metric 
from deepclustering.utils.graphics_utils import scatter_plot
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
        if clear_old:
            self.delete_file(data_dir / 'model.pt')
            self.delete_file(data_dir / 'tabular_perturbation.pt')
            self.delete_file(data_dir / 'data_trimmer.pt')
            self.delete_file(data_dir / 'score.joblib')

    def __call__(self,score,model,tabular_perturbation,data_trimmer):

        best_score = self.load_best_score()
        if score>best_score:
            model.save(self.data_dir / 'model.pt')
            tabular_perturbation.save(self.data_dir / 'tabular_perturbation.pt')
            data_trimmer.save(self.data_dir / 'data_trimmer.pt')
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
        return joblib.load(self.data_dir / 'score.joblib') 

    def overwrite_best_score(self,score):
        joblib.dump(score, self.data_dir / 'score.joblib')

    def load_data_trimmer(self):
        return DataTrimmer.load(self.data_dir / 'data_trimmer.pt')

    def load_tabular_perturbation(self):
        return TabularPerturbation.load(self.data_dir / 'tabular_perturbation.pt')

    def load_model(self):
        return TabularNet.load(self.data_dir / 'model.pt')




def objective(trial, encoder_params, dataset_params, optimization_params, checkpoint):

    np.random.seed(0)
    torch.manual_seed(0)

    embedding_dim = trial.suggest_int("embedding_dim",*encoder_params["embedding_dim"])
    representation_dim = trial.suggest_int("representation_dim",*encoder_params["representation_dim"])
    final_dim = trial.suggest_int("final_dim",*encoder_params["final_dim"])
    temp = trial.suggest_float("temp",*encoder_params["temp"],log=True)
    sparsity_thresh = trial.suggest_float("sparsity_thresh",*encoder_params["sparsity_thresh"])
    approx_constant_thresh = trial.suggest_float("approx_constant_thresh",*encoder_params["approx_constant_thresh"])
    corruption_factor = trial.suggest_float("corruption_factor",*encoder_params["corruption_factor"])
    sample_prior = trial.suggest_float("sample_prior",*encoder_params["sample_prior"])
    learning_rate_enc = trial.suggest_float("learning_rate_enc",*encoder_params["learning_rate_enc"],log=True)
    num_epochs = trial.suggest_int("num_epochs",*encoder_params["num_epochs"])
    batch_size = trial.suggest_int("batch_size",*encoder_params["batch_size"])
    coarseness = trial.suggest_categorical("coarseness",encoder_params["coarseness"])
    loss_algo = trial.suggest_categorical("loss_algo",optimization_params["loss_algo"])
    batch_size = 2**batch_size
    null_char = 0

    data_path = dataset_params['data_path']
    data_class = dataset_params['data_class']
    num_samples_dist = dataset_params['num_samples_dist']
    num_samples_train = dataset_params['num_samples_train']
    num_samples_cluster = dataset_params['num_samples_cluster']

    data = data_class.load(data_path)
    dataset = data_class(data,num_samples_dist,'distribution')
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)
    data_trimmer = DataTrimmer(approx_constant_thresh=approx_constant_thresh)
    for n,(batch,labels) in enumerate(dataloader):
        batch = pd.DataFrame(batch)
        data_trimmer.learn_sparsities(batch)

    x_dim = data_trimmer.get_data_dim()
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
        batch = data_trimmer.trim_constant_columns(batch)
        tabular_perturbation.update(batch)

    categorical_indices = tabular_perturbation.get_categorical_indices()
    encodings = train_categorical_encodings(big_batch.iloc[:,categorical_indices])
    cat_bool_index = tabular_perturbation.get_cat_bool_index()
    metric = Metric(cat_bool_index=cat_bool_index)
    categorical = Categorical(~cat_bool_index,coarseness=coarseness,min_max=False)
    gpu_count = torch.cuda.device_count()
    device = assign_device(trial.number, gpu_count)
    contrastive_attention = ContrastiveAttention(categorical,metric,device=device)
    vic_reg_cluster_loss = VICRegCluster()
    vic_reg = VICReg(sim_coeff=100.,std_coeff=10.,cov_coeff=1.)


    model = TabularNet(x_dim, categorical_indices, encodings, embedding_dim, representation_dim, final_dim,device=device)
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate_enc,
    )
    dataset = data_class(data,num_samples_train,'train')
    for epoch in range(num_epochs):

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for n,(batch,labels) in enumerate(dataloader):

            batch = pd.DataFrame(batch)
            if batch.shape[0]>1:

                batch = data_trimmer.trim_constant_columns(batch)
                perturbed_batch = tabular_perturbation.perturb_batch(batch)
                representations = model.create_representation(batch) 
                representations_perturbed = model.create_representation(perturbed_batch)
                embeddings = model.final(representations)
                embeddings_perturbed = model.final(representations_perturbed)

                optimizer.zero_grad()
                if loss_algo=='contrastive_only':
                    loss = contrastive_loss(embeddings,embeddings_perturbed,temp=temp)
                elif loss_algo=='contrastive_attention_only':
                    loss = contrastive_attention(embeddings,embeddings_perturbed,batch,perturbed_batch,temp=temp)
                elif loss_algo=='contrastive_vicreg':
                    cont_loss = contrastive_loss(embeddings,embeddings_perturbed,temp=temp) 
                    representations_total = torch.vstack([representations,representations_perturbed])
                    vicreg_loss = vic_reg_cluster_loss(representations_total)
                    loss = cont_loss + vicreg_loss
                elif loss_algo=='contrastive_attention_vicreg':
                    cont_atten_loss = contrastive_attention(embeddings,embeddings_perturbed,batch,perturbed_batch,temp=temp)
                    representations_total = torch.vstack([representations,representations_perturbed])
                    vicreg_loss = vic_reg_cluster_loss(representations_total)
                    loss = cont_atten_loss + vicreg_loss 
                elif loss_algo=='vicreg_only':
                    loss = vic_reg(embeddings,embeddings_perturbed)

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
        labels_learned = HDBSCAN().fit_predict(embeddings.cpu().detach().numpy())
    except Exception:
        return 0

    score = normalized_mutual_info_score(labels_true,labels_learned)
    checkpoint(score,model,tabular_perturbation,data_trimmer)

    return score



# Create a wrapper function to pass additional arguments
def wrapped_objective(arg1, arg2, arg3, arg4):
    return lambda trial: objective(trial, arg1, arg2, arg3, arg4)



def encode(dataset_params,checkpoint):

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
    dataset = data_class(data_class.load(dataset_params['data_path']),dataset_params['num_samples_cluster'],'test')

    dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
    total= []
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

    return embeddings,labels_true



if __name__=='__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('dataset',type=str,default='mnist')

    args = parser.parse_args()
    dataset = args.dataset 

    results_dir = Path(f'../../../sandbox/results/{dataset}/')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)  

    checkpoint_dir = Path(f'../../../sandbox/tmp_experiment_data/{dataset}/')
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)


    if dataset=='mnist':

        data_path = Path('../../../sandbox/data/mnist/mnist.csv')
        data_class = MNISTDataset

    elif dataset=='ember':

        data_path = Path('../../../sandbox/data/ember/top_k/ember_top_5.csv')
        data_class = EmberDataset

    elif dataset=='sorel':

        data_path = Path('../../../sandbox/data/sorel/top_k/ember_top_5.csv')
        data_class = SorelDataset


    run_name = dataset
    dataset_params = {}
    dataset_params['data_class'] = data_class
    dataset_params['data_path'] = data_path  
    dataset_params['num_samples_dist'] = 100_00
    dataset_params['num_samples_train'] = 10_00
    dataset_params['num_samples_cluster'] = 250
    n_trials = 3

    optimization_params = {
    'loss_algo' : ['contrastive_only',
                    'contrastive_attention_only',
                    'contrastive_vicreg',
                    'contrastive_attention_vicreg',
                    'vicreg_only']
                        }

    encoder_params = {
                    "embedding_dim" : [3,3],
                    "representation_dim" : [30,400],
                    "sparsity_thresh" : [0.01,0.95],
                    "final_dim" : [5,30],
                    "temp" : [0.1,10],
                    "approx_constant_thresh" : [0.5,0.99],
                    "corruption_factor" : [0.01,0.99],
                    "sample_prior" : [0.01,0.99],
                    "learning_rate_enc": [0.0001,0.1],
                    "num_epochs" : [1,1],
                    "batch_size" :[8,13],
                    "coarseness" : [1,50,100],
    }

    checkpoint = Checkpoint(checkpoint_dir)
    checkpoint.seed()

    sampler = TPESampler(seed=10)  # Make the sampler behave in a deterministic way.
    study = optuna.create_study(sampler=sampler,direction='maximize')
    study.optimize(wrapped_objective(encoder_params, dataset_params, optimization_params, checkpoint), n_trials=n_trials)

    # Print the best parameters and objective value
    best_params = study.best_params
    best_value = study.best_value
    print(f"Best parameters: {best_params}")
    print(f"Best value: {best_value}")
    best_params_df = pd.DataFrame.from_dict(best_params,orient='index')
    best_params_df.to_csv(results_dir / 'best_params.csv')


    best_params = study.best_params
    embeddings,labels_true = encode(dataset_params,checkpoint)
    best_params_string = dict_2_string(best_params)
    scatter_path = results_dir / f"{run_name}__scatter.png"
    scatter_plot(embeddings, labels_true, best_params_string, scatter_path)

    fig = optuna.visualization.plot_parallel_coordinate(study, params=[
                    "embedding_dim",
                    "representation_dim",
                    "final_dim",
                    "temp",
                    "sparsity_thresh",
                    "approx_constant_thresh",
                    "corruption_factor",
                    "sample_prior",
                    "learning_rate_enc",
                    "num_epochs",
                    "batch_size",
                    "coarseness",
                    "loss_algo"
                    ]
    )
    idx = len(best_params_string)//2
    fig.update_layout(
        title=best_params_string[:idx] + '\n' + best_params_string[idx:],
        font=dict(size=18),
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
        font=dict(size=18),
        height=1200,  # Set the height in pixels
        width=2000,  # Set the width in pixels
    )
    pio.write_image(fig, results_dir / f"{run_name}__bar.png")
    del fig



















