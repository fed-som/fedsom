from sklearn.metrics import normalized_mutual_info_score
from hdbscan.hdbscan_ import HDBSCAN
import optuna 
import numpy as np   
from deepclustering.encoder.tabular_perturbations import TabularPerturbation
import torch  
from deepclustering.encoder.tabular_encoder import TabularNet
from optuna.samplers import TPESampler
import torch.optim as optim
import joblib 
from torch.utils.data import Dataset, DataLoader
import plotly.express as px
import plotly.io as pio
from deepclustering.loss.contrastive_loss import contrastive_loss,ContrastiveAttention   
from deepclustering.datasets.datasets import MNISTDataset
from deepclustering.loss.vicreg import VICRegCluster,VICReg
from pathlib import Path    
from deepclustering.encoder.custom_metric import Categorical,Metric 
from deepclustering.utils.graphics_utils import scatter_plot
from deepclustering.utils.general_utils import dict_2_string
from deepclustering.utils.device_utils import assign_device
import pandas as pd    
import optuna
from joblib import Parallel, delayed
import warnings
import os 
warnings.filterwarnings("ignore")


class Checkpoint(object):
    def __init__(self,data_dir,clear_old=True):

        self.data_dir = data_dir 
        if clear_old:
            self.delete_file(data_dir / 'model.pt')
            self.delete_file(data_dir / 'tabular_perturbation.pt')
            self.delete_file(data_dir / 'score.joblib')

    def __call__(self,score,model,tabular_perturbation):

        best_score = self.load_best_score()
        if score>best_score:
            model.save(self.data_dir / 'model.pt')
            tabular_perturbation.save(self.data_dir / 'tabular_perturbation.pt')
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
    num_samples_dist = dataset_params['num_samples_dist']
    num_samples_train = dataset_params['num_samples_train']
    num_samples_cluster = dataset_params['num_samples_cluster']

    mnist_data = pd.read_csv(data_path)
    vic_reg_cluster_loss = VICRegCluster()
    vic_reg = VICReg(sim_coeff=100.,std_coeff=10.,cov_coeff=1.)

    gpu_count = torch.cuda.device_count()
    device = assign_device(trial.number, gpu_count)

    tabular_perturbation = TabularPerturbation(
                                x_dim=784, 
                                sparsity_thresh=sparsity_thresh, 
                                approx_constant_thresh=approx_constant_thresh, 
                                corruption_factor=corruption_factor,
                                sample_prior=sample_prior,
                                null_char=null_char
          )
    big_batch = []
    mnist_dataset = MNISTDataset(mnist_data,num_samples_dist,'distribution')
    dataloader = DataLoader(mnist_dataset, batch_size=1000, shuffle=False)
    for n,(batch,labels) in enumerate(dataloader):
        batch = pd.DataFrame(batch)
        big_batch.append(batch)
    big_batch = pd.concat(big_batch)
    tabular_perturbation.find_categorical_indices(big_batch)

    dataloader = DataLoader(mnist_dataset, batch_size=1000, shuffle=False)
    for n,(batch,labels) in enumerate(dataloader):

        batch = pd.DataFrame(batch)
        tabular_perturbation.update(batch)

    cat_bool_index = tabular_perturbation.get_trimmed_cat_bool_index()
    metric = Metric(cat_bool_index=cat_bool_index)

    categorical = Categorical(~cat_bool_index,coarseness=coarseness,min_max=False)
    contrastive_attention = ContrastiveAttention(categorical,metric,device=device)

    cat_indices = []
    encodings = {}

    x_dim = tabular_perturbation.get_data_dim()
    model = TabularNet(x_dim, cat_indices, encodings, embedding_dim, representation_dim, final_dim,device=device)
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate_enc,
    )
    mnist_dataset = MNISTDataset(mnist_data,num_samples_train,'train')


    for epoch in range(num_epochs):

        dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=False)
        for n,(batch,labels) in enumerate(dataloader):

            batch = pd.DataFrame(batch)
            if batch.shape[0]>1:
                perturbed_batch = tabular_perturbation.perturb_batch(batch)
                batch_trimmed = tabular_perturbation.trim_constant_columns(batch)
                perturbed_batch_trimmed = tabular_perturbation.trim_constant_columns(perturbed_batch)

                representations = model.create_representation(batch_trimmed) 
                representations_perturbed = model.create_representation(perturbed_batch_trimmed)

                embeddings = model.final(representations)
                embeddings_perturbed = model.final(representations_perturbed)

                optimizer.zero_grad()
                if loss_algo=='contrastive_only':

                    loss = contrastive_loss(embeddings,embeddings_perturbed,temp=temp)

                elif loss_algo=='contrastive_attention_only':

                    loss = contrastive_attention(embeddings,embeddings_perturbed,batch_trimmed,perturbed_batch_trimmed,temp=temp)

                elif loss_algo=='contrastive_vicreg':
                    
                    cont_loss = contrastive_loss(embeddings,embeddings_perturbed,temp=temp) 
                    representations_total = torch.vstack([representations,representations_perturbed])
                    vicreg_loss = vic_reg_cluster_loss(representations_total)
                    loss = cont_loss + vicreg_loss

                elif loss_algo=='contrastive_attention_vicreg':

                    cont_atten_loss = contrastive_attention(embeddings,embeddings_perturbed,batch_trimmed,perturbed_batch_trimmed,temp=temp)
                    representations_total = torch.vstack([representations,representations_perturbed])
                    vicreg_loss = vic_reg_cluster_loss(representations_total)
                    loss = cont_atten_loss + vicreg_loss 

                elif loss_algo=='vicreg_only':

                    loss = vic_reg(embeddings,embeddings_perturbed)

                loss.backward()
                optimizer.step()


    model.eval()
    mnist_dataset = MNISTDataset(mnist_data,num_samples_cluster,'test')
    dataloader = DataLoader(mnist_dataset, batch_size=100, shuffle=False)
    total = []
    labels_list = []
    for n,(batch, labels) in enumerate(dataloader):
        embeddings = model.create_representation(tabular_perturbation.trim_constant_columns(pd.DataFrame(batch)))
        total.append(embeddings)
        labels_list.append(labels)
    embeddings = torch.cat(total,dim=0)
    labels_true = torch.cat(labels_list)

    try:
        labels_learned = HDBSCAN().fit_predict(embeddings.cpu().detach().numpy())
    except Exception:
        return 0

    score = normalized_mutual_info_score(labels_true,labels_learned)
    checkpoint(score,model,tabular_perturbation)

    return score



# Create a wrapper function to pass additional arguments
def wrapped_objective(arg1, arg2, arg3, arg4):
    return lambda trial: objective(trial, arg1, arg2, arg3, arg4)



def encode(dataset_params,checkpoint):

    np.random.seed(0)
    torch.manual_seed(0)

    gpu_count = torch.cuda.device_count()
    device = assign_device(0, gpu_count)

    tabular_perturbation = checkpoint.load_tabular_perturbation()
    model = checkpoint.load_model()
    model.to(device)
    model.eval()
    mnist_dataset = MNISTDataset(pd.read_csv(dataset_params['data_path']),dataset_params['num_samples_cluster'],'test')

    dataloader = DataLoader(mnist_dataset, batch_size=100, shuffle=False)
    total= []
    labels_list = []
    for n,(batch, labels) in enumerate(dataloader):
        embeddings = model.create_representation(tabular_perturbation.trim_constant_columns(pd.DataFrame(batch)))
        total.append(embeddings)
        labels_list.append(labels)
    embeddings = torch.cat(total,dim=0)
    labels_true = torch.cat(labels_list)

    return embeddings,labels_true






if __name__=='__main__':


    results_dir = Path('../../../sandbox/results/mnist/')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    checkpoint_dir = Path('../../../sandbox/tmp_experiment_data/')
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    data_path = Path('../../../sandbox/data/mnist/mnist.csv')


    run_name = 'TESTING'
    dataset_params = {}
    dataset_params['data_path'] = data_path   
    dataset_params['num_samples_dist'] = 100_000
    dataset_params['num_samples_train'] = 100_000
    dataset_params['num_samples_cluster'] = 2500
    n_trials = 150

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
                    "num_epochs" : [10,10],
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



















