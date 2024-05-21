from sklearn.metrics import normalized_mutual_info_score
from hdbscan.hdbscan_ import HDBSCAN
import optuna 
import numpy as np   
from deepclustering.encoder.tabular_perturbations_new import TabularPerturbation
import torch  
from deepclustering.encoder.tabular_encoder import TabularNet
from optuna.samplers import TPESampler
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import plotly.express as px
import plotly.io as pio
from deepclustering.loss.contrastive_loss import contrastive_loss,ContrastiveAttention   
from deepclustering.datasets.datasets import MNISTDataset
from deepclustering.loss.vicreg import VICRegCluster
from deepclustering.encoder.custom_metric import Categorical,Metric 
from deepclustering.utils.graphics_utils import scatter_plot
from deepclustering.utils.general_utils import dict_2_string
import pandas as pd    
import warnings
import os 
warnings.filterwarnings("ignore")




def objective(trial, encoder_params, dataset_params, optimization_params):

    np.random.seed(0)
    torch.manual_seed(0)

    embedding_dim = trial.suggest_int("embedding_dim",*encoder_params["embedding_dim"])
    hidden_dim = trial.suggest_int("hidden_dim",*encoder_params["hidden_dim"])
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
    batch_size = 2**batch_size
    null_char = 0

    filepath = dataset_params['filepath']
    num_samples_dist = dataset_params['num_samples_dist']
    num_samples_train = dataset_params['num_samples_train']
    num_samples_cluster = dataset_params['num_samples_cluster']

    contrastive_only = optimization_params['contrastive_only']
    contrastive_attention_only = optimization_params['contrastive_attention_only'] 
    contrastive_vicreg = optimization_params['contrastive_vicreg']
    contrastive_attention_vicreg = optimization_params['contrastive_atttention_vicreg']


    mnist_data = pd.read_csv(filepath)
    vic_reg_cluster_loss = VICRegCluster()



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

    cat_bool_index = tabular_perturbation.get_cat_bool_index()
    metric = Metric(cat_bool_index=cat_bool_index)
    categorical = Categorical(~cat_bool_index,coarseness=coarseness,min_max=False)
    contrastive_attention = ContrastiveAttention(categorical,metric)

    cat_indices = []
    encodings = {}
    x_dim = tabular_perturbation.get_data_dim()
    model = TabularNet(x_dim, cat_indices, encodings, embedding_dim, hidden_dim, final_dim)
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
                if contrastive_only:

                    loss = contrastive_loss(embeddings,embeddings_perturbed,temp=temp)

                elif contrastive_attention_only:

                    loss = contrastive_attention(embeddings,embeddings_perturbed,batch,perturbed_batch,temp)

                elif contrastive_vicreg:
                    
                    cont_loss = contrastive_loss(embeddings,embeddings_perturbed,temp=temp) 
                    representations_total = torch.vstack([representations,representations_perturbed])
                    vicreg_loss = vic_reg_cluster_loss(representations_total)
                    loss = cont_loss + vicreg_loss

                elif contrastive_attention_vicreg:

                    cont_atten_loss = contrastive_attention(embeddings,embeddings_perturbed,batch,perturbed_batch,temp)
                    representations_total = torch.vstack([representations,representations_perturbed])
                    vicreg_loss = vic_reg_cluster_loss(representations_total)
                    loss = cont_atten_loss + vicreg_loss 

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
        labels_learned = HDBSCAN().fit_predict(embeddings.detach().numpy())
    except Exception:
        return 0

    return normalized_mutual_info_score(labels_true,labels_learned)



# Create a wrapper function to pass additional arguments
def wrapped_objective(arg1, arg2, arg3):
    return lambda trial: objective(trial, arg1, arg2, arg3)




def encode(encoder_params):

    np.random.seed(0)
    torch.manual_seed(0)

    embedding_dim = encoder_params["embedding_dim"]
    hidden_dim = encoder_params["hidden_dim"]
    final_dim = encoder_params["final_dim"]
    temp = encoder_params["temp"]
    sparsity_thresh = encoder_params["sparsity_thresh"]
    approx_constant_thresh = encoder_params["approx_constant_thresh"]
    corruption_factor = encoder_params["corruption_factor"]
    sample_prior = encoder_params["sample_prior"]
    learning_rate_enc = encoder_params["learning_rate_enc"]
    num_epochs = encoder_params["num_epochs"]
    batch_size = encoder_params["batch_size"]
    coarseness = encoder_params["coarseness"]
    null_char = 0

    filepath = dataset_params['filepath']
    num_samples_dist = dataset_params['num_samples_dist']
    num_samples_train = dataset_params['num_samples_train']
    num_samples_cluster = dataset_params['num_samples_cluster']

    tabular_perturbation = TabularPerturbation(
                                x_dim=784, 
                                sparsity_thresh=sparsity_thresh, 
                                approx_constant_thresh=approx_constant_thresh, 
                                corruption_factor=corruption_factor,
                                sample_prior=sample_prior,
                                null_char=null_char
          )


    mnist_data = pd.read_csv(filepath)

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


    cat_indices = []
    encodings = {}
    x_dim = tabular_perturbation.get_data_dim()
    model = TabularNet(x_dim, cat_indices, encodings, embedding_dim, hidden_dim, final_dim)

    model.eval()
    mnist_dataset = MNISTDataset(mnist_data,num_samples_cluster,'test')

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


    results_dir = '../../../sandbox/results/mnist/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    run_name = 'cont_atten_vicreg'
    dataset_params = {}
    dataset_params['filepath'] = '../../../sandbox/data/mnist/mnist.csv'
    dataset_params['num_samples_dist'] = 100_000
    dataset_params['num_samples_train'] = 10_000
    dataset_params['num_samples_cluster'] = 2500
    n_trials = 500

    optimization_params = {}
    optimization_params['contrastive_only'] = False
    optimization_params['contrastive_attention_only'] = False
    optimization_params['contrastive_vicreg'] = False
    optimization_params['contrastive_atttention_vicreg'] = True
    assert sum([x for x in optimization_params.values()])==1


    encoder_params = {}
    encoder_params["embedding_dim"] = [3,3]
    encoder_params["hidden_dim"] = [50,400]
    encoder_params["final_dim"] = [5,50]
    encoder_params["temp"] = [0.01,10]
    encoder_params["sparsity_thresh"] = [0.01,0.95]
    encoder_params["approx_constant_thresh"] = [0.5,0.999]
    encoder_params["corruption_factor"] = [0.01,0.99]
    encoder_params["sample_prior"] = [0.01,0.99]
    encoder_params["learning_rate_enc"] = [0.0001,0.1]
    encoder_params["num_epochs"] = [15,15]
    encoder_params["batch_size"] = [5,10]
    encoder_params["coarseness"] = [1,25,50,75,100]


    # Create an Optuna study

    sampler = TPESampler(seed=10)  # Make the sampler behave in a deterministic way.
    study = optuna.create_study(sampler=sampler,direction='maximize')
    study.optimize(wrapped_objective(encoder_params, dataset_params, optimization_params), n_trials=n_trials)


    # Print the best parameters and objective value
    best_params = study.best_params
    best_value = study.best_value
    print(f"Best parameters: {best_params}")
    print(f"Best value: {best_value}")


    best_params = study.best_params
    embeddings,labels_true = encode(best_params)
    best_params_string = dict_2_string(best_params)
    filepath = f"{results_dir}{run_name}__scatter.png"
    scatter_plot(embeddings, labels_true, best_params_string, filepath)

    fig = optuna.visualization.plot_parallel_coordinate(study, params=[
                    "embedding_dim",
                    "hidden_dim",
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
    pio.write_image(fig, f"{results_dir}{run_name}__parallel.png")
    del fig

    idx = len(best_params_string)//2
    fig = optuna.visualization.plot_param_importances(study)
    fig.update_layout(
        title=best_params_string[:idx] + '\n' + best_params_string[idx:],
        font=dict(size=18),
        height=1200,  # Set the height in pixels
        width=2000,  # Set the width in pixels
    )
    pio.write_image(fig, f"{results_dir}{run_name}__bar.png")
    del fig



















