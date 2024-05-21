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
from deepclustering.loss.contrastive_loss import contrastive_loss
from deepclustering.datasets.datasets import MNISTDataset
from deepclustering.utils.graphics_utils import scatter_plot
from deepclustering.utils.general_utils import dict_2_string
import pandas as pd    
import warnings
import os 
warnings.filterwarnings("ignore")




def objective(trial, encoder_params, dataset_params):

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
    batch_size = 2**batch_size
    null_char = 0

    filepath = dataset_params['filepath']
    num_samples_dist = dataset_params['num_samples_dist']
    num_samples_train = dataset_params['num_samples_train']
    num_samples_cluster = dataset_params['num_samples_cluster']

    mnist_data = pd.read_csv(filepath)

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

                embeddings = model(batch_trimmed) 
                embeddings_perturbed = model(perturbed_batch_trimmed)
                optimizer.zero_grad()
                cont_loss = contrastive_loss(embeddings,embeddings_perturbed,temp=temp)

                loss = cont_loss 

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
def wrapped_objective(arg1, arg2):
    return lambda trial: objective(trial, arg1, arg2)




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


    results_dir = '../../../sandbox/results/mnist/testing/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    run_name = 'testing'
    dataset_params = {}
    dataset_params['filepath'] = '../../../sandbox/data/mnist/mnist.csv'
    dataset_params['num_samples_dist'] = 100_0
    dataset_params['num_samples_train'] = 10_0
    dataset_params['num_samples_cluster'] = 1000
    n_trials = 2


    encoder_params = {}
    encoder_params["embedding_dim"] = [3,3]
    encoder_params["hidden_dim"] = [30,200]
    encoder_params["final_dim"] = [5,30]
    encoder_params["temp"] = [0.01,10]
    encoder_params["sparsity_thresh"] = [0.01,0.95]
    encoder_params["approx_constant_thresh"] = [0.999,0.999]
    encoder_params["corruption_factor"] = [0.01,0.99]
    encoder_params["sample_prior"] = [0.01,0.99]
    encoder_params["learning_rate_enc"] = [0.0001,0.1]
    encoder_params["num_epochs"] = [1,3]
    encoder_params["batch_size"] = [2,10]


    # Create an Optuna study

    sampler = TPESampler(seed=10)  # Make the sampler behave in a deterministic way.
    study = optuna.create_study(sampler=sampler,direction='maximize')
    study.optimize(wrapped_objective(encoder_params, dataset_params), n_trials=n_trials)


    # Print the best parameters and objective value
    best_params = study.best_params
    best_value = study.best_value
    print(f"Best parameters: {best_params}")
    print(f"Best value: {best_value}")


    best_params = study.best_params
    embeddings,labels_true = encode(best_params)
    best_params_string = dict_2_string(best_params)
    filepath = f"{results_dir}{run_name}___{best_params_string}_scatter.png"
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
    pio.write_image(fig, f"{results_dir}parallel_{run_name}__{best_params_string}.png")
    del fig

    idx = len(best_params_string)//2
    fig = optuna.visualization.plot_param_importances(study)
    fig.update_layout(
        title=best_params_string[:idx] + '\n' + best_params_string[idx:],
        font=dict(size=18),
        height=1200,  # Set the height in pixels
        width=2000,  # Set the width in pixels
    )
    pio.write_image(fig, f"{results_dir}bar_{run_name}__{best_params_string}.png")
    del fig



















