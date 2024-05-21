import numpy as np 
import argparse
import os 
from torch.utils.data import DataLoader
import yaml
from pathlib import Path 
import torch
from deepclustering.datasets.datasets import *  
from deepclustering.encoder.tabular_perturbations import TabularPerturbation, DataTrimmer 
from deepclustering.encoder.tabular_encoder import TabularNet
from deepclustering.loss.c3_loss import C3Loss
from deepclustering.loss.contrastive_loss import contrastive_loss,ContrastiveAttention 
from deepclustering.loss.som_loss import som_loss 
from torch.optim.lr_scheduler import ExponentialLR
from deepclustering.loss.vicreg import VICRegCluster,VICReg   
from deepclustering.encoder.custom_metric import Categorical,Metric 
from deepclustering.utils.graphics_utils import scatter_plot
from deepclustering.som.som import SelfOrganizingMap
from deepclustering.utils.device_utils import assign_device  
from deepclustering.encoder.utils.preprocessing_utils import (
    encode_categoricals,
    train_categorical_encodings,
)



class Checkpoint(object):
    def __init__(self,data_dir,clear_old=True):

        self.data_dir = data_dir
        self.model_name = 'model.pt'
        self.tabular_perturbation_name = 'tabular_perturbation.pt'
        self.data_trimmer_name = 'data_trimmer.pt'  
        self.som_name = 'som.pt' 
        if clear_old:
            self.delete_file(data_dir / self.model_name)
            self.delete_file(data_dir / self.tabular_perturbation_name)
            self.delete_file(data_dir / self.data_trimmer_name)
            self.delete_file(data_dir / self.som_name)

    def save(self,model,tabular_perturbation,data_trimmer,som):

        model.save(self.data_dir / self.model_name)
        tabular_perturbation.save(self.data_dir / self.tabular_perturbation_name)
        data_trimmer.save(self.data_dir / self.data_trimmer_name)
        som.save(self.data_dir / self.som_name)
    
    @staticmethod
    def delete_file(filepath):

        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError as e:
                print(f"Error: {e} - {filepath}")
        else:
            print(f"The file {filepath} does not exist.")

    def load_data_trimmer(self):
        return DataTrimmer.load(self.data_dir / self.data_trimmer_name)

    def load_tabular_perturbation(self):
        return TabularPerturbation.load(self.data_dir / self.tabular_perturbation_name)

    def load_model(self):
        return TabularNet.load(self.data_dir / self.model_name)

    def load_som(self):
        return SelfOrganizingMap.load(self.data_dir / self.som_name)


def train_nn_encoder(config,checkpoint):

    np.random.seed(0)
    torch.manual_seed(0)

    loss_algo = config.training_params.optimizer.loss_algo
    loss_weight = config.training_params.optimizer.loss_weight
    embedding_dim = config.training_params.encoder.embedding_dim
    representation_dim = config.training_params.encoder.representation_dim
    final_dim = config.training_params.encoder.final_dim
    temp = config.training_params.encoder.temp
    sparsity_thresh = config.training_params.encoder.sparsity_thresh
    approx_constant_thresh = config.training_params.encoder.approx_constant_thresh
    corruption_factor = config.training_params.encoder.corruption_factor
    sample_prior = config.training_params.encoder.sample_prior
    learning_rate_enc = config.training_params.encoder.learning_rate_enc
    scheduler_gamma = config.training_params.encoder.scheduler_gamma
    num_epochs = config.training_params.encoder.num_epochs
    batch_size = 2**config.training_params.encoder.batch_size
    # coarseness = config.training_params.encoder.coarseness
    vicreg_weight = config.training_params.encoder.vicreg_weight
    zeta = config.training_params.encoder.zeta
    perturb_both = config.training_params.encoder.perturb_both
    null_char = config.training_params.encoder.null_char
    grid_edge_length = config.training_params.som.grid_edge_length
    grid_dim = config.training_params.som.grid_dim
    grid_size = tuple([grid_edge_length for _ in range(grid_dim)])
    learning_rate_som = config.training_params.som.learning_rate_som
    sigma = config.training_params.som.sigma
    num_epochs_som = config.training_params.som.num_epochs_som
    cosine = config.training_params.som.cosine

    data_class = eval(config.data_class)
    train_data_path = config.paths.train_data_path
    train_data = data_class.load(config.paths.train_data_path)
    num_samples_train = config.num_samples_train
    dataset = data_class(train_data,num_samples_train,'train')
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)
    data_trimmer = DataTrimmer(approx_constant_thresh=approx_constant_thresh)
    for n,(batch,labels,index) in enumerate(dataloader):
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
    for n,(batch,labels,index) in enumerate(dataloader):
        batch = pd.DataFrame(batch)
        batch = data_trimmer.trim_constant_columns(batch)
        big_batch.append(batch)
    big_batch = pd.concat(big_batch)
    tabular_perturbation.find_categorical_indices(big_batch)

    dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)
    for n,(batch,labels,index) in enumerate(dataloader):
        batch = pd.DataFrame(batch)
        batch = data_trimmer.trim_constant_columns(batch)
        tabular_perturbation.update(batch)

    categorical_indices = tabular_perturbation.get_categorical_indices()
    encodings = train_categorical_encodings(big_batch.iloc[:,categorical_indices])
    cat_bool_index = tabular_perturbation.get_cat_bool_index()
    # metric = Metric(cat_bool_index=cat_bool_index)
    # categorical = Categorical(~cat_bool_index,coarseness=coarseness,min_max=False)
    gpu_count = torch.cuda.device_count()
    device = assign_device(0, gpu_count)
    # contrastive_attention = ContrastiveAttention(categorical,metric,device=device)
    vic_reg_cluster_loss = VICRegCluster()
    vic_reg = VICReg(sim_coeff=vicreg_weight,std_coeff=vicreg_weight)
    c3_loss = C3Loss(zeta=zeta).to(device)

    model = TabularNet(x_dim, categorical_indices, encodings, embedding_dim, representation_dim, final_dim,device=device)
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate_enc,
    )
    scheduler = ExponentialLR(optimizer, gamma=scheduler_gamma)

    som = SelfOrganizingMap(grid_size, input_size=representation_dim, learning_rate=learning_rate_som, sigma=sigma)
    som.eval()
    dataset = data_class(train_data,num_samples_train,'train')
    for epoch in range(num_epochs):

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for n,(batch,labels,index) in enumerate(dataloader):

            batch = pd.DataFrame(batch)
            if batch.shape[0]>1:
                batch = data_trimmer.trim_constant_columns(batch)
                if perturb_both:
                    perturbed_batch = tabular_perturbation.perturb_batch(batch)
                    representations = model.create_representation(perturbed_batch)
                else:
                    representations = model.create_representation(batch) 

                perturbed_batch = tabular_perturbation.perturb_batch(batch)
                representations_perturbed = model.create_representation(perturbed_batch)

                embeddings = model.final_layer(representations)
                embeddings_perturbed = model.final_layer(representations_perturbed)

                representations_total = torch.vstack([representations,representations_perturbed])
                # som.batch_train(representations,num_epochs_som)

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

                # som_loss_value = som_loss(representations,representations_perturbed,cosine=cosine)
                # loss = loss_weight*encoder_loss + (1-loss_weight)*som_loss_value
                loss = encoder_loss

                loss.backward()
                optimizer.step()
        scheduler.step()

    checkpoint.save(model,tabular_perturbation,data_trimmer,som)



def encode(dataloader,config,checkpoint):

    np.random.seed(0)
    torch.manual_seed(0)

    gpu_count = torch.cuda.device_count()
    device = assign_device(0, gpu_count)

    data_trimmer = checkpoint.load_data_trimmer()
    tabular_perturbation = checkpoint.load_tabular_perturbation()
    som = checkpoint.load_som()
    model = checkpoint.load_model()
    model.to(device)
    model.eval()

    total= []
    labels_list = []
    index_list = []
    for n,(batch, labels, index) in enumerate(dataloader):
        embeddings = model.create_representation(data_trimmer.trim_constant_columns(pd.DataFrame(batch)))
        total.append(embeddings)
        labels_list.append(labels)
        index_list.append(index)
    embeddings = torch.cat(total,dim=0)
    if isinstance(labels_list[0][0],str):
        labels_true = [x for L in labels_list for x in L]
    else:
        labels_true = torch.cat(labels_list)
    if isinstance(index_list[0][0],str):
        index = [x for L in index_list for x in L]
    else:
        index = torch.cat(index_list).detach().numpy()
    
    embeddings_df = pd.DataFrame(embeddings.detach().numpy())
    embeddings_df['label'] = labels_true
    embeddings_df.index = index   
    embeddings_df.index.name = 'Index'

    return embeddings_df


def load_yaml_config(fpath):

    if 's3' in fpath:
        with sm_open(fpath) as file_handle:
            config = yaml.load(file_handle, Loader=yaml.FullLoader)
    else:
        with open(fpath, "r") as f:
            config = yaml.full_load(f)
    return ConfigObject(config)


def _get_args():
    parser = argparse.ArgumentParser(description="deepclustering embedding creation")
    parser.add_argument("config_path",nargs="?",type=str,default=os.getenv("config_path"))
    return parser.parse_args()


class ConfigObject:
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigObject(value))
            else:
                setattr(self, key, value)


def config_to_dict(config_obj):
    if isinstance(config_obj, ConfigObject):
        return {key: config_to_dict(value) for key, value in vars(config_obj).items()}
    else:
        return config_obj




if __name__=='__main__':

    # python optuna_encoder_som.py --dataset mnist --n_trials 3

    np.random.seed(0)
    args = _get_args()
    config = load_yaml_config(args.config_path)
    dataset = config.dataset

    results_dir = Path(f'../../../sandbox/results/SINGLE_ENCODER_SOM_EMBER_ONLY/{dataset}/')
    if not os.path.exists(results_dir):
        results_dir.mkdir(parents=True)  

    checkpoint_dir = Path(f'../../../sandbox/results/SINGLE_ENCODER_SOM_EMBER_ONLY/checkpoint/{dataset}/')
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)



    checkpoint = Checkpoint(checkpoint_dir)
    train_nn_encoder(config,checkpoint)

    data_class = eval(config.data_class)
    train_data_path = config.paths.train_data_path
    train_data = data_class.load(config.paths.train_data_path)
    num_samples_train = config.num_samples_train
    dataset = data_class(train_data,num_samples_train,'train',train_proportion=1.0)
    train_dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)

    test_data_path = config.paths.test_data_path
    test_data = data_class.load(config.paths.test_data_path)
    num_samples_test = config.num_samples_test
    dataset = data_class(test_data,num_samples_test,'train',train_proportion=1.0)
    test_dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)

    train_embeddings = encode(train_dataloader,config,checkpoint)
    train_embeddings.to_csv(results_dir / f'{config.dataset}_train_encoder_embeddings.csv',index='Index')
    test_embeddings = encode(test_dataloader,config,checkpoint)
    test_embeddings.to_csv(results_dir / f'{config.dataset}_test_encoder_embeddings.csv',index='Index')

    dataset = config.dataset
    scatter_path = results_dir / f"{dataset}__scatter_true_labels.png"
    X = test_embeddings[[c for c in test_embeddings.columns if c!='label']].values
    labels = test_embeddings.label.values
    scatter_plot(X, labels, dataset, scatter_path)

    print('completed')





















