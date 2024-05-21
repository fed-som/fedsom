import numpy as np  
import argparse
import os    
import yaml     
import joblib
from pathlib import Path 
from deepclustering.datasets.datasets import *     
from deepclustering.som.som_temp import SelfOrganizingMap
from deepclustering.som.federated_som import FederatedSelfOrganizingMap
from torch.utils.data import Dataset, DataLoader
import torch   
from sklearn.metrics import normalized_mutual_info_score
from deepclustering.utils.general_utils import dict_2_string
from deepclustering.utils.graphics_utils import * #image_array_to_image,tensor_to_image_array,vector_to_square_tensor
import itertools 
from deepclustering.utils.graphics_utils import scatter_plot
from sklearn.metrics import normalized_mutual_info_score,accuracy_score,adjusted_rand_score



def _get_args():
    parser = argparse.ArgumentParser(description="deepclustering embedding creation")
    parser.add_argument("config_path",nargs="?",type=str,default=os.getenv("config_path"))
    return parser.parse_args()


def load_yaml_config(fpath):

    if 's3' in fpath:
        with sm_open(fpath) as file_handle:
            config = yaml.load(file_handle, Loader=yaml.FullLoader)
    else:
        with open(fpath, "r") as f:
            config = yaml.full_load(f)
    return ConfigObject(config)


class ConfigObject:
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigObject(value))
            else:
                setattr(self, key, value)



class Checkpoint(object):
    def __init__(self,data_dir,som_name,clear_old=True):

        self.data_dir = data_dir
        self.som_name = som_name 
        if clear_old:
            self.delete_file(data_dir / self.som_name)

    def save(self,som):

        som.save(self.data_dir / f"{self.som_name}.pt")

    @staticmethod
    def delete_file(filepath):

        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError as e:
                print(f"Error: {e} - {filepath}")
        else:
            print(f"The file {filepath} does not exist.")

    def load_som(self):   
        if 'fed_som' in self.som_name:
            return FederatedSelfOrganizingMap.load(self.data_dir / f"{self.som_name}.pt")
        else:
            return SelfOrganizingMap.load(self.data_dir / f"{self.som_name}.pt")




def train_som(config,checkpoint):


    data_path = config.paths.test_data_path
    data_class = eval(config.data_class)
    data = data_class.load(config.paths.train_data_path)

    num_epochs = config.optimization_params.num_epochs
    batch_size = config.optimization_params.batch_size
    parallel = config.optimization_params.parallel
    batch_size = 2**batch_size

    input_size = data.shape[1]-1
    if config.som_type=='som':
        grid_edge_length = config.som_params.grid_edge_length
        grid_dim = config.som_params.grid_dim
        learning_rate = config.som_params.learning_rate
        sigma = config.som_params.sigma
        num_epochs_som = config.som_params.num_epochs_som
        grid_size = tuple([grid_edge_length for _ in range(grid_dim)])
        som = SelfOrganizingMap(grid_size, 
                                input_size=input_size,
                                learning_rate=learning_rate,
                                sigma=sigma
                                )



    elif config.som_type=='fed_som':
        grid_edge_length = config.som_params.grid_edge_length
        grid_dim = config.som_params.grid_dim
        learning_rate = config.som_params.learning_rate
        sigma = config.som_params.sigma
        num_epochs_som = config.som_params.num_epochs_som
        grid_size = tuple([grid_edge_length for _ in range(grid_dim)])

        meta_grid_edge_length = config.fedsom_params.meta_grid_edge_length
        meta_grid_dim = config.fedsom_params.meta_grid_dim
        meta_learning_rate = config.fedsom_params.meta_learning_rate
        meta_sigma = config.fedsom_params.meta_sigma
        num_epochs_meta = config.fedsom_params.num_epochs_meta
        num_soms = config.fedsom_params.num_soms

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

    num_samples_train = config.num_samples_train
    dataset = data_class(data,num_samples_train)
    if config.som_type=='som':
        for epoch in range(num_epochs):
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for n,(batch,labels,index) in enumerate(dataloader):
                som.batch_train(batch,num_epochs_som)

    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
    embeddings = []
    labels_true = []
    for n,(batch,labels,index) in enumerate(dataloader):
        embeddings.append(batch)
        labels_true.append(labels)
    embeddings = torch.tensor(np.vstack(embeddings))
    if isinstance(labels_true[0][0],str):
        labels_true = [x for L in labels_true for x in L]
    else:
        labels_true = torch.cat(labels_true)


    if config.som_type=='fed_som':
        for epoch in range(num_epochs):
            som.train(embeddings,num_epochs_som,num_epochs_meta)

    labels_learned = som.get_labels(embeddings)
    score = normalized_mutual_info_score(labels_true,labels_learned)
    checkpoint.save(som)


def som_cluster(config,checkpoint):

    np.random.seed(0)
    torch.manual_seed(0)

    data_class = eval(config.data_class)
    test_data_path = config.paths.test_data_path
    test_data = data_class.load(config.paths.test_data_path)
    dataset = data_class(test_data,config.num_samples_test)
    som = checkpoint.load_som()

    dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
    total= []
    labels_list = []
    index_list = []
    for n,(batch, labels, index) in enumerate(dataloader):
        total.append(batch)
        labels_list.append(labels)
        index_list.append(index)
    embeddings = torch.cat(total,dim=0)
    labels_learned = som.get_labels(embeddings)

    if isinstance(labels_list[0][0],str):
        labels_true = [x for L in labels_list for x in L]
    else:
        labels_true = torch.cat(labels_list)

    if isinstance(index_list[0][0],str):
        index = [x for L in index_list for x in L]
    else:
        index = torch.cat(index_list).detach().numpy()

    embeddings_df = pd.DataFrame(embeddings.detach().numpy())
    embeddings_df['labels_true'] = labels_true
    embeddings_df['labels_learned'] = labels_learned
    embeddings_df.index = index   
    embeddings_df.index.name = 'Index'

    return embeddings_df



def som_interpolate(config,checkpoint):

    np.random.seed(0)
    torch.manual_seed(0)

    data_class = eval(config.data_class)
    test_data_path = config.paths.test_data_path
    test_data = data_class.load(config.paths.test_data_path)
    dataset = data_class(test_data,config.num_samples_test)
    som = checkpoint.load_som()

    dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
    total= []
    labels_list = []
    index_list = []
    for n,(batch, labels, index) in enumerate(dataloader):
        total.append(batch)
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
        index = torch.cat(index_list).tolist()

    num_pairs_interpolate = config.num_pairs_interpolate 
    num_pairs_interpolate = 100
    indices = itertools.product(*[range(num_pairs_interpolate),range(num_pairs_interpolate)])
    interpolations = {}
    embeddings = pd.DataFrame(embeddings,index=index)
    embeddings['label'] = labels_true
    for n,(first,second) in enumerate(indices):
        embeddings_values = embeddings[[c for c in embeddings.columns if c!='label']]
        source = torch.tensor(embeddings_values.iloc[first,:])
        target = torch.tensor(embeddings_values.iloc[second,:])
        interpolations[n] = {}
        # source_info = [embeddings.index[first],embeddings['label'][first],torch.tensor(embeddings_values.iloc[first,:].values)]
        # target_info = [embeddings.index[second],embeddings['label'][second],torch.tensor(embeddings_values.iloc[second,:].values)]
        source_info = [embeddings.index[first],embeddings['label'][first]]
        target_info = [embeddings.index[second],embeddings['label'][second]]
        interpolations[n]['bounds'] = [source_info,target_info]
        interpolations[n]['path'] = som.interpolate(source,target,embeddings)

    return interpolations



if __name__=='__main__':

    np.random.seed(0)
    args = _get_args()
    config = load_yaml_config(args.config_path)
    dataset = config.dataset

    results_dir = Path(f'../../../sandbox/results/SOM_SINGLE/{dataset}/{config.som_type}/')
    if not os.path.exists(results_dir):
        results_dir.mkdir(parents=True)  

    checkpoint_dir = Path(f'../../../sandbox/results/SOM_SINGLE/checkpoints/{dataset}/{config.som_type}/')
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    if config.run_training:
        print('Training SOM...')
        som_name = f"{dataset}_{config.som_type}"
        checkpoint = Checkpoint(checkpoint_dir,som_name)
        train_som(config,checkpoint)

    if config.compute_clustering:
        print('Clustering...')
        checkpoint = Checkpoint(checkpoint_dir,som_name,clear_old=False)
        embeddings_df = som_cluster(config,checkpoint)
        embeddings = embeddings_df[[c for c in embeddings_df.columns if c not in ['labels_learned','labels_true']]].values
        labels_true = embeddings_df['labels_true'].values
        labels_learned = embeddings_df['labels_learned'].values

        clustering_scores = pd.DataFrame(index=[0])
        clustering_scores['nmi'] = normalized_mutual_info_score(labels_learned,labels_true)
        clustering_scores['ars'] = adjusted_rand_score(labels_learned,labels_true)
        clustering_scores.index = [f"{dataset}_{config.som_type}"]
        clustering_scores.index.name = 'Index'
        clustering_scores.to_csv(results_dir / f'clustering_scores_{dataset}_{config.som_type}',index='Index')

        print('Rendering...')
        scatter_path = results_dir / f"{som_name}__scatter_true_labels.png"
        scatter_plot(embeddings[:config.num_samples_render,:], labels_true[:config.num_samples_render], f"{dataset}_{som_name}", scatter_path)
        scatter_path = results_dir / f"{som_name}__scatter_learned_labels.png"
        scatter_plot(embeddings[:config.num_samples_render,:], labels_learned[:config.num_samples_render], f"{dataset}_{som_name}", scatter_path)

    if config.compute_interpolations and config.som_type=='som':
        print('Interpolating...')
        checkpoint = Checkpoint(checkpoint_dir,som_name,clear_old=False)
        interpolations = som_interpolate(config,checkpoint)
        joblib.dump(interpolations,results_dir / f"{som_name}_interpolations.joblib")

































