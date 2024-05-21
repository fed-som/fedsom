from deepclustering.som.som_temp import SelfOrganizingMap
from deepclustering.som.utils.som_utils import convert_bmu_labels,bmu_to_string
from sklearn.datasets import make_blobs
import numpy as np   
import matplotlib.pyplot as plt
from multiprocessing import Manager, Pool
import torch    
import torch.nn as nn 
from pathlib import Path    



def train_som_parallel(som_dict_parallel,som_dict,key,input_data,num_epochs):

    som_dict[key].batch_train(torch.tensor(input_data),num_epochs)
    som_dict_parallel[key] = som_dict[key]   


class SomAtlas(nn.Module):
    def __init__(self,soms,cluster_labels=None):
        self.soms = soms 
        self.cluster_labels = cluster_labels

    # provide the algo with disparate datasets, embeddings from different modalities 
    # train a single som for each cluster
    # then train fed_som 
    # then interpolate between clusters by tracing a path through the meta_som






class FederatedSelfOrganizingMap(nn.Module):
    def __init__(self,grid_size,num_soms,input_size,learning_rate,meta_learning_rate,sigma,meta_sigma,meta_grid_dim,parallel=False):
        super().__init__()
        self.num_soms = num_soms  
        self.grid_size = grid_size
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.sigma = sigma

        self.meta_learning_rate = meta_learning_rate
        self.meta_sigma = meta_sigma 
        self.meta_grid_dim = meta_grid_dim  
        self.som_dict = {i:SelfOrganizingMap(grid_size, input_size=input_size, learning_rate=learning_rate, sigma=sigma) for i in range(num_soms)}

        self.partition_indices = None
        self.parallel = parallel 

    def train(self,input_data,num_epochs_som,num_epochs_meta):

        splits = np.array_split(input_data, self.num_soms) 
        self.partition_indices = np.repeat(np.arange(self.num_soms), [len(part) for part in splits])

        if self.parallel:
            som_dict_parallel = Manager().dict()
            with Pool() as p:
                p.starmap(
                    train_som_parallel,
                    [[som_dict_parallel,self.som_dict,key,splits[split_index],num_epochs_som] for split_index,key in enumerate(self.som_dict.keys())],
                )
            p.close()
            p.join()
            self.som_dict = dict(som_dict_parallel)
        else:
            for split_index,key in enumerate(self.som_dict.keys()):
                self.som_dict[key].batch_train(torch.tensor(splits[split_index]),num_epochs = num_epochs_som)

        weights_list = [s.weights.view(np.prod(s.weights.shape[:-1]),self.input_size) for s in self.som_dict.values()]
        weight_tensor = torch.cat(weights_list, dim=0)
        grid_edge_length = np.ceil(np.power(self.num_soms*np.prod(self.grid_size),1./self.meta_grid_dim)).astype(int)
        self.meta_grid_size = tuple([grid_edge_length for _ in range(self.meta_grid_dim)])
        self.meta_som = SelfOrganizingMap(self.meta_grid_size, input_size=self.input_size, learning_rate=self.meta_learning_rate, sigma=self.meta_sigma)
        self.meta_som.batch_train(weight_tensor,num_epochs = num_epochs_meta)


    def find_best_matching_unit(self,input_vector,partition):

        unit = self.som_dict[partition].find_best_matching_unit(torch.tensor(input_vector))
        coords = torch.cat([tensor.unsqueeze(0) for tensor in unit])
        weight = self.som_dict[partition].weights[tuple(coords) + (slice(None),)]
        return self.meta_som.find_best_matching_unit(weight) 


    def get_partition_indices(self):

        return self.partition_indices

    def get_labels(self,input_vectors):

        bmu_labels = {}
        partition_indices = self.get_partition_indices()
        for n,(partition,x) in enumerate(zip(partition_indices,input_vectors)):
            bmu = self.find_best_matching_unit(x,partition)
            bmu_labels[n] =  bmu_to_string(bmu)
        
        return convert_bmu_labels(bmu_labels) 


    def save(self,model_filepath: Path):

        checkpoint = {
        "num_soms":self.num_soms,  
        "grid_size":self.grid_size,
        "input_size":self.input_size,
        "learning_rate":self.learning_rate,
        "sigma":self.sigma,
        "meta_learning_rate":self.meta_learning_rate,
        "meta_sigma":self.meta_sigma,
        "meta_grid_dim":self.meta_grid_dim,
        "partition_indices":self.partition_indices,
        "parallel":self.parallel,
        }
        som_dict = {}
        for i in range(self.num_soms):
            som_dict[i] = {"grid_size":self.som_dict[i].grid_size,
                           "input_size":self.som_dict[i].input_size,
                           "learning_rate":self.som_dict[i].learning_rate,
                           "sigma":self.som_dict[i].sigma,
                           "weights":self.som_dict[i].weights}
        checkpoint['som_dict'] = som_dict
        checkpoint['meta_som'] = {"grid_size":self.meta_grid_size,
                                  "input_size":self.input_size,
                                  "learning_rate":self.meta_learning_rate,
                                  "sigma":self.meta_sigma,
                                  "weights":self.meta_som.weights}
        torch.save(checkpoint,str(model_filepath))


    @classmethod
    def load(cls,model_filepath: Path):

        checkpoint = torch.load(model_filepath)
        obj = cls(**{k:v for k,v in checkpoint.items() if k not in ['som_dict','partition_indices','meta_som']})
        obj.partition_indices = checkpoint['partition_indices']
        som_dict = {}
        for i in range(len(checkpoint['som_dict'])):
            this_som_kwargs = {this_k:this_v for this_k,this_v in checkpoint['som_dict'][i].items() if this_k!='weights'}
            this_som = SelfOrganizingMap(**this_som_kwargs)
            this_som.weights = checkpoint['som_dict'][i]['weights']
            som_dict[i] = this_som   
        obj.som_dict = som_dict
        meta_som_kwargs = {k:v for k,v in checkpoint['meta_som'].items() if k!='weights'}
        meta_som = SelfOrganizingMap(**meta_som_kwargs)
        meta_som.weights = checkpoint['meta_som']['weights']
        obj.meta_som = meta_som 
        return obj    










if __name__=='__main__':



    plt.style.use('ggplot')

    n_samples = 10000
    centers = 50
    cluster_std = 0.1
    num_soms = 5


    X, labels = make_blobs(n_samples=n_samples, centers=centers, random_state=42, cluster_std=cluster_std)
    


    grid_size = (8,8)
    input_size = 2
    learning_rate = 1.5
    meta_learning_rate = 1.5  
    sigma = 0.01
    meta_sigma = 0.01
    meta_grid_dim = 2
    parallel = True                         
    fed_som = FederatedSelfOrganizingMap(grid_size,num_soms,input_size,learning_rate,meta_learning_rate,sigma,meta_sigma,meta_grid_dim,parallel=parallel)

    num_epochs = 40  
    num_epochs_meta = 40
    fed_som.batch_train(X,num_epochs,num_epochs_meta)
    labels = fed_som.get_labels(X)

    fig = plt.figure(figsize=(14, 8))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab20",alpha=0.2)  #'viridis')
    plt.title('distributed', fontsize=12, fontweight="bold")
    plt.colorbar(label="Cluster")
    plt.show(block=False)
    # input('enter to close')
    filepath = '../../../sandbox/som_results/fed_som.png'
    plt.savefig(filepath)
    plt.close()




# ===============================================================================================================================

    # n_samples = 10000
    # centers = 50
    # cluster_std = 0.1
    # num_soms = 5
    # grid_size = (8,8)


    # X, labels = make_blobs(n_samples=n_samples, centers=centers, random_state=42, cluster_std=cluster_std)
    

    
    # splits = np.array_split(X, num_soms) 
    # partition_indices = np.repeat(np.arange(num_soms), [len(part) for part in splits])



    
    # input_size = X.shape[1]
    # learning_rate = 1.5
    # sigma = 0.01
    # som_dict = {i:SelfOrganizingMap(grid_size, input_size=input_size, learning_rate=learning_rate, sigma=sigma) for i in range(num_soms)}



    # num_epochs = 40
    # for key,split_index in zip(som_dict.keys(),range(len(splits))):
    #     som_dict[key].train(torch.tensor(splits[split_index]),num_epochs = num_epochs)


    # weights_list = [s.weights.view(np.prod(s.weights.shape[:-1]),input_size) for s in som_dict.values()]
    # weight_tensor = torch.cat(weights_list, dim=0)
    # grid_edge_length = np.ceil(np.sqrt(num_soms*np.prod(grid_size))).astype(int)
    # grid_size = (grid_edge_length,grid_edge_length)
    # meta_som = SelfOrganizingMap(grid_size, input_size=input_size, learning_rate=learning_rate, sigma=sigma)
    # num_epochs = 30
    # meta_som.train(weight_tensor,num_epochs = num_epochs)



    # bmu_labels = {}
    # for n,(partition,x) in enumerate(zip(partition_indices,X)):

    #     unit = som_dict[partition].find_best_matching_unit(torch.tensor(x))
    #     coords = torch.cat([tensor.unsqueeze(0) for tensor in unit])
    #     weight = som_dict[partition].weights[tuple(coords) + (slice(None),)]
    #     bmu = meta_som.find_best_matching_unit(weight) 
    #     bmu_labels[n] = bmu_to_string(bmu)

    # labels = convert_bmu_labels(bmu_labels)
    

    # print(labels)



    # fig = plt.figure(figsize=(14, 8))
    # plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab20",alpha=0.2)  #'viridis')
    # plt.title('distributed', fontsize=12, fontweight="bold")
    # plt.colorbar(label="Cluster")
    # plt.show(block=False)
    # input('enter to close')
    # # plt.savefig(filepath)
    # plt.close()

























