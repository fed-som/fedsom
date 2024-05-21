import copy
import warnings
from pathlib import Path     
import torch
import numpy as np     
import torch.nn as nn     
import pandas as pd    
import networkx as nx     
from deepclustering.som.utils.som_utils import bmu_to_string,convert_bmu_labels
from sklearn.metrics import calinski_harabasz_score

warnings.filterwarnings("ignore")




def unravel_index(index, shape):
    unravel_idx = []
    for size in reversed(shape):
        unravel_idx.append(index % size)
        index //= size
    return tuple(reversed(unravel_idx))


class SelfOrganizingMap(nn.Module):
    def __init__(self, grid_size, input_size, learning_rate=0.1, sigma=1.0):

        super().__init__()
        self.grid_size = grid_size
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.sigma = sigma

        torch.manual_seed(0)
        self.weights = torch.randn(*grid_size, input_size,requires_grad=False)
        grid = torch.meshgrid(*[torch.arange(size,requires_grad=False) for size in grid_size])
        self.grid_coordinates = torch.stack(grid, dim=-1).float()
        self.graph = self.create_graph()
        self.weights_added = False


    def add_weights_to_graph(self,input_data):

        bmu_array = []
        bmu_labels = {}
        for n, x in enumerate(input_data):
            units = self.find_best_matching_unit(x)
            bmu_array.append(units)
            bmu_labels[n] = bmu_to_string(units)
        numerical_labels = convert_bmu_labels(bmu_labels)
        bmu_array = np.array(bmu_array)

        for u,v,data in self.graph.edges(data=True):

            u = np.array([int(i) for i in u])
            v = np.array([int(i) for i in v])
            cluster_u = input_data[np.all(bmu_array==u,axis=1),:]
            cluster_v = input_data[np.all(bmu_array==v,axis=1),:]
            cluster_u_labels = numerical_labels[np.all(bmu_array==u,axis=1)]
            cluster_v_labels = numerical_labels[np.all(bmu_array==v,axis=1)]
            clusters = np.vstack([cluster_u,cluster_v])
            labels = np.concatenate([cluster_u_labels,cluster_v_labels])

            if cluster_u.shape[0]>2 and cluster_v.shape[0]>2:
                ch = calinski_harabasz_score(clusters,labels)
                data['weight'] = 1-(ch/(ch+1))
            else:
                data['weight'] = 0

        self.weights_added = True


    def find_shortest_path(self,source,target):

        shortest_path= nx.shortest_path(self.graph,source=source,target=target,weight='weight')
        length = nx.shortest_path_length(self.graph,source=source,target=target,weight='weight')

        return shortest_path,length


    def create_graph(self):
        graph = nx.Graph()
        nodes = self.grid_coordinates.view(-1, self.grid_coordinates.shape[-1]).numpy()
        nodes = [tuple(node) for node in nodes]
        graph.add_nodes_from(nodes)
        for node in nodes:
            neighbors = self.find_neighbors(node, nodes)
            graph.add_edges_from([(tuple(node), tuple(neighbor)) for neighbor in neighbors])
        return graph


    def interpolate(self,v,w,dataset):

        dataset_index,dataset_labels,vectors = dataset.index,dataset['label'].values,torch.tensor(dataset[[c for c in dataset.columns if c!='label']].values)

        source = tuple([x.item() for x in self.find_best_matching_unit(v)])
        target = tuple([x.item() for x in self.find_best_matching_unit(w)])
        if not self.weights_added:
            self.add_weights_to_graph(vectors)

        shortest_path,length = self.find_shortest_path(source=source,target=target)
        interpolation = {}
        interpolation['node_bounds'] = [source,target]
        interpolative_vectors = []
        for node in shortest_path:
            node = tuple([int(i) for i in list(node)])
            this_weight_vector = self.weights[node]
            index,vector = self.find_nearest_neighbor(query_vector=this_weight_vector,vectors=vectors)
            # interpolative_vectors.append([dataset_index[index],dataset_labels[index],vector])
            interpolative_vectors.append([dataset_index[index],dataset_labels[index],node])

        interpolation['path'] = interpolative_vectors
        return interpolation


    @staticmethod
    def find_nearest_neighbor(query_vector,vectors):

        distances = torch.norm(vectors-query_vector,dim=1)
        return torch.argmin(distances).item(),vectors[torch.argmin(distances).item()]
        

    def bmu_to_weight_vector(self,bmu):

        return self.weights[bmu]


    def find_neighbors(self, node, nodes):

        neighbors = []
        for other_node in nodes:
            if np.linalg.norm(np.array(node) - np.array(other_node)) == 1:
                neighbors.append(other_node)
        return neighbors

    def map_coordinates_to_nodes(self, coordinates):
        return tuple(coordinates)


    def map_nodes_to_coordinates(self, node):
        return tuple(node)


    def find_best_matching_unit(self, input_vector):
        # Calculate the Euclidean distances between input vector and SOM weights
        distances = torch.norm(input_vector - self.weights, dim=-1)

        # Find the index of the best matching unit (BMU)
        bmu_index = unravel_index(torch.argmin(distances), self.grid_size)

        return bmu_index

    def update_weights(self, input_vector, bmu_index):
        # Calculate the neighborhood function
        distances = torch.norm(self.grid_coordinates - self.grid_coordinates[bmu_index], dim=-1)
        neighborhood = torch.exp(-distances / (2 * self.sigma**2))

        # Update the SOM weights
        delta = self.learning_rate * neighborhood[..., None] * (input_vector - self.weights)
        self.weights += delta

    def batch_train(self, input_data, num_epochs):
        for epoch in range(num_epochs):
            for input_vector in input_data:
                bmu_index = self.find_best_matching_unit(input_vector)
                self.update_weights(input_vector, bmu_index)

    def score(self, input_vector):
        bmu_index = self.find_best_matching_unit(input_vector)
        return self.weights[bmu_index]


    def get_weight_vectors(self,input_vectors):

        return torch.vstack([self.score(v) for v in input_vectors])


    def get_labels(self, input_vectors):
        bmu_labels = {}
        for n, x in enumerate(input_vectors):
            units = self.find_best_matching_unit(x)
            bmu_labels[n] = bmu_to_string(units)

        return convert_bmu_labels(bmu_labels)


    def save(self, model_filepath: Path):

        checkpoint = {
        "grid_size" : self.grid_size,
        "input_size" : self.input_size,
        "learning_rate" : self.learning_rate,
        "sigma" : self.sigma,
        "weights" : self.weights,
        "graph": self.graph
        }

        torch.save(checkpoint, str(model_filepath))

    @classmethod
    def load(cls,model_filepath: Path):

        checkpoint = torch.load(model_filepath)
        obj = cls(**{k:v for k,v in checkpoint.items() if k not in ['weights','graph']})
        obj.weights = checkpoint['weights']
        if 'graph' in checkpoint.keys():
            obj.graph = checkpoint['graph']
        return obj



if __name__ == "__main__":
    

    # som = SelfOrganizingMap(grid_size=(4,4), input_size=10, learning_rate=0.1, sigma=1.0)
    # print(som.weights)
    # som.save('./test.pt')

    # obj = SelfOrganizingMap.load('./test.pt')
    # print(obj.weights)


    from sklearn.datasets import make_blobs
    from deepclustering.loss.som_loss import som_loss





    n_samples = 1000
    n_clusters = 10    
    centers = n_clusters
    X, _ = make_blobs(n_samples=n_samples, n_features=10, centers=centers, random_state=42, cluster_std=1.)




    grid_size = (5,5)
    learning_rate = 1.5
    sigma = .01
    num_epochs = 10  
    som = SelfOrganizingMap(grid_size, input_size=X.shape[1], learning_rate=learning_rate, sigma=sigma)
    som.batch_train(torch.tensor(X), num_epochs=num_epochs)

    input_vectors = torch.tensor(X)
    labels = som.get_labels(input_vectors)


    X = torch.tensor(X)
    v = X[0,:]
    w = X[1,:]
    interpolation = som.interpolate(v,w,X)

    print(interpolation)




    # print(labels)

    # X = torch.tensor(X)

    # v = X[:16,:]
    # w = X[16:32,:]

    # loss = som_loss(v,w,som,on_bmu=True)
    # print(loss)


    # loss = som_loss(v,w,som,on_bmu=False)
    # print(loss)

    # bmu_labels = {}
    # for n, x in enumerate(X):
    #     units = som.find_best_matching_unit(torch.tensor(x))
    #     bmu_labels[n] = bmu_to_string(units)

    # labels = convert_bmu_labels(bmu_labels)



    # import numpy as np
    # from sklearn.manifold import TSNE

 
    # tsne = TSNE(n_components=2)  # Reduce to 2 dimensions
    # X_reduced = tsne.fit_transform(X)



    # # import umap
    # # umap_model = umap.UMAP(n_components=2)  # Reduce to 2 dimensions
    # # X_reduced = umap_model.fit_transform(X)


    # scatter_plot(X_reduced, labels,"tab20")


















    # from deepclustering.som.utils.som_utils import bmu_to_string,convert_bmu_labels
    # # Example usage
    # grid_size = (5, 5)  # Example grid size: 5x5
    # input_size = 6  # Example input vector size
    # input_data = torch.randn(1000, input_size)  # Example input data with 100 samples

    # print(input_data)


    # # print('training first batch')
    # som = SelfOrganizingMap(grid_size, input_size)
    # # som.train(input_data, num_epochs=10)

    # for i in range(1):
    #     print(f"training batch {i}")
    #     input_data = torch.randn(1000, input_size)
    #     som.train(input_data, num_epochs=10)


    # # Score a new input vector
    # new_input = torch.randn(3, input_size)#torch.randn(input_size)  # Example new input vector
    # # score = som.score(new_input)
    # # indices = som.find_best_matching_unit(new_input)
    # # print("Score:", score)
    # # print("Indices:", indices)


    # bmu_labels = {}
    # print(new_input)
    # for n, x in enumerate(new_input):
    #     units = som.find_best_matching_unit(torch.tensor(x))
    #     bmu_labels[n] = bmu_to_string(units)

    # labels = convert_bmu_labels(bmu_labels)
    # print(labels)


    # print('=============================================================')

    # print(som.weights.shape)
    # weight = som.weights[0,0,:]
    # print(weight)
    # print(weight.shape)


    # print('-----------------------------------------------------------')

    # fixed_coordinates = torch.tensor([0,0])
    # weight = som.weights[tuple(fixed_coordinates) + (slice(None),)]

    # print(weight)
    # print(weight.shape)









