



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from deepclustering.encoder.tabular_perturbations import TabularPerturbation,recast_columns 
from deepclustering.encoder.tabular_encoder import TabularNet
from deepclustering.encoder.contrastive_loss import contrastive_loss
from deepclustering.som.distributed_som import DistributedSelfOrganizingMap
from deepclustering.som.som import SelfOrganizingMap     
from deepclustering.som.utils.som_utils import bmu_to_string, convert_bmu_labels
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np    



# Custom Dataset class for MNIST
class MNISTDataset(Dataset):
    def __init__(self, csv_file,num_samples):
        self.data = pd.read_csv(csv_file).loc[:num_samples,:]
        self.labels = self.data['class']
        self.features = (1./255)*self.data[[c for c in self.data.columns if c!='class']].values.astype('float32')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.features[idx]
        label = self.labels[idx]
        return image, label


def scatter_plot(X, labels, best_params_string, filepath):

    if X.shape[1]>2:
        tsne = TSNE(n_components=2)  # Reduce to 2 dimensions
        X = tsne.fit_transform(X)

    fig = plt.figure(figsize=(14, 8))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab20")  #'viridis')
    plt.title(best_params_string, fontsize=8, fontweight="bold")
    plt.colorbar(label="Cluster")
    plt.show(block=False)
    input('')
    # plt.savefig(filepath)
    plt.close()
    del fig




if __name__=='__main__':



    csv_file = '../../../sandbox/data/mnist/mnist.csv'


    num_samples_dist = 10_000
    num_samples_train = 100_000
    num_samples_cluster = 10_000
    batch_size = 512
    x_dim = 784
    embedding_dim = 3
    hidden_dim = 20
    final_dim = 10
    learning_rate_enc = 0.00001
    sample_prior = 0.1
    num_epochs = 20
    use_true_labels = True

    grid_size = (4,4)
    input_size = hidden_dim
    learning_rate_som = 1.5
    sigma = 1e-5
    num_epochs_som = 4



    
    tabular_perturbation = TabularPerturbation(x_dim,sample_prior=sample_prior,sparse=False,null_char=None)
    mnist_dataset = MNISTDataset(csv_file,num_samples_dist)
    dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)
    first_iteration = True
    for n,(batch, _) in enumerate(dataloader):

        batch = pd.DataFrame(batch)
        if first_iteration:
            tabular_perturbation.find_categorical_indices(batch)
            tabular_perturbation.initialize_distributions()
            first_iteration = False
        tabular_perturbation.update(batch)
        print(f'{n} of {mnist_dataset.data.shape[0]/batch_size}')
    print('Distributions learned.')


    cat_indices = []
    encodings = {}
    model = TabularNet(x_dim, cat_indices, encodings, embedding_dim, hidden_dim, final_dim)
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate_enc,
    )
    mnist_dataset = MNISTDataset(csv_file,num_samples_train)
    for epoch in range(num_epochs):

        dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)
        for n,(batch,labels) in enumerate(dataloader):

            batch = pd.DataFrame(batch)
            perturbed_batch = tabular_perturbation.perturb_batch(batch)
            embeddings = model(batch)
            embeddings_perturbed = model(perturbed_batch)
            optimizer.zero_grad()
            loss = contrastive_loss(embeddings,embeddings_perturbed)
            loss.backward()
            optimizer.step()

            if n%100==0:
                print(f'Training {n+1}: {loss.item()}  total batches: {np.ceil(mnist_dataset.data.shape[0]/batch_size).astype(int)}')



    print('Encoder trained.')


    model.eval()
    mnist_dataset = MNISTDataset(csv_file,num_samples_cluster)
    dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)
    total= []
    labels_list = []
    for n,(batch, labels) in enumerate(dataloader):
        embeddings = model(pd.DataFrame(batch))
        total.append(embeddings)
        labels_list.append(labels)
    embeddings = torch.cat(total,dim=0)



    if use_true_labels:
        labels = torch.cat(labels_list,dim=0)
    else:
        print('Training SOM')
        som = SelfOrganizingMap(grid_size=grid_size, input_size=input_size, learning_rate=learning_rate_som, sigma=sigma)
        som.train(embeddings.detach(),num_epochs=num_epochs_som)

        bmu_labels = {}
        for n, x in enumerate(embeddings):
            units = som.find_best_matching_unit(x)
            bmu_labels[n] = bmu_to_string(units)
        labels = convert_bmu_labels(bmu_labels)




    filepath = 'test.png'
    scatter_plot(embeddings.detach().numpy(), labels, 'hi', filepath) 


































