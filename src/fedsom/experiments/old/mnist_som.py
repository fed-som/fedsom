



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from deepclustering.encoder.tabular_perturbations_new import TabularPerturbation
from deepclustering.encoder.tabular_encoder import TabularNet
from deepclustering.encoder.contrastive_loss import contrastive_loss
from deepclustering.som.distributed_som import DistributedSelfOrganizingMap
from deepclustering.som.som import SelfOrganizingMap     
from deepclustering.som.utils.som_utils import bmu_to_string, convert_bmu_labels
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.manifold import TSNE
import umap    
import numpy as np    

from deepclustering.encoder.utils.data_utils import recast_columns
from deepclustering.som.som import SelfOrganizingMap
from deepclustering.som.utils.som_utils import bmu_to_string, convert_bmu_labels





def compute_off_diagonal_sum(points):

    points = points - points.mean(axis=0)

    # Compute the covariance matrix
    covariance_matrix = torch.matmul(points.T, points) / (points.size(0) - 1)

    # Extract the off-diagonal elements
    diagonal_mask = torch.eye(covariance_matrix.size(0), dtype=torch.bool)
    off_diagonal = covariance_matrix[~diagonal_mask]

    off_diagonal_sum = torch.norm(off_diagonal)

    # Compute the sum of the off-diagonal elements
    # off_diagonal_sum = torch.sum(off_diagonal)
    

    return off_diagonal_sum



def calinski_harabasz_score(embeddings, labels):
    # Compute the centroids for each cluster
    unique_labels = torch.unique(labels)
    num_clusters = len(unique_labels)

    # centroids = torch.stack(results)
    centroids = torch.stack([embeddings[labels == label].mean(0) for label in unique_labels])

    # Compute the total number of samples
    total_samples = embeddings.shape[0]

    # Compute the between-cluster scatter matrix
    between_scatter = torch.norm(centroids.unsqueeze(1) - centroids.unsqueeze(0), dim=2).pow(2)
    between_scatter = between_scatter.sum(dim=1)*(embeddings.shape[0]-num_clusters)

    within_scatter = torch.stack([
        F.pairwise_distance(embeddings[labels == label], centroids[i].unsqueeze(0)).pow(2).sum()
        for i, label in enumerate(unique_labels)
    ]) * (num_clusters-1)

    score = torch.sum(between_scatter) / (torch.sum(within_scatter))

    return score



def covariance_loss(embeddings,labels):

    unique_labels = torch.unique(labels)
    return torch.sum(torch.stack([compute_off_diagonal_sum(embeddings[labels == label]) for label in unique_labels if (labels==label).sum().item()>1]))


def calinski_harabazs_loss(embeddings,labels):

    return -calinski_harabasz_score(embeddings, labels)




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
        umap_model = umap.UMAP(n_components=2,random_state=42)
        X = umap_model.fit_transform(X)

    fig = plt.figure(figsize=(14, 8))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab20")  #'viridis')

    for i, label in enumerate(labels):
        if np.random.rand()<0.1:
            plt.annotate(label.item(), (X[i,0], X[i,1]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.title(best_params_string, fontsize=8, fontweight="bold")
    plt.colorbar(label="Cluster")
    plt.savefig(filepath)
    plt.close()
    del fig



def plot_mnist_digits_side_by_side(vector1, vector2):

    image_size = int(np.sqrt(len(vector1)))
    mnist_digit_image1 = np.array(vector1).reshape(image_size, image_size)
    mnist_digit_image2 = np.array(vector2).reshape(image_size, image_size)

    # Create a subplot with 1 row and 2 columns
    plt.subplot(1, 2, 1)
    plt.imshow(mnist_digit_image1, cmap='gray')
    plt.title('MNIST Digit 1')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(mnist_digit_image2, cmap='gray')
    plt.title('MNIST Digit 2')
    plt.axis('off')

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Show the combined image
    plt.show(block=False)
    input('')
    plt.close()





if __name__=='__main__':


    np.random.seed(0)
    torch.manual_seed(0)


    csv_file = '../../../sandbox/data/mnist/mnist.csv'


    num_samples_dist = 10_0000
    num_samples_train = 10_000
    num_samples_cluster = 10_00
    x_dim = 784




    batch_size = 512
    embedding_dim = 3
    hidden_dim = 20
    final_dim = 10
    learning_rate_enc = 0.002
    temp = 2.
    num_epochs = 60

    sparsity_thresh = 0.95
    approx_constant_thresh = 0.99
    corruption_factor = 0.5
    sample_prior = 0.5
    null_char = None




    use_true_labels = True
    show_perturbed_digits = False

    grid_size = (4,4)
    input_size = hidden_dim
    learning_rate_som = 1.5
    sigma = 1e-5
    num_epochs_som = 1



    

    tabular_perturbation = TabularPerturbation(
                                x_dim=x_dim, 
                                sparsity_thresh=sparsity_thresh, 
                                approx_constant_thresh=approx_constant_thresh, 
                                corruption_factor=corruption_factor,
                                sample_prior=sample_prior,
                                null_char=null_char
          )

    big_batch = []
    mnist_dataset = MNISTDataset(csv_file,num_samples_dist)
    dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)
    for n,(batch,labels) in enumerate(dataloader):

        batch = pd.DataFrame(batch)
        big_batch.append(batch)
    big_batch = pd.concat(big_batch)
    tabular_perturbation.find_categorical_indices(big_batch)


    dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)
    for n,(batch,labels) in enumerate(dataloader):

        batch = pd.DataFrame(batch)
        tabular_perturbation.update(batch)

        print(f'{n+1} of {np.ceil(mnist_dataset.data.shape[0]/batch_size).astype(int)}')


    print('Distributions learned.')



    som = SelfOrganizingMap(grid_size, input_size=hidden_dim, learning_rate=learning_rate_som, sigma=sigma)




    cat_indices = []
    encodings = {}
    x_dim = tabular_perturbation.get_data_dim()
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

            if show_perturbed_digits:
                perturbed_batch = tabular_perturbation.perturb_batch(batch)
                vector1 = batch.values[0,:]
                vector2 = perturbed_batch.values[0,:]
                plot_mnist_digits_side_by_side(vector1, vector2)


            perturbed_batch = tabular_perturbation.perturb_batch(batch)
            batch_trimmed = tabular_perturbation.trim_constant_columns(batch)
            perturbed_batch_trimmed = tabular_perturbation.trim_constant_columns(perturbed_batch)

            embeddings = model(batch_trimmed) 
            embeddings_perturbed = model(perturbed_batch_trimmed)
            optimizer.zero_grad()
            cont_loss = contrastive_loss(embeddings,embeddings_perturbed,temp=temp)


            # model.eval()
            representations = model.create_representation(tabular_perturbation.trim_constant_columns(pd.DataFrame(batch)))
            som.train(representations, num_epochs=num_epochs_som)
            # labels = som.get_labels(embeddings)
            bmu_labels = {}
            for i, x in enumerate(representations):
                units = som.find_best_matching_unit(x)
                bmu_labels[i] = bmu_to_string(units)
            labels = torch.tensor(convert_bmu_labels(bmu_labels))

            # print(labels)
            # print(labels.tolist())
            # input('')
            cov_loss = covariance_loss(representations,labels)
            ch_loss = calinski_harabazs_loss(representations,labels)



            loss = 0.1*cont_loss + 0.9*cov_loss #+ ch_loss  

            # covariance per cluster loss 
            # calinski-harabazs loss 

            loss.backward()
            optimizer.step()

            # print(n)
            # input('this')

            if n%10==0:
                print(f'Training {n}: {loss.item(),cont_loss.item(),cov_loss.item(),ch_loss.item()}  total batches: {np.ceil(mnist_dataset.data.shape[0]/batch_size).astype(int)}')

                print(len(list(set(labels.tolist()))))


    print('Encoder trained.')


    model.eval()
    mnist_dataset = MNISTDataset(csv_file,num_samples_cluster)
    dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)
    total= []
    labels_list = []
    for n,(batch, labels) in enumerate(dataloader):
        embeddings = model.create_representation(tabular_perturbation.trim_constant_columns(pd.DataFrame(batch)))
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


































