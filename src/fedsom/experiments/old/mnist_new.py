



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from deepclustering.encoder.tabular_perturbations_new import TabularPerturbation
from deepclustering.encoder.tabular_encoder import TabularNet
from deepclustering.loss.contrastive_loss import contrastive_loss
from deepclustering.som.distributed_som import DistributedSelfOrganizingMap
from deepclustering.som.som import SelfOrganizingMap     
from deepclustering.som.utils.som_utils import bmu_to_string, convert_bmu_labels
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from deepclustering.loss.vicreg import VICReg
import time    
import numpy as np    
from deepclustering.loss.contrastive_loss import ContrastiveAttention
from deepclustering.encoder.custom_metric import Categorical,Metric 
from deepclustering.datasets.datasets import MNISTDataset
from deepclustering.encoder.utils.data_utils import recast_columns


# class VICReg(nn.Module):
#     def __init__(self):
#         super(VICReg, self).__init__()

#         pass

#     def forward(self, x, y):

#         std_x = torch.sqrt(x.var(dim=0) + 0.0001)
#         std_y = torch.sqrt(y.var(dim=0) + 0.0001)

#         x = (x - x.mean(dim=0))/std_x
#         y = (y - y.mean(dim=0))/std_y

#         cov_x = (x.T @ x) / (x.shape[0] - 1)
#         cov_y = (y.T @ y) / (y.shape[0] - 1)
#         cov_loss = off_diagonal(cov_x).pow_(2).sum().div(x.shape[1]
#         ) + off_diagonal(cov_y).pow_(2).sum().div(y.shape[1])


#         return cov_loss





# # Custom Dataset class for MNIST
# class MNISTDataset(Dataset):
#     def __init__(self, csv_file,num_samples):
#         self.data = pd.read_csv(csv_file).loc[:num_samples,:]
#         self.labels = self.data['class']
#         self.features = (1./255)*self.data[[c for c in self.data.columns if c!='class']].values.astype('float32')

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         image = self.features[idx]
#         label = self.labels[idx]
#         return image, label



# class MNISTDataset(Dataset):
#     def __init__(self,data,num_samples,split,train_proportion=0.8):

#         self.features = (1./255)*data[[c for c in data.columns if c!='class']].values.astype('float32')
#         self.labels = data['class'].values

#         self.train_proportion = train_proportion
#         self.partition = np.floor(self.train_proportion*data.shape[0]).astype(int)
#         self.split = split
#         self.num_samples = num_samples
           
#         if self.split=='distribution':

#             self.features = self.features[:self.num_samples,:]
#             self.labels = self.labels[:self.num_samples]

#         elif self.split=='train':

#             assert self.num_samples<self.partition
#             self.features = self.features[:self.num_samples,:]
#             self.labels = self.labels[:self.num_samples]

#         elif self.split=='test':

#             assert self.num_samples<(len(self.labels)-self.partition)
#             self.features = self.features[self.partition:self.partition+self.num_samples,:]
#             self.labels = self.labels[self.partition:self.partition+self.num_samples]


#             print(self.features)
#             print(self.labels)

#     def __len__(self):
#         return self.num_samples  

#     def __getitem__(self,idx):
#         return self.features[idx],self.labels[idx]



            





def scatter_plot(X, labels, best_params_string, filepath):

    if X.shape[1]>2:
        tsne = TSNE(n_components=2)  # Reduce to 2 dimensions
        X = tsne.fit_transform(X)

    fig = plt.figure(figsize=(14, 8))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab20")  #'viridis')

    for i, label in enumerate(labels):
        if np.random.rand()<0.1:
            plt.annotate(label.item(), (X[i,0], X[i,1]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.title(best_params_string, fontsize=8, fontweight="bold")
    plt.colorbar(label="Cluster")
    plt.show(block=False)
    input('')
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


    num_samples_dist = 20_00
    num_samples_train = 10_00
    num_samples_cluster = 10_00
    x_dim = 784




    batch_size = 1024
    embedding_dim = 3
    hidden_dim = 294
    final_dim = 14
    learning_rate_enc = 0.0016
    temp = 1.29
    num_epochs = 5
    coarseness = 100

    sparsity_thresh = 0.89
    approx_constant_thresh = 0.98
    corruption_factor = 0.14
    sample_prior = 0.62
    null_char = None




    use_true_labels = True
    show_perturbed_digits = False

    grid_size = (4,4)
    input_size = hidden_dim
    learning_rate_som = 1.5
    sigma = 1e-5
    num_epochs_som = 4


    mnist_data = pd.read_csv(csv_file)



    vic_reg = VICReg()
    

    tabular_perturbation = TabularPerturbation(
                                x_dim=x_dim, 
                                sparsity_thresh=sparsity_thresh, 
                                approx_constant_thresh=approx_constant_thresh, 
                                corruption_factor=corruption_factor,
                                sample_prior=sample_prior,
                                null_char=null_char
          )

    big_batch = []
    # mnist_dataset = MNISTDataset(mnist_data,num_samples_dist)
    mnist_dataset = MNISTDataset(mnist_data,num_samples_dist,'distribution')
    dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=False)
    for n,(batch,labels) in enumerate(dataloader):

        batch = pd.DataFrame(batch)
        big_batch.append(batch)
    big_batch = pd.concat(big_batch)
    tabular_perturbation.find_categorical_indices(big_batch)


    mnist_dataset = MNISTDataset(mnist_data,num_samples_dist,'distribution')
    dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=False)
    for n,(batch,labels) in enumerate(dataloader):

        batch = pd.DataFrame(batch)
        tabular_perturbation.update(batch)

        print(f'{n+1} of {np.ceil(mnist_dataset.features.shape[0]/batch_size).astype(int)}')




    cat_bool_index = tabular_perturbation.get_cat_bool_index()
    metric = Metric(cat_bool_index=cat_bool_index)
    categorical = Categorical(~cat_bool_index,coarseness=coarseness,min_max=False)
    # categorical = Categorical(big_batch,cat_bool_index=cat_bool_index,courseness=courseness)
    contrastive_attention = ContrastiveAttention(categorical,metric)



    print('Distributions learned.')


    cat_indices = []
    encodings = {}
    batch_size = 1024
    x_dim = 784#tabular_perturbation.get_data_dim()
    model = TabularNet(x_dim, cat_indices, encodings, embedding_dim, hidden_dim, final_dim)
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate_enc,
    )
    # mnist_dataset = MNISTDataset(csv_file,num_samples_train)
    mnist_dataset = MNISTDataset(mnist_data,num_samples_train,'train')
    for epoch in range(num_epochs):

        dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=False)
        for n,(batch,labels) in enumerate(dataloader):

            batch = pd.DataFrame(batch)

            if show_perturbed_digits:
                perturbed_batch = tabular_perturbation.perturb_batch(batch)
                vector1 = batch.values[0,:]
                vector2 = perturbed_batch.values[0,:]
                plot_mnist_digits_side_by_side(vector1, vector2)


            perturbed_batch = tabular_perturbation.perturb_batch(batch)
            # batch = tabular_perturbation.trim_constant_columns(batch)
            # perturbed_batch = tabular_perturbation.trim_constant_columns(perturbed_batch)

            embeddings = model(batch) 
            embeddings_perturbed = model(perturbed_batch)
            optimizer.zero_grad()


            t = time.time()
            loss = contrastive_loss(embeddings,embeddings_perturbed,temp=temp)
            # loss = contrastive_attention(embeddings,embeddings_perturbed,batch,perturbed_batch,temp)

            print(time.time()-t)
            # covariance per cluster loss 
            # calinski-harabazs loss 


            # cov_loss = vic_reg(embeddings,embeddings_perturbed)

            # loss = cont_loss #+ cov_loss

            loss.backward()
            optimizer.step()

            if n%1==0:
                pass
                # print(f'Training {n}: {loss.item()}  total batches: {np.ceil(mnist_dataset.data.shape[0]/batch_size).astype(int)}')
                # input('')


    print('Encoder trained.')


    model.eval()
    mnist_dataset = MNISTDataset(mnist_data,num_samples_cluster,'test')
    # mnist_dataset = MNISTDataset(csv_file,num_samples_cluster)
    dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=False)
    total= []
    labels_list = []
    for n,(batch, labels) in enumerate(dataloader):
        embeddings = model.create_representation(pd.DataFrame(batch))
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


































