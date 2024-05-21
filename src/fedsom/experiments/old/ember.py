
from torch.utils.data import Dataset, DataLoader
import numpy as np   
import torch 
import pandas as pd    
from deepclustering.encoder.tabular_perturbations import TabularPerturbation
from deepclustering.encoder.tabular_encoder import TabularNet
from deepclustering.encoder.utils.preprocessing_utils import (
    encode_categoricals,
    train_categorical_encodings,
)
from deepclustering.loss.contrastive_loss import contrastive_loss
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")



class EmberDataset(Dataset):
    def __init__(self, csv_file,num_samples):

        self.data = pd.read_csv(csv_file,index_col='sha256').iloc[:num_samples,:]
        self.features = self.data[[c for c in self.data.columns if c!='avclass']]
        self.labels = self.data['avclass'].tolist()
        del self.data

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features.iloc[idx].values
        label = self.labels[idx]
        return x, label


def scatter_plot(X, labels,annotations, best_params_string, filepath):

    if X.shape[1]>2:
        tsne = TSNE(n_components=2)
        X = tsne.fit_transform(X)

    fig = plt.figure(figsize=(14, 8))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab20")  #'viridis')

    for i, annotation in enumerate(annotations):
        if np.random.rand()<0.1:
            plt.annotate(annotation, (X[i,0], X[i,1]), fontsize=6, textcoords="offset points", xytext=(0,10), ha='center')

    plt.title(best_params_string, fontsize=8, fontweight="bold")
    plt.colorbar(label="Cluster")
    plt.show(block=False)
    input('')
    plt.close()
    del fig




if __name__=='__main__':


    np.random.seed(0)
    torch.manual_seed(0)


    csv_file = '../../../sandbox/data/ember/ember_malicious_vectors_0.csv'
    # csv_file = '../../../sandbox/data/ember/ember.csv'
    csv_file = '../../../sandbox/data/ember/top_k/ember_top_5.csv'
    



    num_samples_dist = 100_00
    num_samples_train = 100_00
    num_samples_cluster = 1000

    x_dim = 2381
    batch_size = 100
    embedding_dim = 3
    hidden_dim = 20
    final_dim = 20
    learning_rate_enc = 0.01
    temp = 0.05
    num_epochs = 4
    sparsity_thresh = 0.95
    approx_constant_thresh = 0.99
    corruption_factor = 0.1
    sample_prior = 0.9
    null_char = 0.0


    use_true_labels = True
    grid_size = (4,4)
    input_size = hidden_dim
    learning_rate_som = 1.5
    sigma = 1e-5
    num_epochs_som = 4



    
    tabular_perturbation = TabularPerturbation(
                                x_dim=x_dim, 
                                sparsity_thresh=sparsity_thresh, 
                                approx_constant_thresh=approx_constant_thresh, 
                                corruption_factor=corruption_factor,
                                sample_prior=sample_prior,
                                null_char=null_char
          )

    big_batch = []
    ember_dataset = EmberDataset(csv_file,num_samples_dist)
    dataloader = DataLoader(ember_dataset, batch_size=batch_size, shuffle=True)
    for n,(batch,labels) in enumerate(dataloader):

        batch = pd.DataFrame(batch)
        big_batch.append(batch)
    big_batch = pd.concat(big_batch)
    tabular_perturbation.find_categorical_indices(big_batch)


    categorical_indices = tabular_perturbation.get_categorical_indices()
    cat_columns = big_batch.iloc[:, categorical_indices]
    encodings = train_categorical_encodings(cat_columns)



    dataloader = DataLoader(ember_dataset, batch_size=batch_size, shuffle=True)
    for n,(batch,labels) in enumerate(dataloader):

        batch = pd.DataFrame(batch)
        tabular_perturbation.update(batch)

        print(f'{n+1} of {np.ceil(ember_dataset.features.shape[0]/batch_size).astype(int)}')
    print('Distributions learned.')



    cat_indices = categorical_indices
    x_dim = batch.shape[1]
    model = TabularNet(x_dim, cat_indices, encodings, embedding_dim, hidden_dim, final_dim)
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate_enc,
    )
    ember_dataset = EmberDataset(csv_file,num_samples_train)
    for epoch in range(num_epochs):

        dataloader = DataLoader(ember_dataset, batch_size=batch_size, shuffle=True)
        for n,(batch,labels) in enumerate(dataloader):

            batch = pd.DataFrame(batch)
            perturbed_batch = tabular_perturbation.perturb_batch(batch)
            embeddings = model(batch) 
            embeddings_perturbed = model(perturbed_batch)

            optimizer.zero_grad()
            loss = contrastive_loss(embeddings,embeddings_perturbed,temp=temp)
            loss.backward()
            optimizer.step()

            if n%10==0:
                print(f'Training {n}: {loss.item()}  total batches: {np.ceil(ember_dataset.features.shape[0]/batch_size).astype(int)}')
    print('Encoder trained.')


    model.eval()
    ember_dataset = EmberDataset(csv_file,num_samples_cluster)
    dataloader = DataLoader(ember_dataset, batch_size=batch_size, shuffle=True)
    total= []
    labels_list = []
    for n,(batch, labels) in enumerate(dataloader):
        embeddings = model.create_representation(pd.DataFrame(batch))
        total.append(embeddings)
        labels_list.append(list(labels))
    embeddings = torch.cat(total,dim=0)

    if use_true_labels:
        labels = [x for L in labels_list for x in L]
    else:
        print('Training SOM')
        som = SelfOrganizingMap(grid_size=grid_size, input_size=input_size, learning_rate=learning_rate_som, sigma=sigma)
        som.train(embeddings.detach(),num_epochs=num_epochs_som)

        bmu_labels = {}
        for n, x in enumerate(embeddings):
            units = som.find_best_matching_unit(x)
            bmu_labels[n] = bmu_to_string(units)
        labels = convert_bmu_labels(bmu_labels)


    codes = {label: n for n,label in enumerate(list(set(labels)))}
    annotations = [codes[label] for label in labels]

    filepath = 'test.png'
    scatter_plot(embeddings.detach().numpy(), annotations, labels, 'hi', filepath) 



