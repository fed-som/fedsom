
from sklearn.datasets import make_blobs
from deepclustering.som.utils.som_utils import bmu_to_string, convert_bmu_labels
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


    def scatter_plot(X, labels,cmap):
        fig = plt.figure(figsize=(14, 8))
        plt.scatter(X[:, 0], X[:, 1], c=labels,cmap=cmap)#"tab20")  #'viridis')
        plt.title('test', fontsize=12, fontweight="bold")
        plt.colorbar(label="Cluster")
        plt.show(block=False)
        input('')
        plt.close()
        del fig

if __name__=='__main__':

	


    n_samples = 1000
    n_clusters = 20     
    centers = n_clusters
    X, _ = make_blobs(n_samples=n_samples, n_features=10, centers=centers, random_state=42, cluster_std=1.)


    grid_size = (5,5)
    learning_rate = 1.5
    sigma = .01
    num_epochs = 10  
    som = SelfOrganizingMap(grid_size, input_size=X.shape[1], learning_rate=learning_rate, sigma=sigma)
    som.train(torch.tensor(X), num_epochs=num_epochs)
    
    bmu_labels = {}
    for n, x in enumerate(X):
        units = som.find_best_matching_unit(torch.tensor(x))
        bmu_labels[n] = bmu_to_string(units)

    labels = convert_bmu_labels(bmu_labels)



    import numpy as np
    from sklearn.manifold import TSNE

 
    tsne = TSNE(n_components=2)  # Reduce to 2 dimensions
    X_reduced = tsne.fit_transform(X)



    # import umap
    # umap_model = umap.UMAP(n_components=2)  # Reduce to 2 dimensions
    # X_reduced = umap_model.fit_transform(X)


    scatter_plot(X_reduced, labels,"tab20")
