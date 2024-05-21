import numpy as np     
import scipy.sparse as sp  
import pandas as pd  
import umap      
import hdbscan
from multiprocessing import Manager, Pool
from multiprocessing.managers import DictProxy
import matplotlib.pyplot as plt    
import joblib   
import sklearn
import matplotlib.style as style
style.use("ggplot") 





def sample_hyperparameters(randomsearch_dict, N: int):
    """construct list of hyperparameter choices by random sampling"""

    np.random.seed(0)
    kwargs_samples = []
    for _ in range(N):
        kwargs_sample = {}
        for key, v in randomsearch_dict.items():
            param, type_ = key.split("__")
            if type_ == "cat":
                value = np.random.choice(v)
            elif type_ == "int":
                value = np.random.randint(*v)
            elif type_ == "uniform":
                value = np.random.uniform(*v)
            elif type_ == "log":
                value = float(np.power(10.0, np.random.uniform(*np.log10(v))))
            elif type_ == "const":
                value = v
            else:
                raise ValueError("type_ must be one of ['cat','int',uniform','log']")
            kwargs_sample[param] = value
        kwargs_samples.append(kwargs_sample)
    return kwargs_samples





def mnist_to_sparse(data):

    data = (1./255)*data[[c for c in data.columns if c!='class']].values.astype('float32')
    num_samples, num_features = data.shape   
    sparse_mnist_data = sp.lil_matrix((num_samples,num_features), dtype=np.float32)
    for row_idx in range(num_samples):

        nonzero_indices = data[row_idx]!=0
        sparse_mnist_data[row_idx,nonzero_indices] = data[row_idx,nonzero_indices]

    return sparse_mnist_data


def clustering_caller(results,hdbscan_param,embeddings,n):
        model = hdbscan.HDBSCAN(**hdbscan_param,core_dist_n_jobs=1,prediction_data=False)
        model.fit(embeddings)
        score = model.relative_validity_
        # score = sklearn.metrics.calinski_harabasz_score(embeddings,model.labels_)
        print(n,score)
        results[n] = score   


if __name__=='__main__':


    # check metric 
    # then check cosine first euclidean second 
    # experiment with n_components (try 2 first, then 10 then 2)


    filepath = '../../../sandbox/data/mnist/mnist.csv'

    embeddings_filepath = '../../../sandbox/data/embeddings/embeddings'
    mapper_filepath = '../../../sandbox/data/embeddings/mapper.joblib'

    data = pd.read_csv(filepath)
    data = data.sample(frac=1,random_state=42)


    idx = np.ceil(data.shape[0]*0.8).astype(int)
    train = data.iloc[:idx,:]
    test = data.iloc[idx:,:]

    train = data   
    test = data    

    sparse_mnist_data = mnist_to_sparse(train)





    mapper = umap.UMAP(metric='cosine', random_state=42, low_memory=True, n_components=10).fit(sparse_mnist_data)
    # joblib.dump(mapper,mapper_filepath)
    # np.save(embeddings_filepath,mapper.embedding_)


    second_mapper = umap.UMAP(metric='cosine', random_state=42, low_memory=True, n_components=2).fit(mapper.embedding_)



    sparse_mnist_data_test = mnist_to_sparse(test)
    new_data_embedding = mapper.transform(sparse_mnist_data_test)
    new_data_embedding = second_mapper.transform(new_data_embedding)

    labels = test['class'].values



    hdbscan_param_dict = {
    "min_cluster_size__int": [2,100],
    "min_samples__int": [2,100],
    "gen_min_span_tree__const": True,  # must be set to True to compute validity_score
    "cluster_selection_method__cat": ["eom","leaf"],
    "cluster_selection_epsilon__uniform": [0,1],
    "algorithm__cat": ["best","prims_kdtree","prims_balltree","boruvka_kdtree"],
    "leaf_size__int": [5,100]
        }


    np.random.seed(0)
    hdbscan_params = sample_hyperparameters(hdbscan_param_dict,1000)
    results = Manager().dict()
    with Pool() as p:
        p.starmap(
            clustering_caller,
            [[results,hdbscan_param,new_data_embedding,n] for n,hdbscan_param in enumerate(hdbscan_params)],
        )
    p.close()
    p.join()

    best_index = 0
    best_score = 0
    for key in results:
        score = results[key]
        if score>best_score:
            best_index = key
            best_score = score

    best_params = hdbscan_params[best_index]
    best_model = hdbscan.HDBSCAN(**best_params,prediction_data=True)
    # best_model.fit(second_mapper.embedding_)
    best_model.fit(new_data_embedding)


    print(f'Best: {best_score}')
    print(best_params)
    # new_labels, _ = hdbscan.approximate_predict(best_model, new_data_embedding)
    new_labels = best_model.labels_

    print(set(new_labels))


    X = new_data_embedding

    # if X.shape[1]>2:
    #     umap_model = umap.UMAP(n_components=2,random_state=42)
    #     X = umap_model.fit_transform(X)

 


    fig = plt.figure(figsize=(14, 8))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab20")  #'viridis')

    for i, label in enumerate(labels):
        if np.random.rand()<0.1:
            plt.annotate(label, (X[i,0], X[i,1]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.colorbar(label="Cluster")
    plt.show(block=False)
    input('')
    plt.close()










    fig = plt.figure(figsize=(14, 8))
    plt.scatter(X[:, 0], X[:, 1], c=new_labels, cmap="tab20")  #'viridis')

    for i, label in enumerate(new_labels):
        if np.random.rand()<0.1:
            plt.annotate(label, (X[i,0], X[i,1]), textcoords="offset points", xytext=(0,10), ha='center')



    plt.colorbar(label="Cluster")
    plt.show(block=False)
    input('')
    plt.close()









