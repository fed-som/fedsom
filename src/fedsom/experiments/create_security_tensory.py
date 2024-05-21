from deepclustering.datasets.datasets import *
import argparse
from pathlib import Path 
import joblib
from deepclustering.encoder.tabular_perturbations import TabularPerturbation
from deepclustering.som.som_temp import SelfOrganizingMap
from deepclustering.utils.graphics_utils import plot_graph
import torch
import os
import networkx as nx    
from collections import Counter 
from scipy.stats import wasserstein_distance as wd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random
plt.style.use('ggplot')




if __name__=='__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str,default='ember')

    args = parser.parse_args()
    dataset = args.dataset 

    results_dir = Path(f'../../../sandbox/results/experiments/final/interpolations_encoder/images/{dataset}/')
    if not os.path.exists(results_dir):
        results_dir.mkdir(parents=True)  


    if dataset=='ember':
        data_class = EmberDatasetInMem
        data_path = Path('../../../sandbox/data/ember/top_k/ember_top_10_test.csv')
        embeddings_path = Path('../../../sandbox/results/experiments/final/embeddings/encoder/SINGLE_ENCODER_SOM_EMBEDDINGS_4_10_24/ember/ember_test_encoder_embeddings.csv')
        embeddings_train_path = Path('../../../sandbox/results/experiments/final/embeddings/encoder/SINGLE_ENCODER_SOM_EMBEDDINGS_4_10_24/ember/ember_train_encoder_embeddings.csv')
        som_path = Path('../../../sandbox/results/experiments/final/soms/encoder/ember/sandbox/results/SOM_SINGLE/checkpoints/ember/som/ember_som.pt')
        interpolation_filepath = Path('../../../sandbox/results/experiments/final/soms/encoder/ember/sandbox/results/SOM_SINGLE/ember/som/ember_som_interpolations.joblib')

    elif dataset=='ccc':
        data_class = CCCSInMem
        data_path = Path('../../../sandbox/data/MalwareDatasets/MalwareDatasets/CCCS-CIC-AndMal2020/total/cccs_cic_andmal2020_test.csv')
        embeddings_path = Path('../../../sandbox/results/experiments/final/embeddings/encoder/SINGLE_ENCODER_SOM_EMBEDDINGS_4_10_24/ccc/ccc_test_encoder_embeddings.csv')
        som_path = Path('../../../sandbox/results/experiments/final/soms/encoder/ccc/sandbox/results/SOM_SINGLE/checkpoints/ccc/som/ccc_som.pt')
        interpolation_filepath = Path('../../../sandbox/results/experiments/final/soms/encoder/ccc/sandbox/results/SOM_SINGLE/ccc/som/ccc_som_interpolations.joblib')

    elif dataset=='sorel':
        data_class = SorelDatasetInMem
        data_path = Path('../../../sandbox/data/sorel/sorel_subset_test.csv')
        embeddings_path = Path('../../../sandbox/results/experiments/final/embeddings/encoder/SINGLE_ENCODER_SOM_EMBEDDINGS_4_10_24/sorel/sorel_test_encoder_embeddings.csv')
        som_path = Path('../../../sandbox/results/experiments/final/soms/encoder/sorel/sandbox/results/SOM_SINGLE/checkpoints/sorel/som/sorel_som.pt')
        interpolation_filepath = Path('../../../sandbox/results/experiments/final/soms/encoder/sorel/sandbox/results/SOM_SINGLE/sorel/som/sorel_som_interpolations.joblib')

    elif dataset=='syscalls':
        data_class = SysCallsInMem
        data_path = Path('../../../sandbox/data/MalwareDatasets/MalwareDatasets/CICMalDroid2020/total/feature_vectors_syscalls_frequency_5_Cat_test.csv')
        embeddings_path = Path('../../../sandbox/results/experiments/final/embeddings/encoder/SINGLE_ENCODER_SOM_EMBEDDINGS_4_10_24/syscalls/syscalls_test_encoder_embeddings.csv')
        som_path = Path('../../../sandbox/results/experiments/final/soms/encoder/syscalls/sandbox/results/SOM_SINGLE/checkpoints/syscalls/som/syscalls_som.pt')
        interpolation_filepath = Path('../../../sandbox/results/experiments/final/soms/encoder/syscalls/sandbox/results/SOM_SINGLE/syscalls/som/syscalls_som_interpolations.joblib')

    elif dataset=='syscallsbinders':
        data_class = SysCallsBindersInMem
        data_path = Path('../../../sandbox/data/MalwareDatasets/MalwareDatasets/CICMalDroid2020/total/feature_vectors_syscallsbinders_frequency_5_Cat_test.csv')
        embeddings_path = Path('../../../sandbox/results/experiments/final/embeddings/encoder/SINGLE_ENCODER_SOM_EMBEDDINGS_4_10_24/syscallsbinders/syscallsbinders_test_encoder_embeddings.csv')
        som_path = Path('../../../sandbox/results/experiments/final/soms/encoder/syscallsbinders/sandbox/results/SOM_SINGLE/checkpoints/syscallsbinders/som/syscallsbinders_som.pt')
        interpolation_filepath = Path('../../../sandbox/results/experiments/final/soms/encoder/syscallsbinders/sandbox/results/SOM_SINGLE/syscallsbinders/som/syscallsbinders_som_interpolations.joblib')

    elif dataset=='malmem':
        data_class = MalMemInMem
        data_path = Path('../../../sandbox/data/MalwareDatasets/MalwareDatasets/MalMem/total/Obfuscated-MalMem2022_test_converted.csv')
        embeddings_path = Path('../../../sandbox/results/experiments/final/embeddings/encoder/SINGLE_ENCODER_SOM_EMBEDDINGS_4_10_24/malmem/malmem_test_encoder_embeddings.csv')
        som_path = Path('../../../sandbox/results/experiments/final/soms/encoder/malmem/sandbox/results/SOM_SINGLE/checkpoints/malmem/som/malmem_som.pt')
        interpolation_filepath = Path('../../../sandbox/results/experiments/final/soms/encoder/malmem/sandbox/results/SOM_SINGLE/malmem/som/malmem_som_interpolations.joblib')


    elif dataset=='pdfmalware':
        data_class = PDFMalwareInMem
        data_path = Path('../../../sandbox/data/MalwareDatasets/MalwareDatasets/PDFMalware/total/pdfmalware_test_converted.csv')
        embeddings_path = Path('../../../sandbox/results/experiments/final/embeddings/encoder/SINGLE_ENCODER_SOM_EMBEDDINGS_4_10_24/pdfmalware/pdfmalware_test_encoder_embeddings.csv')
        som_path = Path('../../../sandbox/results/experiments/final/soms/encoder/pdfmalware/sandbox/results/SOM_SINGLE/checkpoints/pdfmalware/som/pdfmalware_som.pt')
        interpolation_filepath = Path('../../../sandbox/results/experiments/final/soms/encoder/pdfmalware/sandbox/results/SOM_SINGLE/pdfmalware/som/pdfmalware_som_interpolations.joblib')

    else:
        assert False


    embeddings_class = EmbeddingsDataset
    embeddings = embeddings_class.load(embeddings_path)
    embeddings_vectors = embeddings[[c for c in embeddings.columns if c!='label']]
    interpolations = joblib.load(interpolation_filepath)
    som = SelfOrganizingMap.load(som_path)

    bmu_table = []
    for idx in embeddings_vectors.index:
        x = embeddings_vectors.loc[idx].values
        bmu = som.find_best_matching_unit(torch.tensor(x))
        label = embeddings['label'].loc[idx]
        bmu = tuple([i.item() for i in bmu])
        bmu_table.append([bmu,label])

    bmu_table = pd.DataFrame(bmu_table,columns=['coords','label'])
    bmu_table.index = embeddings_vectors.index

    node_labels = {}
    for coord in set(bmu_table.coords):
        result = Counter(bmu_table['label'].loc[bmu_table.coords==coord].values)
        node_labels[coord] = result.most_common(1)[0][0]

    som.add_weights_to_graph(torch.tensor(embeddings_vectors.values))
    for node in som.graph.nodes():
        if node not in node_labels:
            node_labels[node] = ''
    plot_graph(som.graph,node_labels)








































