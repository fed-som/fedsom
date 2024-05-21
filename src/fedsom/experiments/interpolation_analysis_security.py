from deepclustering.datasets.datasets import *
import argparse
from pathlib import Path 
import joblib
from deepclustering.encoder.tabular_perturbations import TabularPerturbation
from deepclustering.som.som_temp import SelfOrganizingMap
import torch
import os
from collections import Counter 
from scipy.stats import wasserstein_distance as wd
import matplotlib.pyplot as plt



def plot_feature_changes(feature_changes_list, savepath, show=False):

    plt.style.use('ggplot')
    num_features = len(feature_changes_list)
    num_cols = 5  # Number of columns in the array
    num_rows = (num_features + num_cols - 1) // num_cols  # Number of rows in the array

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))

    # for ax in axs.flat:
    #     ax.set_aspect('equal')

    for idx, feature_changes in enumerate(feature_changes_list):
        ax = axs[idx // num_cols, idx % num_cols] if num_rows > 1 else axs[idx % num_cols]
        cluster_first = feature_changes['cluster_1']
        cluster_second = feature_changes['cluster_2']

        ax.hist(cluster_first[feature_changes['most_changed']], alpha=0.65, label=feature_changes['label_1'],color='#00cc99',density=True)
        ax.hist(cluster_second[feature_changes['most_changed']], alpha=0.65, label=feature_changes['label_2'],color='#3300FF',density=True)

        ax.set_xlabel(f"{feature_changes['label_1']} {feature_changes['coords_1']}\n{feature_changes['label_2']} {feature_changes['coords_2']}",fontsize=10,fontweight='bold')
        ax.ticklabel_format(style='sci', axis='x',scilimits=(0,0))
        ax.ticklabel_format(style='sci', axis='y',scilimits=(0,0))
        ax.set_ylabel('Frequency',fontsize=10)
        ax.set_title(f"feature {feature_changes['most_changed']}",fontsize=10)

    plt.tight_layout()


    plt.savefig(savepath)

    if show:
        plt.show(block=False)

    # input('DONE')
    plt.close()




def find_most_changed_feature(cluster_first,cluster_second,categorical_indices):
    max_value = 0
    max_feature = None
    for n,c in enumerate(cluster_first.columns):

        # if n not in categorical_indices:

        #     if len(cluster_first[c].values)>0 and len(cluster_second[c].values)>0:
        #         value = wd(cluster_first[c].values,cluster_second[c].values)
        #     else:
        #         value = 0

        #     if value>max_value:
        #         max_value = value  
        #         max_feature = c 

        if n in categorical_indices:

            if len(cluster_first[c].values)>0 and len(cluster_second[c].values)>0:
                c_first = Counter(cluster_first[c].values)
                c_second = Counter(cluster_second[c].values)
                keys_first = set(c_first.keys())
                keys_second = set(c_second.keys())
                union = keys_first.union(keys_second)
                total_first = sum([count for k,count in c_first.items()])
                total_second = sum([count for k,count in c_second.items()])

                for k in c_first:
                    c_first[k] = c_first[k]/total_first
                for k in c_second:
                    c_second[k] = c_second[k]/total_second

                distance = 0
                for k in union:
                    if k in c_first:
                        value_first = c_first[k]
                    else:
                        value_first = 0 
                    if k in c_second:
                        value_second = c_second[k]
                    else:
                        value_second = 0
                    distance+=np.abs(value_first-value_second)

                if distance > max_value:
                    max_value = distance 
                    max_feature = c


    return max_feature,max_value 



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

    

    data = data_class.load(data_path)
    tabular_perturbation = TabularPerturbation(x_dim=data.shape[1]-1)
    tabular_perturbation.find_categorical_indices(data[[c for c in data.columns if c!='avclass']])
    categorical_indices = tabular_perturbation.get_categorical_indices()

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

    for n,(k,v) in enumerate(interpolations.items()):
        if n%100==0:
            print(f'{dataset}: {n}')

        if True:#n>1000:#True:#(n in [4391,4367,4338,4314,3895,3884,3863,3843,1495,1484,1463,1443]) or (n>4452):

            node_coords = []
            labels = []
            source = v['bounds'][0]
            target = v['bounds'][1]
            source_index,source_label = source[0],source[1]
            target_index,target_label = target[0],target[1]
            # node_coords.append(v['path']['node_bounds'][0])  # commented
            # labels.append(source_label)                      # commented
            for path_vector in v['path']['path']:
                node_coords.append(path_vector[2])
                labels.append(path_vector[1])
            # node_coords.append(v['path']['node_bounds'][1])  # commented
            # labels.append(target_label)                      # commented

            labels_coords = pd.DataFrame(zip(labels,node_coords),columns=['labels','coords'])
            cluster_list = []
            cluster_list_labels = []
            cluster_list_coords = []
            for row in labels_coords.index:
                label,coords = labels_coords.iloc[row]
                idx = [t==coords for t in bmu_table['coords'].values]
                cluster_hashes = bmu_table.index[idx]
                cluster_values = data[[c for c in data.columns if c!='avclass']].loc[cluster_hashes,:]
                cluster_list.append(cluster_values)
                cluster_list_labels.append(label)
                cluster_list_coords.append(coords)

            clusters_labels = list(zip(cluster_list,cluster_list_labels,cluster_list_coords))
            feature_changes_list = []
            if len(node_coords)>1: # >3
                good_state = True
                for i in range(1,len(clusters_labels)):
                    feature_changes = {}

                    cluster_first = clusters_labels[i-1][0]
                    label_first = clusters_labels[i-1][1]
                    coords_first = clusters_labels[i-1][2]

                    cluster_second = clusters_labels[i][0]
                    label_second = clusters_labels[i][1]
                    coords_second = clusters_labels[i][2]

                    most_changed_feature,distance = find_most_changed_feature(cluster_first,cluster_second,categorical_indices)
                    if distance==0:
                        good_state = False

                    feature_changes['most_changed'] = most_changed_feature
                    feature_changes['distance'] = distance
                    feature_changes['cluster_1'] = cluster_first 
                    feature_changes['cluster_2'] = cluster_second
                    feature_changes['label_1'] = label_first 
                    feature_changes['label_2'] = label_second 
                    feature_changes['coords_1'] = coords_first 
                    feature_changes['coords_2'] = coords_second
                    feature_changes_list.append(feature_changes)
            
                if good_state and (cluster_first.shape[0]>1) and (cluster_second.shape[0]>1):
                    # print(n)
                    # print(node_coords)

                    savepath = results_dir / f'interpolations_{n}.png'
                    plot_feature_changes(feature_changes_list, savepath=savepath, show=False)

                # plot_distribution_changes(feature_changes_list,savepath=results_dir / f'interpolations_{n}.png')





































