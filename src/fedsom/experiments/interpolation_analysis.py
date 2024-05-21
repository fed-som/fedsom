

from deepclustering.datasets.datasets import *
from deepclustering.utils.graphics_utils import * #image_array_to_image,tensor_to_image_array,vector_to_square_tensor
import joblib 
import os 
import argparse
from pathlib import Path 



# def fetch_vt_info(sha256):
#     requests library
# https://docs.virustotal.com/reference/file-info
# https://csvthashinfo-s001-main.barb.beta.eyrie.cloud/api/api-docs/
# curl --request GET \
#   --url https://www.virustotal.com/api/v3/files/{id} \
#   --header 'x-apikey: <your API key>'



if __name__=='__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str,default='mnist')

    args = parser.parse_args()
    dataset = args.dataset 

    interp_dir = Path('../../../sandbox/results/experiments/final/interpolations/UMAP_INTERPOLATIONS_FROM_S3/')
    results_dir = Path(f'../../../sandbox/results/experiments/final/interpolations_umap/images/{dataset}/')
    if not os.path.exists(results_dir):
        results_dir.mkdir(parents=True)  

    if dataset=='mnist':
        data_path = Path('../../../sandbox/data/mnist_data/mnist_test.csv')
        interpolation_filepath = interp_dir / 'mnist/sandbox/results/SOM_SINGLE/mnist/som/mnist_som_interpolations.joblib'
        data_class = MNISTDatasetInMem

    if dataset=='fashionmnist':
        data_path = Path('../../../sandbox/data/fashion_mnist/fashion_mnist_test.csv')
        interpolation_filepath = interp_dir / 'fashionmnist/sandbox/results/SOM_SINGLE/fashionmnist/som/fashionmnist_som_interpolations.joblib'
        data_class = FashionMNISTDatasetInMem

    if dataset=='chars74k':
        data_path = Path('../../../sandbox/data/chars74k/chars74k_coarse_vectors_normalized_test.csv')
        interpolation_filepath = interp_dir / 'chars74k/sandbox/results/SOM_SINGLE/chars74k/som/chars74k_som_interpolations.joblib'
        data_class = Chars74kDatasetInMem

    if dataset=='cifar10':
        data_path = Path('../../../sandbox/data/cifar10/cifar10_test.csv')
        interpolation_filepath = interp_dir / 'cifar10/sandbox/results/SOM_SINGLE/cifar10/som/cifar10_som_interpolations.joblib'
        data_class = CIFAR10DatasetInMem

    if dataset=='emnist':
        data_path = Path('../../../sandbox/data/emnist_data/emnist_normalized_test.csv')
        interpolation_filepath = interp_dir / 'emnist/sandbox/results/SOM_SINGLE/emnist/som/emnist_som_interpolations.joblib'
        data_class = EMNISTInMem

    if dataset=='kuz':
        data_path = Path('../../../sandbox/data/kuzushiji_data/kuzushiji_mnist_normalized_test.csv')
        interpolation_filepath = interp_dir / 'kuz/sandbox/results/SOM_SINGLE/kuz/som/kuz_som_interpolations.joblib'
        data_class = KuzushijiInMem

    if dataset=='notmnist':
        data_path = Path('../../../sandbox/data/notmnist/not_mnist_test.csv')
        interpolation_filepath = interp_dir / 'notmnist/sandbox/results/SOM_SINGLE/notmnist/som/notmnist_som_interpolations.joblib'
        data_class = NotMNISTDatasetInMem

    if dataset=='quickdraw':
        data_path = Path('../../../sandbox/data/quickdraw/quickdraw_test.csv')
        interpolation_filepath = interp_dir / 'quickdraw/sandbox/results/SOM_SINGLE/quickdraw/som/quickdraw_som_interpolations.joblib'
        data_class = QuickdrawInMem

    if dataset=='slmnist':
        data_path = Path('../../../sandbox/data/slmnist/sign_mnist_test.csv')
        interpolation_filepath = interp_dir / 'slmnist/sandbox/results/SOM_SINGLE/slmnist/som/slmnist_som_interpolations.joblib'
        data_class = SLMNISTInMem

    data = data_class.load(data_path)
    print(f"{dataset}: {len(set(data['label']))} =====================================================")
    data = data[[c for c in data.columns if c!='label']]
    interpolations = joblib.load(interpolation_filepath)
    for n,(k,v) in enumerate(interpolations.items()):

        images = []
        node_coords = []
        labels = []
        source = v['bounds'][0]
        target = v['bounds'][1]
        source_index,source_label = source[0],source[1]
        target_index,target_label = target[0],target[1]

        images.append(data.iloc[source_index,:].values)
        node_coords.append(v['path']['node_bounds'][0])
        labels.append(source_label)
        for path_vector in v['path']['path']:
            path_vector_index = path_vector[0]
            path_image = data.iloc[path_vector_index,:].values
            images.append(path_image)
            node_coords.append(path_vector[2])
            labels.append(path_vector[1])
        images.append(data.iloc[target_index,:].values)
        node_coords.append(v['path']['node_bounds'][1])
        labels.append(target_label)

        if True:#n<10:
            rows = 1+len(images)//10
            plot_images(images,rows,10,node_coords,labels,dataset,savepath=results_dir / f'interpolations_{n}.png')








