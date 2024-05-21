import numpy as np   
from deepclustering.encoder.tabular_perturbations import TabularPerturbation,DataTrimmer
import joblib 
import umap 
import scipy.sparse as sp 
from deepclustering.encoder.utils.data_utils import CategoricalOneHotEncoder   
from deepclustering.utils.graphics_utils import scatter_plot
from deepclustering.encoder.utils.preprocessing_utils import (
    encode_categoricals,
    train_categorical_encodings,
    parse_columns,
    replace_nans,
    replace_nans_np_array,
    train_power_transformer,
)
from deepclustering.datasets.datasets import *
import os 
import argparse
import yaml    
from pathlib import Path      


class Checkpoint(object):
    def __init__(self,data_dir,clear_old=True):

        self.data_dir = data_dir
        self.power_transformer_name = 'power_transformer.joblib'
        self.tabular_perturbation_name = 'tabular_perturbation.pt'
        self.categorical_onehot_encoder_name = 'categorical_onehot_encoder.joblib'
        self.umap_names = ['umap_mapper_categorical','umap_mapper_numerical','umap_mapper_combo']
        self.data_trimmer_name = 'data_trimmer.pt'  
        if clear_old:
            self.delete_file(data_dir / self.data_trimmer_name)
            self.delete_file(data_dir / self.categorical_onehot_encoder_name)
            for umap_name in self.umap_names:
                self.delete_file(data_dir / f'{umap_name}.joblib')

    def save(self,umap_mapper_dict,data_trimmer,tabular_perturbation,categorical_onehot_encoder,power_transformer):

        self.save_umap(umap_mapper_dict)
        joblib.dump(power_transformer,self.data_dir / self.power_transformer_name)
        data_trimmer.save(self.data_dir / self.data_trimmer_name)
        tabular_perturbation.save(self.data_dir / self.tabular_perturbation_name)
        categorical_onehot_encoder.save(self.data_dir / self.categorical_onehot_encoder_name)
 
    @staticmethod
    def delete_file(filepath):

        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError as e:
                print(f"Error: {e} - {filepath}")
        else:
            print(f"The file {filepath} does not exist.")


    def save_umap(self,umap_mapper_dict):
        for umap_name in umap_mapper_dict.keys():
            if umap_mapper_dict[umap_name]: 
                joblib.dump(umap_mapper_dict[umap_name],self.data_dir / f'{umap_name}.joblib')

    def seed(self):
        self.overwrite_best_score(0)

    def load_power_transformer(self):
        return joblib.load(self.data_dir / self.power_transformer_name)

    def load_data_trimmer(self):
        return DataTrimmer.load(self.data_dir / self.data_trimmer_name)

    def load_categorical_onehot_encoder(self):
        return CategoricalOneHotEncoder.load(self.data_dir / self.categorical_onehot_encoder_name)

    def load_perturbation(self):
        return TabularPerturbation.load(self.data_dir / self.tabular_perturbation_name)

    def load_hdbscan(self):
        return joblib.load(self.data_dir / self.hdbscan_name)

    def load_mappers(self):
        umap_mapper_dict = {}
        for umap_name in self.umap_names:
            filepath = self.data_dir / f'{umap_name}.joblib'
            if os.path.exists(filepath):
                umap_mapper_dict[umap_name] = joblib.load(filepath)
        return umap_mapper_dict  



def data_to_sparse(data):

    if isinstance(data,pd.DataFrame):
        data = data.values

    num_samples, num_features = data.shape   
    sparse_data = sp.lil_matrix((num_samples,num_features), dtype=np.float32)
    for row_idx in range(num_samples):

        nonzero_indices = data[row_idx]!=0
        sparse_data[row_idx,nonzero_indices] = data[row_idx,nonzero_indices]

    return sparse_data


def train_umap_encodings(data,config,checkpoint):

    vectors = data['vectors']
    labels = data['label']
    
    umap_params_numerical = config.umap.numerical
    umap_params_categorical = config.umap.categorical
    umap_params_combo = config.umap.combo

    approx_constant_thresh = config.encoder_params.approx_constant_thresh
    data_trimmer = DataTrimmer(approx_constant_thresh=approx_constant_thresh)
    data_trimmer.learn_sparsities(vectors)

    train_trimmed = data_trimmer.trim_constant_columns(vectors)
    tabular_perturbation = TabularPerturbation(train_trimmed.shape[1])
    tabular_perturbation.find_categorical_indices(train_trimmed)
    categorical_indices = tabular_perturbation.get_categorical_indices()
    categorical,numerical = parse_columns(train_trimmed,categorical_indices)

    categorical_onehot_encoder = CategoricalOneHotEncoder()
    if not categorical.empty:
        categorical_onehot_encoder.record_high_entropy_cols(categorical)
        categorical = categorical_onehot_encoder.fit_transform(categorical)

    umap_mapper_categorical = None   
    umap_mapper_numerical = None   
    umap_mapper_combo = None 
    power_transformer = None 
    if not categorical.empty:
        sparse_categorical = data_to_sparse(categorical)
        umap_params_categorical = config_to_dict(config.umap.categorical)
        umap_params_categorical['n_components'] = min(umap_params_categorical['n_components'],sparse_categorical.shape[1])
        umap_mapper_categorical = umap.UMAP(**umap_params_categorical,random_state=None, low_memory=True).fit(sparse_categorical)

    if not numerical.empty:
        numerical = replace_nans(numerical)
        if not categorical.empty:
            power_transformer = train_power_transformer(numerical)
            numerical = pd.DataFrame(power_transformer.transform(numerical),columns=numerical.columns,index=numerical.index)

        sparse_numerical = data_to_sparse(numerical)
        umap_params_numerical = config_to_dict(config.umap.numerical)
        umap_params_numerical['n_components'] = min(umap_params_numerical['n_components'],sparse_numerical.shape[1])
        umap_mapper_numerical = umap.UMAP(**umap_params_numerical,random_state=None, low_memory=True).fit(sparse_numerical)

    if (not categorical.empty) and (not numerical.empty):
        joined = np.concatenate((umap_mapper_numerical.embedding_,umap_mapper_categorical.embedding_),axis=1)
        umap_params_combo = config_to_dict(config.umap.combo)
        umap_params_combo['n_components'] = min(umap_params_combo['n_components'],umap_params_categorical['n_components']+umap_params_numerical['n_components'])
        umap_mapper_combo = umap.UMAP(**umap_params_combo,random_state=None,low_memory=True).fit(joined)

    if (not categorical.empty) and numerical.empty:
        embedding = umap_mapper_categorical.embedding_   
    elif categorical.empty and (not numerical.empty):
        embedding = umap_mapper_numerical.embedding_     
    else:
        embedding = umap_mapper_combo.embedding_

    umap_mapper_dict = {}
    umap_mapper_dict['umap_mapper_categorical'] = umap_mapper_categorical
    umap_mapper_dict['umap_mapper_numerical'] = umap_mapper_numerical
    umap_mapper_dict['umap_mapper_combo'] = umap_mapper_combo
             
    checkpoint.save(umap_mapper_dict,data_trimmer,tabular_perturbation,categorical_onehot_encoder,power_transformer)


def encode(data,checkpoint):

    np.random.seed(0)
    data_trimmer = checkpoint.load_data_trimmer()
    umap_mapper_dict = checkpoint.load_mappers()
    tabular_perturbation = checkpoint.load_perturbation()
    categorical_onehot_encoder = checkpoint.load_categorical_onehot_encoder()
    power_transformer = checkpoint.load_power_transformer()

    vectors = data['vectors']
    labels_true = data['label']
    embeddings = embed_via_umap_dict(vectors,umap_mapper_dict,data_trimmer,tabular_perturbation,categorical_onehot_encoder,power_transformer)

    embeddings_df = pd.DataFrame(embeddings)       
    embeddings_df['label'] = labels_true
    embeddings_df.index = data['index']
    embeddings_df.index.name = 'Index'

    return embeddings_df    



def load_yaml_config(fpath):

    if 's3' in fpath:
        with sm_open(fpath) as file_handle:
            config = yaml.load(file_handle, Loader=yaml.FullLoader)
    else:
        with open(fpath, "r") as f:
            config = yaml.full_load(f)
    return ConfigObject(config)


def _get_args():
    parser = argparse.ArgumentParser(description="deepclustering embedding creation")
    parser.add_argument("config_path",nargs="?",type=str,default=os.getenv("config_path"))
    return parser.parse_args()


class ConfigObject:
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigObject(value))
            else:
                setattr(self, key, value)


def config_to_dict(config_obj):
    if isinstance(config_obj, ConfigObject):
        return {key: config_to_dict(value) for key, value in vars(config_obj).items()}
    else:
        return config_obj


def embed_via_umap_dict(vectors,umap_mapper_dict,data_trimmer,tabular_perturbation,categorical_onehot_encoder,power_transformer):

    trimmed = data_trimmer.trim_constant_columns(vectors)
    categorical_indices = tabular_perturbation.get_categorical_indices()
    categorical,numerical = parse_columns(trimmed,categorical_indices)

    if not categorical.empty:
        categorical = categorical_onehot_encoder.transform(categorical)
    if not categorical.empty:
        sparse_categorical = data_to_sparse(categorical)
        cat_embedding = umap_mapper_dict['umap_mapper_categorical'].transform(sparse_categorical)
        cat_embedding = replace_nans_np_array(cat_embedding)
    if not numerical.empty:
        numerical = replace_nans(numerical)
        if power_transformer:
            numerical = pd.DataFrame(power_transformer.transform(numerical),columns=numerical.columns,index=numerical.index)
        sparse_numerical = data_to_sparse(numerical)
        num_embedding = umap_mapper_dict['umap_mapper_numerical'].transform(sparse_numerical)
        num_embedding = replace_nans_np_array(num_embedding)
    if (not categorical.empty) and (not numerical.empty):
        embedding = umap_mapper_dict['umap_mapper_combo'].transform(np.concatenate((cat_embedding,num_embedding),axis=1))
    elif (not categorical.empty) and (numerical.empty):
        embedding = cat_embedding   
    elif (categorical.empty) and (not numerical.empty):
        embedding = num_embedding 

    return embedding 


if __name__=='__main__':

    # python optuna_encoder_som.py --dataset mnist --n_trials 3

    np.random.seed(0)
    args = _get_args()
    config = load_yaml_config(args.config_path)
    dataset = config.dataset

    results_dir = Path(f'../../../sandbox/results/UMAP_SINGLE/{dataset}/')
    if not os.path.exists(results_dir):
        results_dir.mkdir(parents=True)  

    checkpoint_dir = Path(f'../../../sandbox/results/UMAP_SINGLE/checkpoint/{dataset}/')
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    data_class = eval(config.data_class)
    train_data_path = config.paths.train_data_path
    test_data_path = config.paths.test_data_path 
    if dataset=='malmem':
        data_class_object = data_class(class_type='subavclass')
        train_vectors,train_labels,train_index = data_class_object.load_single(train_data_path)
        test_vectors,test_labels,test_index = data_class_object.load_single(test_data_path)
    else:
        train_vectors,train_labels,train_index = data_class.load_single(train_data_path)
        test_vectors,test_labels,test_index = data_class.load_single(test_data_path)

    train_vectors = train_vectors[:config.num_samples_train,:]
    train_labels = train_labels[:config.num_samples_train]
    train_index = train_index[:config.num_samples_train]
    test_vectors = test_vectors[:config.num_samples_test,:]
    test_labels = test_labels[:config.num_samples_test]
    test_index = test_index[:config.num_samples_test]
    train_data = {}
    train_data['vectors'] = train_vectors
    train_data['label'] = train_labels
    train_data['index'] = train_index 
    test_data = {}
    test_data['vectors'] = test_vectors
    test_data['label'] = test_labels 
    test_data['index'] = test_index
    checkpoint = Checkpoint(checkpoint_dir)

    train_umap_encodings(train_data,config.training_params,checkpoint)
    train_embeddings = encode(train_data,checkpoint)
    train_embeddings.to_csv(config.paths.train_embeddings_path,index='Index')
    test_embeddings = encode(test_data,checkpoint)
    test_embeddings.to_csv(config.paths.test_embeddings_path,index='Index')

    scatter_path = results_dir / f"{dataset}__scatter_true_labels.png"
    X = test_embeddings[[c for c in test_embeddings.columns if c!='label']].values
    labels = test_embeddings.label.values
    scatter_plot(X, labels, dataset, scatter_path)



















    