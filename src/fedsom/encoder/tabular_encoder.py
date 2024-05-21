import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing
from torch.nn import ModuleDict

from deepclustering.encoder.utils.generate_data import generate_mixed_dataset
from deepclustering.encoder.utils.preprocessing_utils import (
    LabelEncoder,
    encode_categoricals,
    get_categorical_indices,
    get_encoding_vocab_lengths,
    get_sample_rows,
    parse_columns,
    train_categorical_encodings,
)
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Initializing zero-element tensors is a no-op")




class TabularNet(nn.Module):
    def __init__(
        self, x_dim: int, cat_indices: List[int], encodings: Dict[int, LabelEncoder], embedding_dim: int, representation_dim: int, final_dim: int, device=None
    ):
        super().__init__()

        self.x_dim = x_dim
        self.embedding_dim = embedding_dim
        self.representation_dim = representation_dim   
        self.final_dim = final_dim
        self.cat_indices = cat_indices
        self.cont_indices = list(set(range(x_dim)) - set(cat_indices))
        self.num_cat_columns = len(cat_indices)
        self.encodings = encodings
        self.device = device if device else 'cpu'

        self.continuous_1 = nn.Linear(x_dim - self.num_cat_columns, representation_dim)
        self.continuous_2 = nn.Linear(x_dim - self.num_cat_columns, representation_dim)
        self.continuous_3 = nn.Linear(representation_dim,representation_dim)

        self.categorical_1 = nn.Linear(embedding_dim * self.num_cat_columns, representation_dim)
        self.categorical_2 = nn.Linear(embedding_dim * self.num_cat_columns, representation_dim)
        self.categorical_3 = nn.Linear(representation_dim,representation_dim)

        self.bn_cont = nn.BatchNorm1d(len(self.cont_indices))
        self.bn_cat = nn.BatchNorm1d(self.embedding_dim*self.num_cat_columns)

        if len(self.cat_indices)==0 or len(self.cat_indices)==x_dim:
            self.joined_linear = nn.Linear(representation_dim,representation_dim)
        else:
            self.joined_linear = nn.Linear(2*representation_dim,representation_dim)

        self.final = nn.Linear(representation_dim,final_dim)
        self.layer_norm_cat = nn.LayerNorm(representation_dim)
        self.layer_norm_cont = nn.LayerNorm(representation_dim)

        self.embeddings = ModuleDict()
        for idx, encoder in self.encodings.items():
            self.embeddings[str(idx)] = nn.Embedding(len(encoder), embedding_dim)


    def create_representation(self,x):

        if x.isna().any().any():
            for c in x.columns:
                if str(x[c].dtype)=='category':
                    x[c].fillna(x[c].mode()[0],inplace=True)
                else:
                    x[c].fillna(x[c].median(),inplace=True)
  
        cat, numerical = parse_columns(x, self.cat_indices)
        if len(self.cat_indices)>0:

            cat_encoded = encode_categoricals(cat, self.encodings)
            x_cat = []
            for idx in range(cat_encoded.shape[1]):
                x_cat.append(self.embeddings[str(idx)](torch.tensor(cat_encoded[:, idx], dtype=torch.long).to(self.device)))
            x_cat = torch.cat(x_cat, 1)
            x_cat_1 = self.categorical_1(x_cat)
            x_cat_2 = self.categorical_2(x_cat)
            x_cat_2 = F.sigmoid(x_cat_2) # change to softmax? F.softmax(x_cat_2)
            x_cat_3 = x_cat_1 * x_cat_2 
            x_cat_4 = self.categorical_3(x_cat_3)
            x_cat = self.layer_norm_cat(x_cat_4)

        else:
            x_cat = torch.tensor([],dtype=torch.float32).to(self.device)

        if numerical.shape[1]>0:

            x_cont = torch.tensor(numerical.values, dtype=torch.float32).to(self.device)
            x_cont = self.bn_cont(x_cont)
            x_cont_1 = self.continuous_1(x_cont)  # linear 1
            x_cont_2 = self.continuous_2(x_cont)  # linear 2
            x_cont_2 = F.sigmoid(x_cont_2) # linear 2 + F.softmax(x_cont_2)
            x_cont_3 = x_cont_1 * x_cont_2
            x_cont_4 = self.continuous_3(x_cont_3) # linear 3
            x_cont = self.layer_norm_cont(x_cont_4) # linear 3 + layer_norm

        else:
            x_cont = torch.tensor([],dtype=torch.float32).to(self.device)

        x = torch.cat([x_cat, x_cont], dim=1)
        x = self.joined_linear(x)
        x = F.leaky_relu(x)

        return x


    def final_layer(self,x):

        x = self.final(x)
        x = F.leaky_relu(x)
        return x   


    def forward(self, x):

        x = self.create_representation(x)
        x = self.final_layer(x)

        return x


    def save(self,model_filepath):

        checkpoint = {
        'x_dim': self.x_dim,
        'cat_indices': self.cat_indices, 
        'encodings': {key: {'mapping':self.encodings[key].mapping,'new_key':self.encodings[key].new_key} for key in self.encodings},
        'embedding_dim': self.embedding_dim,
        'representation_dim': self.representation_dim,
        'final_dim' : self.final_dim,
        'device': self.device,
        'state_dict': self.state_dict()
        }   
        torch.save(checkpoint,str(model_filepath))


    @classmethod
    def load(cls,model_filepath):

        checkpoint = torch.load(model_filepath)
        hyperparameters = {key: checkpoint[key] for key in set(checkpoint.keys()) - set(["state_dict","encodings"])}
        encodings = {}
        for key in checkpoint['encodings']:
            encoding = LabelEncoder()
            encoding.mapping = checkpoint['encodings'][key]['mapping']
            encoding.new_key = checkpoint['encodings'][key]['new_key']
            encodings[key] = encoding
        hyperparameters['encodings'] = encodings   

        model = cls(**hyperparameters)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        return model




if __name__ == "__main__":
    data_dir = Path("../../../sandbox/data/pescan/")
    filepaths = list(data_dir.glob("*.parquet"))

    sample_rows = get_sample_rows(filepaths, total_rows=100000)
    cat_indices = get_categorical_indices(sample_rows)
    cat_columns = sample_rows.iloc[:, cat_indices]
    encodings = train_categorical_encodings(cat_columns)
    cat_vocab_lengths = get_encoding_vocab_lengths(encodings)

    x_dim = sample_rows.shape[1]
    embedding_dim = 3
    model = TabularNet(x_dim, cat_indices, encodings, embedding_dim)

    for filepath in filepaths:
        batch = pd.read_parquet(filepath)
        embedding = model(batch)


    model_filepath = './test.pt'
    model.save(model_filepath)
    obj = TabularNet.load(model_filepath)


    for filepath in filepaths:
        batch = pd.read_parquet(filepath)
        embedding = obj(batch)


























