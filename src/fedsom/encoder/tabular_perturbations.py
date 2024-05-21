import copy
from collections import Counter
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from welford import Welford
from pathlib import Path     
import time    

from deepclustering.encoder.utils.data_utils import recast_columns


class DataTrimmer(nn.Module):

    def __init__(self, approx_constant_thresh):

        self.approx_constant_thresh = approx_constant_thresh 
        self.sparsities = Welford()

    def learn_sparsities(self, batch: pd.DataFrame) -> None:
        """we count None as 0 for categorical columns for the sake of measuring sparsity"""

        if not isinstance(batch,pd.DataFrame):
            batch = pd.DataFrame(batch)
        self.sparsities.add_all((batch.isnull() + (batch == 0)).astype(int).values)

    def trim_constant_columns(self,batch):

        if not isinstance(batch,pd.DataFrame):
            batch = pd.DataFrame(batch)
        return batch.iloc[:,self.sparsities.mean<self.approx_constant_thresh]

    def get_data_dim(self):
        return (self.sparsities.mean<self.approx_constant_thresh).sum()

    def save(self, model_filepath):

        checkpoint = {
        'approx_constant_thresh' : self.approx_constant_thresh,
        'sparsities' : self.sparsities,
        }
        torch.save(checkpoint,str(model_filepath))

    @classmethod
    def load(cls,model_filepath:Path):

        checkpoint = torch.load(model_filepath)
        obj = cls(**{k:v for k,v in checkpoint.items() if k not in ['sparsities']})
        obj.sparsities = checkpoint['sparsities']
        return obj 



class TabularPerturbation(nn.Module):
    """Class implements methods for
    i) learning the column distributions of tabular data in an online fashion
    ii) sampling from the learned distribution depending on whether data has been
        specified as sparse
    iii) perturbing a given tabular batch for the sake of contrastive learning
    """

    def __init__(self, x_dim, sparsity_thresh:float=0.9, corruption_factor:float=0.5, sample_prior: float = 0.5, null_char: Union[int, float, None] = None):
        super().__init__()

        self.x_dim = x_dim
        self.sparsity_thresh = sparsity_thresh
        self.corruption_factor = corruption_factor 
        self.sample_prior = sample_prior
        self.null_char = null_char
        self.sparsities = Welford()
        self.sample_spaces = [set([]) for _ in range(x_dim)]
        self.cat_bool_index = None   


    def find_categorical_indices(self, batch: pd.DataFrame) -> None:
        """columns are treated differently depending on whether they contain categorical or numerical data"""

        batch = recast_columns(batch)
        self.cat_bool_index = np.array([True if str(batch[c].dtype) == "category" else False for c in batch.columns])


    def get_categorical_indices(self):
        return np.where(self.cat_bool_index)[0]


    def get_cat_bool_index(self):

        return self.cat_bool_index


    def update_sample_spaces(self,batch:pd.DataFrame) -> None:

        for n,c in enumerate(batch.columns):
            if self.cat_bool_index[n]:
                self.sample_spaces[n].update(set(batch[c].unique()))
            else:
                self.sample_spaces[n].update(set([batch[c].min(),batch[c].max()]))


    def learn_sparsities(self, batch: pd.DataFrame) -> None:
        """we count None as 0 for categorical columns for the sake of measuring sparsity"""

        self.sparsities.add_all((batch.isnull() + (batch == 0)).astype(int).values)


    def sample_these_columns(self):
        """choose which columns to perturb by sampling a Bernoulli distribution for every column independently"""
 
        prior = torch.bernoulli(torch.full((self.x_dim,), self.sample_prior)).bool()
        by_sparsity = torch.bernoulli(torch.tensor(self.sparsities.mean)).bool()
        return np.where(prior & by_sparsity)[0]


    def update(self, batch: pd.DataFrame) -> None:
        """update sparsity and column distributions, update for every batch during training"""

        self.learn_sparsities(batch)
        self.update_sample_spaces(batch)


    def perturb_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        """sample chosen columns and create perturbed batch constructed from sampled columns and unsampled columlns"""

        col_idx = self.sample_these_columns()
        perturbed_batch = batch.copy()
        for c in col_idx:
            idx = np.random.permutation(batch.shape[0])
            perturbed_batch.iloc[:,c] = perturbed_batch.iloc[idx,c] 
            if self.sparsities.mean[c]>self.sparsity_thresh:
                non_corrupted_rows = ~torch.bernoulli(torch.full((batch.shape[0],), self.corruption_factor)).bool().numpy()
                if self.cat_bool_index[c]:
                    new_values = np.random.choice(list(self.sample_spaces[c]),batch.shape[0],replace=True)
                    new_values[non_corrupted_rows] = batch.iloc[non_corrupted_rows,c]
                else:
                    this_sample_space = np.array(list(self.sample_spaces[c]))
                    low,high = np.min(this_sample_space),np.max(this_sample_space)
                    new_values = np.random.uniform(low,high,size=batch.shape[0])
                    new_values[non_corrupted_rows] = batch.iloc[non_corrupted_rows,c]
                perturbed_batch.iloc[:,c] = new_values

        return perturbed_batch


    def save(self, model_filepath):

        checkpoint = {
        'x_dim' : self.x_dim,
        'sparsity_thresh' : self.sparsity_thresh,
        'corruption_factor' : self.corruption_factor,
        'sample_prior' : self.sample_prior,
        'null_char' : self.null_char,
        'sparsities' : self.sparsities,
        'sample_spaces' : self.sample_spaces,
        'cat_bool_index' : self.cat_bool_index
        }

        torch.save(checkpoint,str(model_filepath))

    @classmethod
    def load(cls,model_filepath:Path):

        checkpoint = torch.load(model_filepath)
        obj = cls(**{k:v for k,v in checkpoint.items() if k not in ['sparsities','sample_spaces','cat_bool_index']})
        obj.sparsities = checkpoint['sparsities']
        obj.sample_spaces = checkpoint['sample_spaces']
        obj.cat_bool_index = checkpoint['cat_bool_index']
        return obj    


def generate_random_dataframe(num_rows, num_categorical_cols, num_numerical_cols, sparsity=0.5):

    np.random.seed(0)
    categorical_data = np.random.choice(['A', 'B', 'C',None], size=(num_rows, num_categorical_cols))
    numerical_data = np.random.randn(num_rows, num_numerical_cols)
    numerical_data[np.random.rand(num_rows,num_numerical_cols)<sparsity] = 0

    cat_columns = [f'cat_{i+1}' for i in range(num_categorical_cols)]
    num_columns = [f'num_{i+1}' for i in range(num_numerical_cols)]

    data = np.concatenate((categorical_data, numerical_data), axis=1)
    df = pd.DataFrame(data, columns=cat_columns + num_columns)

    return df









if __name__=='__main__':



    num_rows = 30
    num_categorical_cols = 5
    num_numerical_cols = 3
    sparsity = 0.9


    batches = []
    for _ in range(2):
        random_df = generate_random_dataframe(num_rows, num_categorical_cols, num_numerical_cols, sparsity)
        batches.append(random_df)
    big_batch = pd.concat(batches)


    x_dim = num_categorical_cols+num_numerical_cols
    sparsity_thresh = 0.5
    approx_constant_thresh = 0.95
    corruption_factor = 0.5
    sample_prior = 1.
    null_char = None
    tabular_perturbation = TabularPerturbation(
                                x_dim=x_dim, 
                                sparsity_thresh=sparsity_thresh, 
                                approx_constant_thresh=approx_constant_thresh, 
                                corruption_factor=corruption_factor,
                                sample_prior=sample_prior,
                                null_char=null_char
          )
    tabular_perturbation.find_categorical_indices(big_batch)


    for batch in batches:
        tabular_perturbation.update(batch)



    print(tabular_perturbation.sample_spaces)
    model_filepath = './test.pt'
    tabular_perturbation.save(model_filepath)
    obj = TabularPerturbation.load(model_filepath)
    print('\n\n\n')
    print(obj.sample_spaces)



    # for batch in batches:

    #     perturbed_batch = tabular_perturbations.perturb_batch(batch)
    #     batch = tabular_perturbations.trim_constant_columns(batch)
    #     perturbed_batch = tabular_perturbations.trim_constant_columns(perturbed_batch)


        # print(batch)

        # print('\n')
        # print(perturbed_batch)

        # input('')






    # def collect_unique(batch,unique):

    #     for n,c in enumerate(batch.columns):
    #         unique[n].update(set(batch[c].unique()))

    #     return unique



    # x_dim = 3
    # unique = [set([]) for _ in range(x_dim)]

    # batch1 = pd.DataFrame(np.random.choice(['a','b'],(2,3)))
    # batch2 = pd.DataFrame(np.random.choice(['a','e','d'],(2,3)))
    # batches = [batch1,batch2]


    # for batch in batches:
    #     unique = collect_unique(batch,unique)


    # print(unique)


    # non_constant = torch.tensor([True,False,True,True,False])
    # sample_prior = 1.0
    # sparsities = torch.tensor([0.99,0.5,0.1,0.99,0.99])

    # prior = torch.bernoulli(torch.full((non_constant.sum(),), sample_prior)).bool()
    # by_sparsity = torch.bernoulli(sparsities[non_constant]).bool()
    # idx = prior & by_sparsity

    # print(idx)




































