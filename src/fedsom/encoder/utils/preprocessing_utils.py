from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import hashlib
from sklearn.preprocessing import PowerTransformer,StandardScaler



def concatenate_and_hash(lst):

    string_list = [str(item) for item in lst]
    concatenated_string = ''.join(string_list)
    hashed_string = hashlib.sha1(concatenated_string.encode()).hexdigest()[:10]

    return hashed_string


class LabelEncoder(object):
    def __init__(
        self,
    ):
        super().__init__()

        self.mapping = None 

    def fit(self, labels: Union[pd.Series, np.ndarray]):
        self.mapping = {label: n for n, label in enumerate(sorted(list(set(labels))))}
        self.new_key = concatenate_and_hash(list(self.mapping.keys()))
        self.mapping[self.new_key] = len(self.mapping)

    def transform(self, data: Union[pd.Series, np.ndarray]):
        self.replacement_func = np.vectorize(lambda x: self.mapping.get(x, x))
        new_keys = np.isin(data,list(self.mapping.keys()),invert=True)
        data[new_keys] = self.new_key
        return self.replacement_func(data)

    def __len__(self):
        return len(self.mapping)



def get_cat_vocab_sizes(batch: pd.DataFrame) -> Dict[int, int]:
    cat_vocab_sizes = {}
    for idx, c in enumerate(batch.columns):
        unique_values = batch[c].dropna().unique()
        cat_vocab_sizes[idx] = len(unique_values)
    return cat_vocab_sizes


def get_encoding_vocab_lengths(encodings: Dict[int, LabelEncoder]) -> Dict[int, int]:
    return {k: len(v) for k, v in encodings.items()}


def get_categorical_indices(batch: pd.DataFrame) -> List[int]:
    cat_indices = []
    for idx, c in enumerate(batch.columns):
        unique_values = batch[c].dropna().unique()
        other_categorical = [isinstance(value, (str, bool)) for value in batch[c]]
        is_bool = set(unique_values) == {0, 1}
        if all(other_categorical) or is_bool:
            cat_indices.append(idx)
    return cat_indices


def get_sample_rows(filepaths: List[Path], total_rows: int = 1000) -> pd.DataFrame:
    num_rows = 0
    batches = []
    for filepath in filepaths:
        batch = pd.read_parquet(filepath)
        num_rows += batch.shape[0]
        batches.append(batch)
        if num_rows > total_rows:
            break
    return pd.concat(batches)


def train_categorical_encodings(categorical_columns: pd.DataFrame) -> Dict[int, LabelEncoder]:
    encodings = {}
    for idx in range(categorical_columns.shape[1]):
        le = LabelEncoder()
        le.fit(categorical_columns.iloc[:, idx])
        encodings[idx] = le
    return encodings


def encode_categoricals(batch: pd.DataFrame, encodings: Dict[int, LabelEncoder]) -> np.ndarray:
    new_batch = []
    for idx in range(batch.shape[1]):
        new_batch.append(encodings[idx].transform(batch.iloc[:, idx]).astype(int).reshape((batch.shape[0], 1)))
    return np.hstack(new_batch)


def parse_columns(batch: pd.DataFrame, categorical_indices: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return (
        batch.iloc[:, categorical_indices],
        batch.iloc[:, [idx for idx in range(batch.shape[1]) if idx not in categorical_indices]],
    )


def replace_nans(df):

    return df.fillna(df.median())


def replace_nans_np_array(arr):

    column_medians = np.nanmedian(arr,axis=0)
    nan_indices = np.isnan(arr)
    return np.where(nan_indices,np.tile(column_medians,(arr.shape[0],1)),arr)


def train_power_transformer(df):

    # pt = PowerTransformer(standardize=False)
    pt = DenseStandardScaler()
    pt.fit(df)

    return pt   


# class DenseStandardScaler(StandardScaler):
#     def __init__(self, copy=True, with_mean=True, with_std=True):
#         super().__init__(copy=copy, with_mean=with_mean, with_std=with_std)

#     def fit(self, X, y=None):

#         # input('THISDLFJSDLKFJSDLKFJDLFKJ')
#         if isinstance(X, pd.DataFrame):
#             X = X.values
#         X_nonzero = X[X != 0].reshape(-1, 1)
#         return super().fit(X_nonzero)

#     def transform(self, X, y=None, copy=None):
#         if copy is None:
#             copy = self.copy
#         if isinstance(X, pd.DataFrame):
#             X_values = X.values
#             X_nonzero = X_values[X_values != 0].reshape(-1, 1)
#             X_transformed = super().transform(X_nonzero, copy)

#             # print(X_nonzero)
#             # print(X_transformed)
#             # input('transformed')
#             X_values[X_values != 0] = X_transformed.flatten()
#             return X_values
#         elif isinstance(X, np.ndarray):
#             X_nonzero = X[X != 0].reshape(-1, 1)
#             X_transformed = super().transform(X_nonzero, copy)
#             X[X != 0] = X_transformed.flatten()
#             return X
#         else:
#             raise ValueError("Input data must be a pandas DataFrame or a numpy array")


class DenseStandardScaler(StandardScaler):
    def __init__(self, copy=True, with_mean=True, with_std=True):
        super().__init__(copy=copy, with_mean=with_mean, with_std=with_std)

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.scalers_ = []
        for col in range(X.shape[1]):
            scaler = StandardScaler(copy=self.copy, with_mean=self.with_mean, with_std=self.with_std)
            scaler.fit(X[X[:, col] != 0, col].reshape(-1, 1))
            self.scalers_.append(scaler)
        return self

    def transform(self, X, y=None, copy=None):
        if copy is None:
            copy = self.copy
        if isinstance(X, pd.DataFrame):
            X_values = X.values
            for col, scaler in enumerate(self.scalers_):
                nonzero_indices = X_values[:, col] != 0
                if np.any(nonzero_indices):
                    X_values[nonzero_indices, col] = scaler.transform(X_values[nonzero_indices, col].reshape(-1, 1)).flatten()
            return X_values
        elif isinstance(X, np.ndarray):
            for col, scaler in enumerate(self.scalers_):
                nonzero_indices = X[:, col] != 0
                if np.any(nonzero_indices):
                    X[nonzero_indices, col] = scaler.transform(X[nonzero_indices, col].reshape(-1, 1)).flatten()
            return X
        else:
            raise ValueError("Input data must be a pandas DataFrame or a numpy array")



if __name__=='__main__':
    import joblib

    np.random.seed(0)

    X = pd.DataFrame(np.random.choice([0,1,2,3],(20,2),replace=True))
    for i in range(X.shape[0]):
        print(X.values[i,:])

    scaler = DenseStandardScaler()
    scaler.fit(X)

    print('------------')

    X = scaler.transform(X)
    for i in range(X.shape[0]):
        print(X.values[i,:])

    joblib.dump(scaler,'test.joblib')

    scaler = joblib.load('test.joblib')


    X = pd.DataFrame(np.random.choice([0,1,2,3],(20,2),replace=True))
    X = scaler.transform(X)
    print(X)




# Example usage:
# scaler = DenseStandardScaler()
# X_scaled = scaler.fit_transform(X)










    
