from typing import Any, Dict
import pandas as pd         
import joblib
import numpy as np    
from sklearn.preprocessing import LabelEncoder



def all_numeric(column):
    try:
        pd.to_numeric(column)
        return True
    except ValueError:
        return False 

def replace_key_in_dict(dictionary: Dict, old_key: Any, new_key: Any) -> Dict:
    new_dict = {}
    for key, value in dictionary.items():
        if key == old_key:
            new_dict[new_key] = value
        else:
            new_dict[key] = value
    return new_dict


def recast_columns(vectors: pd.DataFrame) -> pd.DataFrame:
    for c in vectors.columns:

        unique_values = vectors[c].dropna().unique()
        if all_numeric(vectors[c]):
            all_ints = all(value==int(value) for value in unique_values)
        else:
            all_ints = False
        other_categorical = [isinstance(value, (str, bool)) or (value is None) for value in vectors[c]]
        contains_strings = [isinstance(value,str) for value in vectors[c]]
        is_bool = set(unique_values) == {0, 1}
        if all(other_categorical) or is_bool or any(contains_strings) or all_ints:
            vectors[c] = pd.Categorical(vectors[c], categories=unique_values)
    return vectors


class CategoricalOneHotEncoder:
    def __init__(self):
        self.column_encodings = {}
        self.remove = None

    def record_high_entropy_cols(self,data):

        self.remove = np.array([True if len(data[col].unique())/data.shape[0]>0.1 else False for col in data.columns])

    def remove_high_entropy_cols(self,data):

        return data.loc[:,~self.remove]

    def fit_transform(self, data):

        data = self.remove_high_entropy_cols(data)
        if data.shape[1]==0:
            return pd.DataFrame()
        columns = []
        for col in data.columns: 
            one_hot = pd.get_dummies(data[col], prefix=col)
            self.column_encodings[col] = one_hot.columns.tolist()
            columns.append(one_hot)

        transformed_data = pd.concat(columns,axis=1)
        return transformed_data.astype(int)

    def transform(self, data):

        data = self.remove_high_entropy_cols(data)
        if data.shape[1]==0:
            return pd.DataFrame()
        columns = []
        for col in data.columns:

            one_hot = pd.get_dummies(data[col], prefix=col)
            missing_cols = set(self.column_encodings[col]) - set(one_hot.columns)
            for c in missing_cols:
                one_hot[c] = 0  # Add missing columns with zeros
            one_hot = one_hot[self.column_encodings[col]]
            columns.append(one_hot)
 
        transformed_new_data = pd.concat(columns,axis=1)
        return transformed_new_data.astype(int)

    def save(self, filepath):
        content = {}
        content['column_encodings'] = self.column_encodings
        content['remove'] = self.remove    
        joblib.dump(content,filepath)

    @classmethod
    def load(cls, filepath):
        loaded_encoder = cls()
        content = joblib.load(filepath)
        loaded_encoder.column_encodings = content['column_encodings']
        loaded_encoder.remove = content['remove']
        # loaded_encoder.column_encodings = joblib.load(filepath)

        return loaded_encoder


def convert_string_columns_to_integers(df,exclude):
    label_encoders = {}
    for col in [c for c in df.columns if c not in exclude]:
        if df[col].dtype == 'object':
            label_encoders[col] = LabelEncoder()
            if any([isinstance(x,str) for x in df[col]]):
                df[col] = df[col].astype(str)
            df[col] = label_encoders[col].fit_transform(df[col]).astype(int)
    return df



if __name__=='__main__':


    # df = pd.DataFrame(np.random.randn(20,2))
    # df = pd.DataFrame(np.random.choice(['a','b'],size=(20,2)))
    # print(df)
    # for c in df.columns:
    #     print(all_numeric(df[c]))


    # input('DONE')




    # pdf 
    train_path = '../../../../sandbox/data/MalwareDatasets/MalwareDatasets/PDFMalware/total/pdfmalware_train.csv'
    test_path = '../../../../sandbox/data/MalwareDatasets/MalwareDatasets/PDFMalware/total/pdfmalware_test.csv'

    new_train_path = '../../../../sandbox/data/MalwareDatasets/MalwareDatasets/PDFMalware/total/pdfmalware_train_converted.csv'
    new_test_path = '../../../../sandbox/data/MalwareDatasets/MalwareDatasets/PDFMalware/total/pdfmalware_test_converted.csv'

    train_data = pd.read_csv(train_path,index_col='sha256')
    test_data = pd.read_csv(test_path,index_col='sha256')

    pdf = pd.concat([train_data,test_data])
    pdf.rename(columns={'Class':'avclass'},inplace=True)
    pdf_converted = convert_string_columns_to_integers(pdf,exclude=['avclass'])

    train_data_converted = pdf.iloc[:train_data.shape[0],:]
    test_data_converted = pdf.iloc[train_data.shape[0]:,:]

    train_data_converted.to_csv(new_train_path,index='sha256')
    test_data_converted.to_csv(new_test_path,index='sha256')

    train_data_converted_test = pd.read_csv(new_train_path,index_col='sha256')
    test_data_converted_test = pd.read_csv(new_test_path,index_col='sha256')

    print(train_data_converted)
    print(train_data_converted_test)


    # cicandmal2017 
    train_path = '../../../../sandbox/data/MalwareDatasets/MalwareDatasets/CICAndMal2017/total/cic_andmal2017_train.csv'
    test_path = '../../../../sandbox/data/MalwareDatasets/MalwareDatasets/CICAndMal2017/total/cic_andmal2017_test.csv'

    new_train_path = '../../../../sandbox/data/MalwareDatasets/MalwareDatasets/CICAndMal2017/total/cic_andmal2017_train_converted.csv'
    new_test_path = '../../../../sandbox/data/MalwareDatasets/MalwareDatasets/CICAndMal2017/total/cic_andmal2017_test_converted.csv'

    train_data = pd.read_csv(train_path,index_col='Flow ID')
    test_data = pd.read_csv(test_path,index_col='Flow ID')

    pdf = pd.concat([train_data,test_data])
    pdf.rename(columns={'Class':'avclass'},inplace=True)
    pdf_converted = convert_string_columns_to_integers(pdf,exclude=['avclass'])

    train_data_converted = pdf.iloc[:train_data.shape[0],:]
    test_data_converted = pdf.iloc[train_data.shape[0]:,:]

    train_data_converted.to_csv(new_train_path,index='Flow ID')
    test_data_converted.to_csv(new_test_path,index='Flow ID')

    train_data_converted_test = pd.read_csv(new_train_path,index_col='Flow ID')
    test_data_converted_test = pd.read_csv(new_test_path,index_col='Flow ID')

    print(train_data_converted)
    print(train_data_converted_test)


    #  malmem
    train_path = '../../../../sandbox/data/MalwareDatasets/MalwareDatasets/MalMem/total/Obfuscated-MalMem2022_train.csv'
    test_path = '../../../../sandbox/data/MalwareDatasets/MalwareDatasets/MalMem/total/Obfuscated-MalMem2022_test.csv'

    new_train_path = '../../../../sandbox/data/MalwareDatasets/MalwareDatasets/MalMem/total/Obfuscated-MalMem2022_train_converted.csv'
    new_test_path = '../../../../sandbox/data/MalwareDatasets/MalwareDatasets/MalMem/total/Obfuscated-MalMem2022_test_converted.csv'
    
    train_data = pd.read_csv(train_path,index_col='sha256')
    test_data = pd.read_csv(test_path,index_col='sha256')

    pdf = pd.concat([train_data,test_data])
    pdf.rename(columns={'Class':'avclass'},inplace=True)
    pdf_converted = convert_string_columns_to_integers(pdf,exclude=['avclass','Class','subavclass'])

    train_data_converted = pdf.iloc[:train_data.shape[0],:]
    test_data_converted = pdf.iloc[train_data.shape[0]:,:]

    train_data_converted.to_csv(new_train_path,index='sha256')
    test_data_converted.to_csv(new_test_path,index='sha256')

    train_data_converted_test = pd.read_csv(new_train_path,index_col='sha256')
    test_data_converted_test = pd.read_csv(new_test_path,index_col='sha256')

    print(train_data_converted)
    print(train_data_converted_test)































