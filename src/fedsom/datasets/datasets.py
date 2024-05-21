from torch.utils.data import Dataset
import numpy as np    
import pandas as pd 
from sklearn.model_selection import train_test_split   


def fill_missing_values(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)
            # For object columns, replace missing or NaN values with the mode
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
        elif pd.api.types.is_numeric_dtype(df[col]):
            # For numeric columns, replace NaN values with the median
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
    return df

def convert_string_to_int(df):
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            # Check if all strings in the column can be naturally cast to integers
            if all(is_integer(val) for val in df[col]):
                df[col] = df[col].astype(int)
            else:
                df.drop(columns=[col],inplace=True)
    return df

# Function to check if a string can be naturally cast to an integer
def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def split_dataframe(df, num_samples_train, num_samples_validate, num_samples_test):
    
    np.random.seed(0)
    total_samples = df.shape[0]

    train_size = min(int(0.8 * total_samples), num_samples_train)
    validate_size = min(int(0.1 * total_samples), num_samples_validate)
    test_size = min(int(0.1 * total_samples), num_samples_test)

    train, remaining = train_test_split(df, train_size=0.8, random_state=42)
    validate, test = train_test_split(remaining, test_size=0.5, random_state=42)

    train_idx = np.random.choice(train.shape[0],train_size,replace=False)
    validate_idx = np.random.choice(validate.shape[0],validate_size,replace=False)
    test_idx = np.random.choice(test.shape[0],test_size,replace=False)

    return train.iloc[train_idx,:], validate.iloc[validate_idx,:], test.iloc[test_idx,:]


class CIFAR10DatasetInMem(object):

    @classmethod
    def load(cls,data_path):
        return pd.read_csv(data_path)

    @classmethod
    def load_single(cls,data_path):
        data = cls.load(data_path)
        vectors = data[[c for c in data.columns if c!='label']].values.astype('float32')
        labels = data['label'].values
        index = data.index
        return vectors,labels,index

    @classmethod
    def split_data(cls,data,num_samples_train, num_samples_validate, num_samples_test):

        train, validate, test = split_dataframe(data, num_samples_train, num_samples_validate, num_samples_test)
        
        train_vectors = train[[c for c in train.columns if c!='label']].values.astype('float32')
        train_labels = train['label'].values

        validation_vectors = validate[[c for c in validate.columns if c!='label']].values.astype('float32')
        validation_labels = validate['label'].values

        test_vectors = test[[c for c in test.columns if c!='label']].values.astype('float32')
        test_labels = test['label'].values

        data_dict = {}
        data_dict['train_vectors'] = train_vectors  
        data_dict['train_labels'] = train_labels  

        data_dict['validation_vectors'] = validation_vectors  
        data_dict['validation_labels'] = validation_labels 

        data_dict['test_vectors'] = test_vectors  
        data_dict['test_labels'] = test_labels 

        return data_dict


class MNISTDatasetInMem(object):

    @classmethod
    def load(cls,data_path):
        return pd.read_csv(data_path)

    @classmethod
    def load_single(cls,data_path):
        data = cls.load(data_path)
        vectors = data[[c for c in data.columns if c!='label']].values.astype('float32')
        labels = data['label'].values
        index = data.index
        return vectors,labels,index

    @classmethod
    def split_data(cls,data,num_samples_train, num_samples_validate, num_samples_test):

        train, validate, test = split_dataframe(data, num_samples_train, num_samples_validate, num_samples_test)
        
        train_vectors = train[[c for c in train.columns if c!='label']].values.astype('float32')
        train_labels = train['label'].values

        validation_vectors = validate[[c for c in validate.columns if c!='label']].values.astype('float32')
        validation_labels = validate['label'].values

        test_vectors = test[[c for c in test.columns if c!='label']].values.astype('float32')
        test_labels = test['label'].values

        data_dict = {}
        data_dict['train_vectors'] = train_vectors  
        data_dict['train_labels'] = train_labels  

        data_dict['validation_vectors'] = validation_vectors  
        data_dict['validation_labels'] = validation_labels 

        data_dict['test_vectors'] = test_vectors  
        data_dict['test_labels'] = test_labels 

        return data_dict


class FashionMNISTDatasetInMem(object):

    @classmethod
    def load(cls,data_path):
        return pd.read_csv(data_path)

    @classmethod
    def load_single(cls,data_path):
        data = cls.load(data_path)
        vectors = data[[c for c in data.columns if c!='label']].values.astype('float32')
        labels = data['label'].values
        index = data.index
        return vectors,labels,index

    @classmethod
    def split_data(cls,data,num_samples_train, num_samples_validate, num_samples_test):

        train, validate, test = split_dataframe(data, num_samples_train, num_samples_validate, num_samples_test)
        
        train_vectors = train[[c for c in train.columns if c!='label']].values.astype('float32')
        train_labels = train['label'].values

        validation_vectors = validate[[c for c in validate.columns if c!='label']].values.astype('float32')
        validation_labels = validate['label'].values

        test_vectors = test[[c for c in test.columns if c!='label']].values.astype('float32')
        test_labels = test['label'].values

        data_dict = {}
        data_dict['train_vectors'] = train_vectors  
        data_dict['train_labels'] = train_labels  

        data_dict['validation_vectors'] = validation_vectors  
        data_dict['validation_labels'] = validation_labels 

        data_dict['test_vectors'] = test_vectors  
        data_dict['test_labels'] = test_labels 

        return data_dict



class Chars74kDatasetInMem(object):

    @classmethod
    def load(cls,data_path):
        d = pd.read_csv(data_path)
        e = 1.-d[[c for c in d.columns if c!='label']]
        e['label'] = d['label']
        return e 

    @classmethod
    def load_single(cls,data_path):
        data = cls.load(data_path)
        vectors = data[[c for c in data.columns if c!='label']].values.astype('float32')
        labels = data['label'].values
        index = data.index 
        return vectors,labels,index

    @classmethod
    def split_data(cls,data,num_samples_train, num_samples_validate, num_samples_test):

        train, validate, test = split_dataframe(data, num_samples_train, num_samples_validate, num_samples_test)
        
        train_vectors = train[[c for c in train.columns if c!='label']].values.astype('float32')
        train_labels = train['label'].values

        validation_vectors = validate[[c for c in validate.columns if c!='label']].values.astype('float32')
        validation_labels = validate['label'].values

        test_vectors = test[[c for c in test.columns if c!='label']].values.astype('float32')
        test_labels = test['label'].values

        data_dict = {}
        data_dict['train_vectors'] = train_vectors  
        data_dict['train_labels'] = train_labels  

        data_dict['validation_vectors'] = validation_vectors  
        data_dict['validation_labels'] = validation_labels 

        data_dict['test_vectors'] = test_vectors  
        data_dict['test_labels'] = test_labels 

        return data_dict



class NotMNISTDatasetInMem(object):

    @classmethod
    def load(cls,data_path):
        return pd.read_csv(data_path)

    @classmethod
    def load_single(cls,data_path):
        data = cls.load(data_path)
        vectors = data[[c for c in data.columns if c!='label']].values.astype('float32')
        labels = data['label'].values
        index = data.index 
        return vectors,labels,index

    @classmethod
    def split_data(cls,data,num_samples_train, num_samples_validate, num_samples_test):

        train, validate, test = split_dataframe(data, num_samples_train, num_samples_validate, num_samples_test)
        
        train_vectors = train[[c for c in train.columns if c!='label']].values.astype('float32')
        train_labels = train['label'].values

        validation_vectors = validate[[c for c in validate.columns if c!='label']].values.astype('float32')
        validation_labels = validate['label'].values

        test_vectors = test[[c for c in test.columns if c!='label']].values.astype('float32')
        test_labels = test['label'].values

        data_dict = {}
        data_dict['train_vectors'] = train_vectors  
        data_dict['train_labels'] = train_labels  

        data_dict['validation_vectors'] = validation_vectors  
        data_dict['validation_labels'] = validation_labels 

        data_dict['test_vectors'] = test_vectors  
        data_dict['test_labels'] = test_labels 

        return data_dict



class QuickdrawInMem(object):

    @classmethod
    def load(cls,data_path):
        return pd.read_csv(data_path)

    @classmethod
    def load_single(cls,data_path):
        data = cls.load(data_path)
        vectors = (1./255)*data[[c for c in data.columns if c!='label']].values.astype('float32')
        labels = data['label'].values
        index = data.index 
        return vectors,labels,index 

    @classmethod
    def split_data(cls,data,num_samples_train, num_samples_validate, num_samples_test):

        train, validate, test = split_dataframe(data, num_samples_train, num_samples_validate, num_samples_test)
        
        train_vectors = (1./255)*train[[c for c in train.columns if c!='label']].values.astype('float32')
        train_labels = train['label'].values

        validation_vectors = (1./255)*validate[[c for c in validate.columns if c!='label']].values.astype('float32')
        validation_labels = validate['label'].values

        test_vectors = (1./255)*test[[c for c in test.columns if c!='label']].values.astype('float32')
        test_labels = test['label'].values

        data_dict = {}
        data_dict['train_vectors'] = train_vectors  
        data_dict['train_labels'] = train_labels  

        data_dict['validation_vectors'] = validation_vectors  
        data_dict['validation_labels'] = validation_labels 

        data_dict['test_vectors'] = test_vectors  
        data_dict['test_labels'] = test_labels 

        return data_dict


class SLMNISTInMem(object):

    @classmethod
    def load(cls,data_path):
        return pd.read_csv(data_path)

    @classmethod
    def load_single(cls,data_path):
        data = cls.load(data_path)
        vectors = (1./255)*data[[c for c in data.columns if c!='label']].values.astype('float32')
        labels = data['label'].values
        index = data.index 
        return vectors,labels,index

    @classmethod
    def split_data(cls,data,num_samples_train, num_samples_validate, num_samples_test):

        train, validate, test = split_dataframe(data, num_samples_train, num_samples_validate, num_samples_test)
        
        train_vectors = (1./255)*train[[c for c in train.columns if c!='label']].values.astype('float32')
        train_labels = train['label'].values

        validation_vectors = (1./255)*validate[[c for c in validate.columns if c!='label']].values.astype('float32')
        validation_labels = validate['label'].values

        test_vectors = (1./255)*test[[c for c in test.columns if c!='label']].values.astype('float32')
        test_labels = test['label'].values

        data_dict = {}
        data_dict['train_vectors'] = train_vectors  
        data_dict['train_labels'] = train_labels  

        data_dict['validation_vectors'] = validation_vectors  
        data_dict['validation_labels'] = validation_labels 

        data_dict['test_vectors'] = test_vectors  
        data_dict['test_labels'] = test_labels 

        return data_dict



class KuzushijiInMem(object):

    @classmethod
    def load(cls,data_path):
        return pd.read_csv(data_path)

    @classmethod
    def load_single(cls,data_path):
        data = cls.load(data_path)
        vectors = data[[c for c in data.columns if c!='label']].values.astype('float32')
        labels = data['label'].values
        index = data.index 
        return vectors,labels,index 

    @classmethod
    def split_data(cls,data,num_samples_train, num_samples_validate, num_samples_test):

        train, validate, test = split_dataframe(data, num_samples_train, num_samples_validate, num_samples_test)
        
        train_vectors = train[[c for c in train.columns if c!='label']].values.astype('float32')
        train_labels = train['label'].values

        validation_vectors = validate[[c for c in validate.columns if c!='label']].values.astype('float32')
        validation_labels = validate['label'].values

        test_vectors = test[[c for c in test.columns if c!='label']].values.astype('float32')
        test_labels = test['label'].values

        data_dict = {}
        data_dict['train_vectors'] = train_vectors  
        data_dict['train_labels'] = train_labels  

        data_dict['validation_vectors'] = validation_vectors  
        data_dict['validation_labels'] = validation_labels 

        data_dict['test_vectors'] = test_vectors  
        data_dict['test_labels'] = test_labels 

        return data_dict



class EMNISTInMem(object):

    @classmethod
    def load(cls,data_path):
        return pd.read_csv(data_path)

    @classmethod
    def load_single(cls,data_path):
        data = cls.load(data_path)
        vectors = data[[c for c in data.columns if c!='label']].values.astype('float32')
        labels = data['label'].values
        index = data.index 
        return vectors,labels,index 

    @classmethod
    def split_data(cls,data,num_samples_train, num_samples_validate, num_samples_test):

        train, validate, test = split_dataframe(data, num_samples_train, num_samples_validate, num_samples_test)
        
        train_vectors = train[[c for c in train.columns if c!='label']].values.astype('float32')
        train_labels = train['label'].values

        validation_vectors = validate[[c for c in validate.columns if c!='label']].values.astype('float32')
        validation_labels = validate['label'].values

        test_vectors = test[[c for c in test.columns if c!='label']].values.astype('float32')
        test_labels = test['label'].values

        data_dict = {}
        data_dict['train_vectors'] = train_vectors  
        data_dict['train_labels'] = train_labels  

        data_dict['validation_vectors'] = validation_vectors  
        data_dict['validation_labels'] = validation_labels 

        data_dict['test_vectors'] = test_vectors  
        data_dict['test_labels'] = test_labels 

        return data_dict


class SorelDatasetInMem(object):

    @classmethod
    def load(cls,data_path):
        return pd.read_csv(data_path,index_col='sha256')

    @classmethod
    def load_single(cls,data_path):
        data = cls.load(data_path)
        vectors = data[[c for c in data.columns if c!='avclass']].values
        labels = data['avclass'].values
        index = data.index 
        return vectors,labels,index 

    @classmethod
    def split_data(cls,data,num_samples_train, num_samples_validate, num_samples_test):

        train, validate, test = split_dataframe(data, num_samples_train, num_samples_validate, num_samples_test)
        
        train_vectors = train[[c for c in train.columns if c!='avclass']].values
        train_labels = np.array(train['avclass'].tolist())

        validation_vectors = validate[[c for c in validate.columns if c!='avclass']].values
        validation_labels = np.array(validate['avclass'].tolist())

        test_vectors = test[[c for c in test.columns if c!='avclass']].values
        test_labels = np.array(test['avclass'].tolist())

        data_dict = {}
        data_dict['train_vectors'] = train_vectors  
        data_dict['train_labels'] = train_labels  

        data_dict['validation_vectors'] = validation_vectors  
        data_dict['validation_labels'] = validation_labels 

        data_dict['test_vectors'] = test_vectors  
        data_dict['test_labels'] = test_labels 

        return data_dict


class EmberDatasetInMem(object):

    @classmethod
    def load(cls,data_path):
        return pd.read_csv(data_path,index_col='sha256')

    @classmethod
    def load_single(cls,data_path):
        data = cls.load(data_path)
        vectors = data[[c for c in data.columns if c!='avclass']].values
        labels = data['avclass'].values
        index = data.index 
        return vectors,labels,index 

    @classmethod
    def split_data(cls,data,num_samples_train, num_samples_validate, num_samples_test):

        train, validate, test = split_dataframe(data, num_samples_train, num_samples_validate, num_samples_test)
        
        train_vectors = train[[c for c in train.columns if c!='avclass']].values
        train_labels = np.array(train['avclass'].tolist())

        validation_vectors = validate[[c for c in validate.columns if c!='avclass']].values
        validation_labels = np.array(validate['avclass'].tolist())

        test_vectors = test[[c for c in test.columns if c!='avclass']].values
        test_labels = np.array(test['avclass'].tolist())

        data_dict = {}
        data_dict['train_vectors'] = train_vectors  
        data_dict['train_labels'] = train_labels  

        data_dict['validation_vectors'] = validation_vectors  
        data_dict['validation_labels'] = validation_labels 

        data_dict['test_vectors'] = test_vectors  
        data_dict['test_labels'] = test_labels 

        return data_dict



class CCCSInMem(object):
    """https://www.unb.ca/cic/datasets/andmal2020.html"""

    @classmethod
    def load(cls,data_path):
        return pd.read_csv(data_path,index_col='sha256')

    @classmethod
    def load_single(cls,data_path):
        data = cls.load(data_path)
        vectors = data[[c for c in data.columns if c!='avclass']].values
        labels = np.array(data['avclass'].tolist())
        index = data.index 
        return vectors,labels,index 


    @classmethod
    def split_data(cls,data,num_samples_train, num_samples_validate, num_samples_test):

        train, validate, test = split_dataframe(data, num_samples_train, num_samples_validate, num_samples_test)
        
        train_vectors = train[[c for c in train.columns if c!='avclass']].values
        train_labels = np.array(train['avclass'].tolist())

        validation_vectors = validate[[c for c in validate.columns if c!='avclass']].values
        validation_labels = np.array(validate['avclass'].tolist())

        test_vectors = test[[c for c in test.columns if c!='avclass']].values
        test_labels = np.array(test['avclass'].tolist())

        data_dict = {}
        data_dict['train_vectors'] = train_vectors  
        data_dict['train_labels'] = train_labels  

        data_dict['validation_vectors'] = validation_vectors  
        data_dict['validation_labels'] = validation_labels 

        data_dict['test_vectors'] = test_vectors  
        data_dict['test_labels'] = test_labels 

        return data_dict



class CICAndMal2017InMem(object):
    """https://www.unb.ca/cic/datasets/andmal2017.html"""

    @classmethod
    def load(cls,data_path):
        return pd.read_csv(data_path,index_col='Flow ID')

    @classmethod
    def load_single(cls,data_path):
        data = cls.load(data_path)
        vectors = data[[c for c in data.columns if c!='avclass']].values
        labels = np.array(data['avclass'].tolist())
        index = data.index 
        return vectors,labels,index 

    @classmethod
    def split_data(cls,data,num_samples_train, num_samples_validate, num_samples_test):

        train, validate, test = split_dataframe(data, num_samples_train, num_samples_validate, num_samples_test)
        
        train_vectors = train[[c for c in train.columns if c!='avclass']].values
        train_labels = np.array(train['avclass'].tolist())

        validation_vectors = validate[[c for c in validate.columns if c!='avclass']].values
        validation_labels = np.array(validate['avclass'].tolist())

        test_vectors = test[[c for c in test.columns if c!='avclass']].values
        test_labels = np.array(test['avclass'].tolist())

        data_dict = {}
        data_dict['train_vectors'] = train_vectors  
        data_dict['train_labels'] = train_labels  

        data_dict['validation_vectors'] = validation_vectors  
        data_dict['validation_labels'] = validation_labels 

        data_dict['test_vectors'] = test_vectors  
        data_dict['test_labels'] = test_labels 

        return data_dict


class SysCallsInMem(object):
    """https://www.unb.ca/cic/datasets/maldroid-2020.html"""

    @classmethod
    def load(cls,data_path):
        return pd.read_csv(data_path,index_col=None)

    @classmethod
    def load_single(cls,data_path):
        data = cls.load(data_path)
        vectors = data[[c for c in data.columns if c!='avclass']].values
        labels = np.array(data['avclass'].tolist())
        index = data.index 
        return vectors,labels,index 

    @classmethod
    def split_data(cls,data,num_samples_train, num_samples_validate, num_samples_test):

        train, validate, test = split_dataframe(data, num_samples_train, num_samples_validate, num_samples_test)
        
        train_vectors = train[[c for c in train.columns if c!='avclass']].values
        train_labels = np.array(train['avclass'].tolist())

        validation_vectors = validate[[c for c in validate.columns if c!='avclass']].values
        validation_labels = np.array(validate['avclass'].tolist())

        test_vectors = test[[c for c in test.columns if c!='avclass']].values
        test_labels = np.array(test['avclass'].tolist())

        data_dict = {}
        data_dict['train_vectors'] = train_vectors  
        data_dict['train_labels'] = train_labels  

        data_dict['validation_vectors'] = validation_vectors  
        data_dict['validation_labels'] = validation_labels 

        data_dict['test_vectors'] = test_vectors  
        data_dict['test_labels'] = test_labels 

        return data_dict


class SysCallsBindersInMem(object):
    """https://www.unb.ca/cic/datasets/maldroid-2020.html"""

    @classmethod
    def load(cls,data_path):
        return pd.read_csv(data_path,index_col=None)

    @classmethod
    def load_single(cls,data_path):
        data = cls.load(data_path)
        vectors = data[[c for c in data.columns if c!='avclass']].values
        labels = np.array(data['avclass'].tolist())
        index = data.index 
        return vectors,labels,index 

    @classmethod
    def split_data(cls,data,num_samples_train, num_samples_validate, num_samples_test):

        train, validate, test = split_dataframe(data, num_samples_train, num_samples_validate, num_samples_test)
        
        train_vectors = train[[c for c in train.columns if c!='avclass']].values
        train_labels = np.array(train['avclass'].tolist())

        validation_vectors = validate[[c for c in validate.columns if c!='avclass']].values
        validation_labels = np.array(validate['avclass'].tolist())

        test_vectors = test[[c for c in test.columns if c!='avclass']].values
        test_labels = np.array(test['avclass'].tolist())

        data_dict = {}
        data_dict['train_vectors'] = train_vectors  
        data_dict['train_labels'] = train_labels  

        data_dict['validation_vectors'] = validation_vectors  
        data_dict['validation_labels'] = validation_labels 

        data_dict['test_vectors'] = test_vectors  
        data_dict['test_labels'] = test_labels 

        return data_dict


class MalMemInMem(object):
    """https://www.unb.ca/cic/datasets/malmem-2022.html"""

    def __init__(self,class_type='subavclass'):

        self.class_type = class_type 
        self.class_types = ['Class','avclass','subavclass']

    @classmethod
    def load(cls,data_path):
        return pd.read_csv(data_path,index_col='sha256')

    def load_single(self,data_path):
        data = pd.read_csv(data_path,index_col='sha256')
        vectors = data[[c for c in data.columns if (c not in self.class_types)]].values
        labels = np.array(data[self.class_type].tolist())
        index = data.index 
        return vectors,labels,index 

    def split_data(self,data,num_samples_train, num_samples_validate, num_samples_test):

        train, validate, test = split_dataframe(data, num_samples_train, num_samples_validate, num_samples_test)
        
        train_vectors = train[[c for c in train.columns if (c not in self.class_types)]].values
        train_labels = np.array(train[self.class_type].tolist())

        validation_vectors = validate[[c for c in validate.columns if (c not in self.class_types)]].values
        validation_labels = np.array(validate[self.class_type].tolist())

        test_vectors = test[[c for c in test.columns if (c not in self.class_types)]].values
        test_labels = np.array(test[self.class_type].tolist())

        data_dict = {}
        data_dict['train_vectors'] = train_vectors  
        data_dict['train_labels'] = train_labels  

        data_dict['validation_vectors'] = validation_vectors  
        data_dict['validation_labels'] = validation_labels 

        data_dict['test_vectors'] = test_vectors  
        data_dict['test_labels'] = test_labels 

        return data_dict



class PDFMalwareInMem(object):
    """https://www.unb.ca/cic/datasets/pdfmal-2022.html"""

    @classmethod
    def load(cls,data_path):
        d = pd.read_csv(data_path,index_col='sha256')
        d.rename(columns={'Class':'avclass'},inplace=True)
        return d

    @classmethod
    def load_single(cls,data_path):
        data = cls.load(data_path)
        vectors = data[[c for c in data.columns if c!='avclass']].values
        labels = np.array(data['avclass'].tolist())
        index = data.index 
        return vectors,labels,index 


    @classmethod
    def split_data(cls,data,num_samples_train, num_samples_validate, num_samples_test):

        train, validate, test = split_dataframe(data, num_samples_train, num_samples_validate, num_samples_test)
        
        train_vectors = train[[c for c in train.columns if c!='avclass']].values
        train_labels = np.array(train['avclass'].tolist())

        validation_vectors = validate[[c for c in validate.columns if c!='avclass']].values
        validation_labels = np.array(validate['avclass'].tolist())

        test_vectors = test[[c for c in test.columns if c!='avclass']].values
        test_labels = np.array(test['avclass'].tolist())

        data_dict = {}
        data_dict['train_vectors'] = train_vectors  
        data_dict['train_labels'] = train_labels  

        data_dict['validation_vectors'] = validation_vectors  
        data_dict['validation_labels'] = validation_labels 

        data_dict['test_vectors'] = test_vectors  
        data_dict['test_labels'] = test_labels 

        return data_dict



class CCCS(Dataset):
    def __init__(self,data,num_samples,split,train_proportion=0.8):

        self.features = data[[c for c in data.columns if c!='avclass']].values
        self.labels = np.array(data['avclass'].tolist())
        self.index = data.index.values

        self.train_proportion = train_proportion
        self.partition = np.floor(self.train_proportion*data.shape[0]).astype(int)
        self.split = split
        self.num_samples = min(num_samples,self.partition) if split=='train' else (min(num_samples,len(self.labels)-self.partition) if split=='test' else num_samples)
           
        if self.split=='distribution':

            self.features = self.features[:self.num_samples,:]
            self.labels = self.labels[:self.num_samples]
            self.index = self.index[:self.num_samples]

        elif self.split=='train':

            assert self.num_samples<=self.partition
            self.features = self.features[:self.num_samples,:]
            self.labels = self.labels[:self.num_samples]
            self.index = self.index[:self.num_samples]

        elif self.split=='test':

            assert self.num_samples<=(len(self.labels)-self.partition)
            self.features = self.features[self.partition:self.partition+self.num_samples,:]
            self.labels = self.labels[self.partition:self.partition+self.num_samples]
            self.index = self.index[self.partition:self.partition+self.num_samples]

    @classmethod
    def load(cls,data_path):
        return pd.read_csv(data_path,index_col='sha256')

    def __len__(self):
        return min(self.num_samples,self.features.shape[0])  

    def __getitem__(self,idx):
        return self.features[idx],self.labels[idx],self.index[idx]



class CICAndMal2017(Dataset):
    def __init__(self,data,num_samples,split,train_proportion=0.8):


        data = fill_missing_values(data)


        self.features = data[[c for c in data.columns if c!='avclass']]
        self.features = convert_string_to_int(self.features).values
        self.labels = np.array(data['avclass'].tolist())
        self.index = data.index.values
        self.index = range(data.shape[0])


        self.train_proportion = train_proportion
        self.partition = np.floor(self.train_proportion*data.shape[0]).astype(int)
        self.split = split
        self.num_samples = min(num_samples,self.partition) if split=='train' else (min(num_samples,len(self.labels)-self.partition) if split=='test' else num_samples)
           
        if self.split=='distribution':

            self.features = self.features[:self.num_samples,:]
            self.labels = self.labels[:self.num_samples]
            self.index = self.index[:self.num_samples]

        elif self.split=='train':

            assert self.num_samples<=self.partition
            self.features = self.features[:self.num_samples,:]
            self.labels = self.labels[:self.num_samples]
            self.index = self.index[:self.num_samples]

        elif self.split=='test':

            assert self.num_samples<=(len(self.labels)-self.partition)
            self.features = self.features[self.partition:self.partition+self.num_samples,:]
            self.labels = self.labels[self.partition:self.partition+self.num_samples]
            self.index = self.index[self.partition:self.partition+self.num_samples]

    @classmethod
    def load(cls,data_path):
        return pd.read_csv(data_path,index_col='Flow ID')

    def __len__(self):
        return min(self.num_samples,self.features.shape[0])  

    def __getitem__(self,idx):
        return self.features[idx],self.labels[idx],self.index[idx]




class SysCalls(Dataset):
    def __init__(self,data,num_samples,split,train_proportion=0.8):

        self.features = data[[c for c in data.columns if c!='avclass']].values
        self.labels = np.array(data['avclass'].tolist())
        self.index = data.index.values 

        self.train_proportion = train_proportion
        self.partition = np.floor(self.train_proportion*data.shape[0]).astype(int)
        self.split = split
        self.num_samples = min(num_samples,self.partition) if split=='train' else (min(num_samples,len(self.labels)-self.partition) if split=='test' else num_samples)
           
        if self.split=='distribution':

            self.features = self.features[:self.num_samples,:]
            self.labels = self.labels[:self.num_samples]
            self.index = self.index[:self.num_samples]

        elif self.split=='train':

            assert self.num_samples<=self.partition
            self.features = self.features[:self.num_samples,:]
            self.labels = self.labels[:self.num_samples]
            self.index = self.index[:self.num_samples]

        elif self.split=='test':

            assert self.num_samples<=(len(self.labels)-self.partition)
            self.features = self.features[self.partition:self.partition+self.num_samples,:]
            self.labels = self.labels[self.partition:self.partition+self.num_samples]
            self.index = self.index[self.partition:self.partition+self.num_samples]

    @classmethod
    def load(cls,data_path):
        return pd.read_csv(data_path,index_col=None)

    def __len__(self):
        return min(self.num_samples,self.features.shape[0])  

    def __getitem__(self,idx):
        return self.features[idx],self.labels[idx],self.index[idx]



class SysCallsBinders(Dataset):
    def __init__(self,data,num_samples,split,train_proportion=0.8):

        self.features = data[[c for c in data.columns if c!='avclass']].values
        self.labels = np.array(data['avclass'].tolist())
        self.index = data.index.values 

        self.train_proportion = train_proportion
        self.partition = np.floor(self.train_proportion*data.shape[0]).astype(int)
        self.split = split
        self.num_samples = min(num_samples,self.partition) if split=='train' else (min(num_samples,len(self.labels)-self.partition) if split=='test' else num_samples)
           
        if self.split=='distribution':

            self.features = self.features[:self.num_samples,:]
            self.labels = self.labels[:self.num_samples]
            self.index = self.index[:self.num_samples]

        elif self.split=='train':

            assert self.num_samples<=self.partition
            self.features = self.features[:self.num_samples,:]
            self.labels = self.labels[:self.num_samples]
            self.index = self.index[:self.num_samples]

        elif self.split=='test':

            assert self.num_samples<=(len(self.labels)-self.partition)
            self.features = self.features[self.partition:self.partition+self.num_samples,:]
            self.labels = self.labels[self.partition:self.partition+self.num_samples]
            self.index = self.index[self.partition:self.partition+self.num_samples]

    @classmethod
    def load(cls,data_path):
        return pd.read_csv(data_path,index_col=None)

    def __len__(self):
        return min(self.num_samples,self.features.shape[0])  

    def __getitem__(self,idx):
        return self.features[idx],self.labels[idx],self.index[idx]




class MalMem(Dataset):
    def __init__(self,data,num_samples,split,train_proportion=0.8,class_type='avclass'):

        self.class_type = class_type 
        self.class_types = ['Class','avclass','subavclass']

        data = fill_missing_values(data)

        self.features = data[[c for c in data.columns if c not in self.class_types]]
        self.features = convert_string_to_int(self.features).values
        self.labels = np.array(data['avclass'].tolist())
        self.index = data.index.values

        self.train_proportion = train_proportion
        self.partition = np.floor(self.train_proportion*data.shape[0]).astype(int)
        self.split = split
        self.num_samples = min(num_samples,self.partition) if split=='train' else (min(num_samples,len(self.labels)-self.partition) if split=='test' else num_samples)
           
        if self.split=='distribution':

            self.features = self.features[:self.num_samples,:]
            self.labels = self.labels[:self.num_samples]
            self.index = self.index[:self.num_samples]

        elif self.split=='train':

            assert self.num_samples<=self.partition
            self.features = self.features[:self.num_samples,:]
            self.labels = self.labels[:self.num_samples]
            self.index = self.index[:self.num_samples]

        elif self.split=='test':

            assert self.num_samples<=(len(self.labels)-self.partition)
            self.features = self.features[self.partition:self.partition+self.num_samples,:]
            self.labels = self.labels[self.partition:self.partition+self.num_samples]
            self.index = self.index[self.partition:self.partition+self.num_samples]

    @classmethod
    def load(cls,data_path):
        return pd.read_csv(data_path,index_col='sha256')

    def __len__(self):
        return min(self.num_samples,self.features.shape[0])  

    def __getitem__(self,idx):
        return self.features[idx],self.labels[idx],self.index[idx]


class PDFMalware(Dataset):
    def __init__(self,data,num_samples,split,train_proportion=0.8):

        data.rename(columns={'Class':'avclass'},inplace=True)

        self.features = data[[c for c in data.columns if c!='avclass']].values
        self.labels = np.array(data['avclass'].tolist())
        self.index = data.index.values 

        self.train_proportion = train_proportion
        self.partition = np.floor(self.train_proportion*data.shape[0]).astype(int)
        self.split = split
        self.num_samples = min(num_samples,self.partition) if split=='train' else (min(num_samples,len(self.labels)-self.partition) if split=='test' else num_samples)
           
        if self.split=='distribution':

            self.features = self.features[:self.num_samples,:]
            self.labels = self.labels[:self.num_samples]
            self.index = self.index[:self.num_samples]

        elif self.split=='train':

            assert self.num_samples<=self.partition
            self.features = self.features[:self.num_samples,:]
            self.labels = self.labels[:self.num_samples]
            self.index = self.index[:self.num_samples]

        elif self.split=='test':

            assert self.num_samples<=(len(self.labels)-self.partition)
            self.features = self.features[self.partition:self.partition+self.num_samples,:]
            self.labels = self.labels[self.partition:self.partition+self.num_samples]
            self.index = self.index[self.partition:self.partition+self.num_samples]

    @classmethod
    def load(cls,data_path):
        return pd.read_csv(data_path,index_col='sha256')

    def __len__(self):
        return min(self.num_samples,self.features.shape[0])  

    def __getitem__(self,idx):
        return self.features[idx],self.labels[idx],self.index[idx]


class EmbeddingsDataset(Dataset):
    def __init__(self,data,num_samples):

        self.features = data[[c for c in data.columns if c!='label']].values.astype('float32')
        self.features = self.features[:num_samples,:]
        self.labels = data['label'].values[:num_samples]
        self.index = data.index.values[:num_samples]
        self.num_samples = num_samples

    @classmethod
    def load(cls,data_path):
        return pd.read_csv(data_path,index_col='Index')

    def __len__(self):
        return self.features.shape[0] 

    def __getitem__(self,idx):
        return self.features[idx],self.labels[idx],self.index[idx]


class MNISTDataset(Dataset):
    def __init__(self,data,num_samples,split,train_proportion=0.8):

        self.features = data[[c for c in data.columns if c!='label']].values.astype('float32')
        self.labels = data['label'].values
        self.index = data.index.values 

        self.train_proportion = train_proportion
        self.partition = np.floor(self.train_proportion*data.shape[0]).astype(int)
        self.split = split
        self.num_samples = min(num_samples,self.partition) if split=='train' else (min(num_samples,len(self.labels)-self.partition) if split=='test' else num_samples)
           
        if self.split=='distribution':

            self.features = self.features[:self.num_samples,:]
            self.labels = self.labels[:self.num_samples]
            self.index = self.index[:self.num_samples]

        elif self.split=='train':

            assert self.num_samples<=self.partition
            self.features = self.features[:self.num_samples,:]
            self.labels = self.labels[:self.num_samples]
            self.index = self.index[:self.num_samples]

        elif self.split=='test':

            assert self.num_samples<=(len(self.labels)-self.partition)
            self.features = self.features[self.partition:self.partition+self.num_samples,:]
            self.labels = self.labels[self.partition:self.partition+self.num_samples]
            self.index = self.index[self.partition:self.partition+self.num_samples]

    @classmethod
    def load(cls,data_path):
        return pd.read_csv(data_path)

    def __len__(self):
        return min(self.num_samples,self.features.shape[0])  

    def __getitem__(self,idx):
        return self.features[idx],self.labels[idx],self.index[idx]


class FashionMNISTDataset(Dataset):
    def __init__(self,data,num_samples,split,train_proportion=0.8):

        self.features = data[[c for c in data.columns if c!='label']].values.astype('float32')
        self.labels = data['label'].values
        self.index = data.index.values 

        self.train_proportion = train_proportion
        self.partition = np.floor(self.train_proportion*data.shape[0]).astype(int)
        self.split = split
        self.num_samples = min(num_samples,self.partition) if split=='train' else (min(num_samples,len(self.labels)-self.partition) if split=='test' else num_samples)
           
        if self.split=='distribution':

            self.features = self.features[:self.num_samples,:]
            self.labels = self.labels[:self.num_samples]
            self.index = self.index[:self.num_samples]

        elif self.split=='train':

            assert self.num_samples<=self.partition
            self.features = self.features[:self.num_samples,:]
            self.labels = self.labels[:self.num_samples]
            self.index = self.index[:self.num_samples]

        elif self.split=='test':

            assert self.num_samples<=(len(self.labels)-self.partition)
            self.features = self.features[self.partition:self.partition+self.num_samples,:]
            self.labels = self.labels[self.partition:self.partition+self.num_samples]
            self.index = self.index[self.partition:self.partition+self.num_samples]

    @classmethod
    def load(cls,data_path):
        return pd.read_csv(data_path)

    def __len__(self):
        return min(self.num_samples,self.features.shape[0])  

    def __getitem__(self,idx):
        return self.features[idx],self.labels[idx],self.index[idx]



class Chars74kDataset(Dataset):
    def __init__(self,data,num_samples,split,train_proportion=0.8):

        self.features = data[[c for c in data.columns if c!='label']].values.astype('float32')
        self.labels = data['label'].values
        self.index = data.index.values 

        self.train_proportion = train_proportion
        self.partition = np.floor(self.train_proportion*data.shape[0]).astype(int)
        self.split = split
        self.num_samples = min(num_samples,self.partition) if split=='train' else (min(num_samples,len(self.labels)-self.partition) if split=='test' else num_samples)
           
        if self.split=='distribution':

            self.features = self.features[:self.num_samples,:]
            self.labels = self.labels[:self.num_samples]
            self.index = self.index[:self.num_samples]

        elif self.split=='train':

            assert self.num_samples<=self.partition
            self.features = self.features[:self.num_samples,:]
            self.labels = self.labels[:self.num_samples]
            self.index = self.index[:self.num_samples]

        elif self.split=='test':

            assert self.num_samples<=(len(self.labels)-self.partition)
            self.features = self.features[self.partition:self.partition+self.num_samples,:]
            self.labels = self.labels[self.partition:self.partition+self.num_samples]
            self.index = self.index[self.partition:self.partition+self.num_samples]

    @classmethod
    def load(cls,data_path):
        d = pd.read_csv(data_path)
        e = 1.-d[[c for c in d.columns if c!='label']]
        e['label'] = d['label']
        return e 

    def __len__(self):
        return min(self.num_samples,self.features.shape[0])  

    def __getitem__(self,idx):
        return self.features[idx],self.labels[idx],self.index[idx]



class NotMNISTDataset(Dataset):
    def __init__(self,data,num_samples,split,train_proportion=0.8):

        self.features = data[[c for c in data.columns if c!='label']].values.astype('float32')
        self.labels = data['label'].values
        self.index = data.index.values 

        self.train_proportion = train_proportion
        self.partition = np.floor(self.train_proportion*data.shape[0]).astype(int)
        self.split = split
        self.num_samples = min(num_samples,self.partition) if split=='train' else (min(num_samples,len(self.labels)-self.partition) if split=='test' else num_samples)
           
        if self.split=='distribution':

            self.features = self.features[:self.num_samples,:]
            self.labels = self.labels[:self.num_samples]
            self.index = self.index[:self.num_samples]

        elif self.split=='train':

            assert self.num_samples<=self.partition
            self.features = self.features[:self.num_samples,:]
            self.labels = self.labels[:self.num_samples]
            self.index = self.index[:self.num_samples]

        elif self.split=='test':

            assert self.num_samples<=(len(self.labels)-self.partition)
            self.features = self.features[self.partition:self.partition+self.num_samples,:]
            self.labels = self.labels[self.partition:self.partition+self.num_samples]
            self.index = self.index[self.partition:self.partition+self.num_samples]

    @classmethod
    def load(cls,data_path):
        return pd.read_csv(data_path)

    def __len__(self):
        return min(self.num_samples,self.features.shape[0])  

    def __getitem__(self,idx):
        return self.features[idx],self.labels[idx],self.index[idx]



class QuickdrawDataset(Dataset):
    def __init__(self,data,num_samples,split,train_proportion=0.8):

        self.features = (1./255)*data[[c for c in data.columns if c!='label']].values.astype('float32')
        self.labels = data['label'].values
        self.index = data.index.values 

        self.train_proportion = train_proportion
        self.partition = np.floor(self.train_proportion*data.shape[0]).astype(int)
        self.split = split
        self.num_samples = min(num_samples,self.partition) if split=='train' else (min(num_samples,len(self.labels)-self.partition) if split=='test' else num_samples)
           
        if self.split=='distribution':

            self.features = self.features[:self.num_samples,:]
            self.labels = self.labels[:self.num_samples]
            self.index = self.index[:self.num_samples]

        elif self.split=='train':

            assert self.num_samples<=self.partition
            self.features = self.features[:self.num_samples,:]
            self.labels = self.labels[:self.num_samples]
            self.index = self.index[:self.num_samples]

        elif self.split=='test':

            assert self.num_samples<=(len(self.labels)-self.partition)
            self.features = self.features[self.partition:self.partition+self.num_samples,:]
            self.labels = self.labels[self.partition:self.partition+self.num_samples]
            self.index = self.index[self.partition:self.partition+self.num_samples]

    @classmethod
    def load(cls,data_path):
        return pd.read_csv(data_path)

    def __len__(self):
        return min(self.num_samples,self.features.shape[0])  

    def __getitem__(self,idx):
        return self.features[idx],self.labels[idx],self.index[idx]



class SLMNISTDataset(Dataset):
    def __init__(self,data,num_samples,split,train_proportion=0.8):

        self.features = (1./255)*data[[c for c in data.columns if c!='label']].values.astype('float32')
        self.labels = data['label'].values
        self.index = data.index.values 

        self.train_proportion = train_proportion
        self.partition = np.floor(self.train_proportion*data.shape[0]).astype(int)
        self.split = split
        self.num_samples = min(num_samples,self.partition) if split=='train' else (min(num_samples,len(self.labels)-self.partition) if split=='test' else num_samples)
           
        if self.split=='distribution':

            self.features = self.features[:self.num_samples,:]
            self.labels = self.labels[:self.num_samples]
            self.index = self.index[:self.num_samples]

        elif self.split=='train':

            assert self.num_samples<=self.partition
            self.features = self.features[:self.num_samples,:]
            self.labels = self.labels[:self.num_samples]
            self.index = self.index[:self.num_samples]

        elif self.split=='test':

            assert self.num_samples<=(len(self.labels)-self.partition)
            self.features = self.features[self.partition:self.partition+self.num_samples,:]
            self.labels = self.labels[self.partition:self.partition+self.num_samples]
            self.index = self.index[self.partition:self.partition+self.num_samples]

    @classmethod
    def load(cls,data_path):
        return pd.read_csv(data_path)

    def __len__(self):
        return min(self.num_samples,self.features.shape[0])  

    def __getitem__(self,idx):
        return self.features[idx],self.labels[idx],self.index[idx]


class KuzushijiDataset(Dataset):
    def __init__(self,data,num_samples,split,train_proportion=0.8):

        self.features = data[[c for c in data.columns if c!='label']].values.astype('float32')
        self.labels = data['label'].values
        self.index = data.index.values 

        self.train_proportion = train_proportion
        self.partition = np.floor(self.train_proportion*data.shape[0]).astype(int)
        self.split = split
        self.num_samples = min(num_samples,self.partition) if split=='train' else (min(num_samples,len(self.labels)-self.partition) if split=='test' else num_samples)
           
        if self.split=='distribution':

            self.features = self.features[:self.num_samples,:]
            self.labels = self.labels[:self.num_samples]
            self.index = self.index[:self.num_samples]

        elif self.split=='train':

            assert self.num_samples<=self.partition
            self.features = self.features[:self.num_samples,:]
            self.labels = self.labels[:self.num_samples]
            self.index = self.index[:self.num_samples]

        elif self.split=='test':

            assert self.num_samples<=(len(self.labels)-self.partition)
            self.features = self.features[self.partition:self.partition+self.num_samples,:]
            self.labels = self.labels[self.partition:self.partition+self.num_samples]
            self.index = self.index[self.partition:self.partition+self.num_samples]

    @classmethod
    def load(cls,data_path):
        return pd.read_csv(data_path)

    def __len__(self):
        return min(self.num_samples,self.features.shape[0])  

    def __getitem__(self,idx):
        return self.features[idx],self.labels[idx],self.index[idx]



class EMNISTDataset(Dataset):
    def __init__(self,data,num_samples,split,train_proportion=0.8):

        self.features = data[[c for c in data.columns if c!='label']].values.astype('float32')
        self.labels = data['label'].values
        self.index = data.index.values 

        self.train_proportion = train_proportion
        self.partition = np.floor(self.train_proportion*data.shape[0]).astype(int)
        self.split = split
        self.num_samples = min(num_samples,self.partition) if split=='train' else (min(num_samples,len(self.labels)-self.partition) if split=='test' else num_samples)
           
        if self.split=='distribution':

            self.features = self.features[:self.num_samples,:]
            self.labels = self.labels[:self.num_samples]
            self.index = self.index[:self.num_samples]

        elif self.split=='train':

            assert self.num_samples<=self.partition
            self.features = self.features[:self.num_samples,:]
            self.labels = self.labels[:self.num_samples]
            self.index = self.index[:self.num_samples]

        elif self.split=='test':

            assert self.num_samples<=(len(self.labels)-self.partition)
            self.features = self.features[self.partition:self.partition+self.num_samples,:]
            self.labels = self.labels[self.partition:self.partition+self.num_samples]
            self.index = self.index[self.partition:self.partition+self.num_samples]

    @classmethod
    def load(cls,data_path):
        return pd.read_csv(data_path)

    def __len__(self):
        return min(self.num_samples,self.features.shape[0])  

    def __getitem__(self,idx):
        return self.features[idx],self.labels[idx],self.index[idx]



class CIFAR10Dataset(Dataset):
    def __init__(self,data,num_samples,split,train_proportion=0.8):

        self.features = data[[c for c in data.columns if c!='label']].values.astype('float32')
        self.labels = data['label'].values
        self.index = data.index.values 

        self.train_proportion = train_proportion
        self.partition = np.floor(self.train_proportion*data.shape[0]).astype(int)
        self.split = split
        self.num_samples = min(num_samples,self.partition) if split=='train' else (min(num_samples,len(self.labels)-self.partition) if split=='test' else num_samples)
           
        if self.split=='distribution':

            self.features = self.features[:self.num_samples,:]
            self.labels = self.labels[:self.num_samples]
            self.index = self.index[:self.num_samples]

        elif self.split=='train':

            assert self.num_samples<=self.partition
            self.features = self.features[:self.num_samples,:]
            self.labels = self.labels[:self.num_samples]
            self.index = self.index[:self.num_samples]

        elif self.split=='test':

            assert self.num_samples<=(len(self.labels)-self.partition)
            self.features = self.features[self.partition:self.partition+self.num_samples,:]
            self.labels = self.labels[self.partition:self.partition+self.num_samples]
            self.index = self.index[self.partition:self.partition+self.num_samples]

    @classmethod
    def load(cls,data_path):
        return pd.read_csv(data_path)

    def __len__(self):
        return min(self.num_samples,self.features.shape[0])  

    def __getitem__(self,idx):
        return self.features[idx],self.labels[idx],self.index[idx]


class EmberDataset(Dataset):
    def __init__(self,data,num_samples,split,train_proportion=0.8):

        self.features = data[[c for c in data.columns if c!='avclass']].values
        self.labels = np.array(data['avclass'].tolist())
        self.index = data.index.values 

        self.train_proportion = train_proportion
        self.partition = np.floor(self.train_proportion*data.shape[0]).astype(int)
        self.split = split
        self.num_samples = min(num_samples,self.partition) if split=='train' else (min(num_samples,len(self.labels)-self.partition) if split=='test' else num_samples)
           
        if self.split=='distribution':

            self.features = self.features[:self.num_samples,:]
            self.labels = self.labels[:self.num_samples]
            self.index = self.index[:self.num_samples]

        elif self.split=='train':

            assert self.num_samples<=self.partition
            self.features = self.features[:self.num_samples,:]
            self.labels = self.labels[:self.num_samples]
            self.index = self.index[:self.num_samples]

        elif self.split=='test':

            assert self.num_samples<=(len(self.labels)-self.partition)
            self.features = self.features[self.partition:self.partition+self.num_samples,:]
            self.labels = self.labels[self.partition:self.partition+self.num_samples]
            self.index = self.index[self.partition:self.partition+self.num_samples]

    @classmethod
    def load(cls,data_path):
        return pd.read_csv(data_path,index_col='sha256')

    def __len__(self):
        return min(self.num_samples,self.features.shape[0])  

    def __getitem__(self,idx):
        return self.features[idx],self.labels[idx],self.index[idx]



class SorelDataset(Dataset):
    def __init__(self,data,num_samples,split,train_proportion=0.8):

        self.features = data[[c for c in data.columns if c!='avclass']].values
        self.labels = np.array(data['avclass'].tolist())
        self.index = data.index.values 

        self.train_proportion = train_proportion
        self.partition = np.floor(self.train_proportion*data.shape[0]).astype(int)
        self.split = split
        self.num_samples = min(num_samples,self.partition) if split=='train' else (min(num_samples,len(self.labels)-self.partition) if split=='test' else num_samples)
           
        if self.split=='distribution':

            self.features = self.features[:self.num_samples,:]
            self.labels = self.labels[:self.num_samples]
            self.index = self.index[:self.num_samples]

        elif self.split=='train':

            assert self.num_samples<=self.partition
            self.features = self.features[:self.num_samples,:]
            self.labels = self.labels[:self.num_samples]
            self.index = self.index[:self.num_samples]

        elif self.split=='test':

            assert self.num_samples<=(len(self.labels)-self.partition)
            self.features = self.features[self.partition:self.partition+self.num_samples,:]
            self.labels = self.labels[self.partition:self.partition+self.num_samples]
            self.index = self.index[self.partition:self.partition+self.num_samples]

    @classmethod
    def load(cls,data_path):
        return pd.read_csv(data_path,index_col='sha256')

    @classmethod
    def load_and_transform(cls,data_path):
        families = ['adware', 'flooder', 'ransomware', 'dropper', 'spyware', 'packed',
                    'crypto_miner', 'file_infector', 'installer', 'worm', 'downloader']
        data = pd.read_csv(data_path,index_col='sha256')
        data = data.loc[(data.packed==0) & (data.is_malware==1)]
        family_columns = data[families]
        family_column = family_columns[families].idxmax(axis=1)
        data = data[[c for c in data.columns if c not in families]]
        data = data[[c for c in data.columns if c!='is_malware']]
        data['avclass'] = family_column    
        return data     

    def __len__(self):
        return min(self.num_samples,self.features.shape[0])  

    def __getitem__(self,idx):
        return self.features[idx],self.labels[idx],self.index[idx]



if __name__=='__main__':

    from pathlib import Path   
    import os     

    def transform_sorel(data_path,save_dir):
        families = ['adware', 'flooder', 'ransomware', 'dropper', 'spyware', 'packed',
                            'crypto_miner', 'file_infector', 'installer', 'worm', 'downloader']
        data = pd.read_csv(data_path,index_col='sha256')
        data = data.loc[(data.packed==0) & (data.is_malware==1)]
        family_column = data[families].idxmax(axis=1)
        data = data[[c for c in data.columns if c not in families]]
        data = data[[c for c in data.columns if c!='is_malware']]
        data['avclass'] = family_column    
        data.to_csv(save_dir / str(data_path).split('/')[-1])  
        os.remove(data_path)


    data_dir = Path('../../../sandbox/data/sorel/')
    save_dir = Path('../../../sandbox/data/sorel/transformed')

    filepaths = data_dir.glob('*.csv')
    for filepath in filepaths:
        transform_sorel(filepath,save_dir)























