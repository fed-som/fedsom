

import ember
import numpy as np  
import os 
import pandas as pd    
from pathlib import Path 

def chunk_dataframe(df,chunk_size):
	num_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size !=0 else 0)
	for i in range(num_chunks):
		start_idx = i*chunk_size  
		end_idx = (i+1)*chunk_size   
		yield df.iloc[start_idx:end_idx]


data_dir = "./data/ember2018"
save_dir = Path("./data/vectors/")
if not os.path.exists(save_dir):
	os.makedirs(save_dir)

X_train, y_train, X_test, y_test = ember.read_vectorized_features("./data/ember2018/")
X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)
X_test = pd.DataFrame(X_test)
y_test = pd.DataFrame(y_test)

metadata_dataframe = ember.read_metadata("./data/ember2018/")
malicious_idx = (metadata_dataframe['label']==1) & (~metadata_dataframe['avclass'].isna())
malicious_idx = malicious_idx.values

metadata_dataframe.set_index('sha256',inplace=True)
family_names = metadata_dataframe['avclass'][malicious_idx]
family_names.columns = ['avclass']
family_names = pd.DataFrame(family_names)

vectors = pd.concat([X_train,X_test])
vectors = vectors[malicious_idx]
vectors.index = family_names.index
vectors = pd.concat([family_names,vectors],axis=1)

for i,vectors_chunk in enumerate(chunk_dataframe(vectors,1000)):
	vectors_chunk.to_csv(save_dir / f'ember_malicious_vectors_{i}.csv')



