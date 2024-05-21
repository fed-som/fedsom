import numpy as np  
import pandas as pd    



def chunk_dataframe(df,chunk_size):
	num_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size !=0 else 0)
	for i in range(num_chunks):
		start_idx = i*chunk_size  
		end_idx = (i+1)*chunk_size   
		yield df.iloc[start_idx:end_idx]



def load_and_extract_npz(filepath):

	npz = np.load(filepath)

	return npz['arr_0'],npz['arr_1']



if __name__=='__main__':

	names = ['train','test','validation']

	total = 0
	for name in names:
		filepath = f'{name}-features.npz'
		vectors,labels = load_and_extract_npz(filepath)
		vectors = pd.DataFrame(vectors)
		print(vectors.shape)
		total+=vectors.shape[0]

	print(f'total: {total}')




	# chunk_size = 1000
	# for i,vectors_chunk in enumerate(chunk_dataframe(vectors,chunk_size)):
	# 	vectors_chunk.to_csv(save_dir / f'sorel20m_vectors_{i}.csv')