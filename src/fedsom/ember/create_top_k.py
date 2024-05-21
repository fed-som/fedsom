import pandas as pd   
from pathlib import Path    



def read_ember_frame(filepath):

	return pd.read_csv(filepath,index_col='sha256')


def create_top_k(frame,column,k):

	counts = frame[column].value_counts()
	top_k = counts.head(k).index.tolist()
	return frame[frame[column].isin(top_k)]

def shuffle(df):

	df = df.sample(frac=1,random_state=42)
	return df    

def save_top_k_frame(top_k_frame,k,save_dir):

	top_k.to_csv(save_dir/ f'ember_top_{k}.csv')



if __name__=='__main__':



	filepath = '../../../sandbox/data/ember/top_k/ember.csv'
	save_dir = Path('../../../sandbox/data/ember/top_k/')

	ember = read_ember_frame(filepath)
	for k in [5,10,15,20]:
	# k = 10
		
		top_k = create_top_k(ember,'avclass',k)
		top_k = shuffle(top_k)
		save_top_k_frame(top_k,k,save_dir)
		print(k)

