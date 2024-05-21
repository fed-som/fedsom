import glob 
import pandas as pd  
from pathlib import Path      
import os     



def list_subdirs(directory):

	return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory,d))]



if __name__=='__main__':

	csv_dir = Path('../../../../sandbox/data/MalwareDatasets/MalwareDatasets/CICAndMal2017/')
	save_dir = Path('../../../../sandbox/data/MalwareDatasets/MalwareDatasets/CICAndMal2017/total/')

	frames = []
	subdirs = list_subdirs(csv_dir)
	for subdir in subdirs:
		if subdir!='Benign' and subdir!='total':
			subsubdirs = list_subdirs(os.path.join(csv_dir,subdir))
			for subsubdir in subsubdirs:
				path = os.path.join(csv_dir,subdir,subsubdir)
				csv_files = glob.glob(f'{path}/*.csv')
				for f in csv_files:
					frames.append(pd.read_csv(f,index_col='Flow ID'))
		
	full_dataset = pd.concat(frames,axis=0)
	full_dataset.rename(columns={' Label':'avclass'},inplace=True)
	full_dataset = full_dataset.sample(frac=1,replace=False,random_state=42)
	print(full_dataset)
	full_dataset.to_csv(save_dir / 'cic_andmal2017.csv')



	# d = pd.read_csv(save_dir / 'cic_andmal2017.csv',index_col='Flow ID')
	# print(d)