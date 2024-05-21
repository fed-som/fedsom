import glob 
import pandas as pd  
from pathlib import Path      





if __name__=='__main__':

	csv_dir = Path('../../../../sandbox/data/MalwareDatasets/MalwareDatasets/CCCS-CIC-AndMal2020/')
	save_dir = Path('../../../../sandbox/data/MalwareDatasets/MalwareDatasets/CCCS-CIC-AndMal2020/total/')

	frames = []
	filepaths = list(csv_dir.glob('*.csv'))#[:3]
	for filepath in filepaths:

		name = str(filepath).split('/')[-1].split('.')[0]
		print(f'Processing: {name}')
		if 'Ben' not in name:	
			d = pd.read_csv(filepath,header=None)
			d['avclass'] = name
			d.set_index(d.columns[0],inplace=True,drop=True)
			frames.append(d)

	full_dataset = pd.concat(frames,axis=0)
	full_dataset = full_dataset.sample(frac=1,replace=False,random_state=42)

	full_dataset.index.name = 'sha256'
	full_dataset.to_csv(save_dir / 'cccs_cic_andmal2020.csv')

	# test = pd.read_csv('../../../../sandbox/data/MalwareDatasets/MalwareDatasets/CCCS-CIC-AndMal2020/cccs_cic_andmal2020.csv',index_col='sha256')
	
