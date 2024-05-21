import glob 
import pandas as pd  
from pathlib import Path      



if __name__=='__main__':

	csv_dir = Path('../../../../sandbox/data/MalwareDatasets/MalwareDatasets/MalMem/')
	save_dir = Path('../../../../sandbox/data/MalwareDatasets/MalwareDatasets/MalMem/total/')

	frames = []
	filepaths = list(csv_dir.glob('*.csv'))#[:3]
	for filepath in filepaths:

		name = str(filepath).split('/')[-1].split('.')[0]
		print(f'Processing: {name}')
		d = pd.read_csv(filepath)
		d = d.loc[d.Category!='Benign']
		new_cols = d['Category'].str.split('-', expand=True)
		d['avclass'] = new_cols[0]
		d['subavclass'] = new_cols[1]
		d.index = new_cols[2]
		d.index.name='sha256'
		d.drop('Category',axis=1,inplace=True)

		d.to_csv(save_dir / 'Obfuscated-MalMem2022.csv')




