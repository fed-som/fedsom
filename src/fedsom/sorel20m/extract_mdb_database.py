

from torch.utils import data
import lmdb
import sqlite3
import baker
import msgpack
import zlib
import numpy as np
import os
import tqdm
from logzero import logger

import config
import json

class LMDBReader(object):

    def __init__(self, path, postproc_func=None):
        self.env = lmdb.open(path, readonly=True, map_size=1e13, max_readers=1024)
        self.postproc_func = postproc_func

    def __call__(self, key):
        with self.env.begin() as txn:
            x = txn.get(key.encode('ascii'))
        if x is None:return None
        x = msgpack.loads(zlib.decompress(x),strict_map_key=False)
        if self.postproc_func is not None:
            x = self.postproc_func(x)
        return x




if __name__=='__main__':

	r = LMDBReader()
	








import pyodbc
import pandas as pd

# Define the connection string
conn_str = r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=data.mdb;"

# Establish a connection to the MDB file
conn = pyodbc.connect(conn_str)

# Define your SQL query
sql_query = 'SELECT * FROM data'

# Use pandas to read the data from the database
df = pd.read_sql(sql_query, conn)

# Close the connection
conn.close()

# Now 'df' contains your data as a pandas DataFrame
print(df)




import csv, pyodbc

# set up some constants
MDB = 'data.mdb'
DRV = '{Microsoft Access Driver (*.mdb, *.accdb)}'
PWD = 'pwd'

# connect to db
con = pyodbc.connect('DRIVER={};DBQ={};PWD={}'.format(DRV,MDB,PWD))
cur = con.cursor()

# run a query and get the results 
SQL = 'SELECT * FROM data;' # your query goes here
rows = cur.execute(SQL).fetchall()
cur.close()
con.close()

# you could change the mode from 'w' to 'a' (append) for any subsequent queries
with open('mdb.csv', 'w') as fou:
    csv_writer = csv.writer(fou) # default field-delimiter is ","
    csv_writer.writerows(rows)
























