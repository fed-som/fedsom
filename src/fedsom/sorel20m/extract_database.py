import sqlite3
import pandas as pd
import argparse
from pathlib import Path 
import lmdb   
import msgpack
import zlib
import boto3
import pandas as pd
from pathlib import Path    
import numpy as np
from io import StringIO
from multiprocessing import Pool   



def load_db_to_dataframe(db_filepath):

    conn = sqlite3.connect(db_filepath)
    name = str(db_filepath).split('/')[-1].split('.')[0]
    query = f"SELECT * FROM {name};"
    df = pd.read_sql_query(query, conn)
    conn.close()
   
    return df


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class BotoWrapper(object):

    def __init__(self,bucket_name,sub_bucket,vector_name):

        self.bucket_name = bucket_name 
        self.sub_bucket = sub_bucket   
        self.vector_name = vector_name

    def send(self,vectors,index):

        csv_buffer = StringIO()
        vectors.to_csv(csv_buffer, index=True)
        csv_content = csv_buffer.getvalue()
        s3 = boto3.client('s3')
        file_name = str(self.sub_bucket / f'{self.vector_name}_{index}.csv')
        s3.put_object(Body=csv_content, Bucket=self.bucket_name, Key=file_name)

        print(f'{file_name} uploaded to {self.bucket_name} in S3.')


def send_vectors_to_s3_helper(features_db,meta_df,boto_wrapper,n,sha_chunk):

    fv_batch = []
    sha_batch = []
    for sha in sha_chunk:

        with features_db.begin() as txn:
            x = txn.get(sha.encode('ascii'))
        if x is not None:
            v = np.array(msgpack.loads(zlib.decompress(x),strict_map_key=False)[0])
            fv_batch.append(v)
            sha_batch.append(sha)

    fv_batch_df = pd.DataFrame(fv_batch,index=sha_batch)
    fv_batch_df.index.name = 'sha256'
    meta_batch = meta_df.loc[sha_batch]
    meta_vectors = pd.merge(meta_batch, fv_batch_df, how='left', on='sha256')
    boto_wrapper.send(meta_vectors,n)


def send_vectors_to_s3(features_db,meta_df,boto_wrapper,chunk_size,parallel=False) -> None:

    shas = meta_df.index.values.tolist()
    chunked_shas = chunks(shas,chunk_size)
    if parallel:
        with Pool() as p:
            p.starmap(
                send_vectors_to_s3_helper,
                [[features_db,meta_df,boto_wrapper,n,sha_chunk] for n,sha_chunk in enumerate(chunked_shas)],
            )
        p.close()
        p.join()

    else:
        for n,sha_chunk in enumerate(chunked_shas):
            send_vectors_to_s3_helper(features_db,meta_df,boto_wrapper,n,sha_chunk)



if __name__=='__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("--db_filepath", type=Path, default="./meta.db") # s3://sorel-20m/09-DEC-2020/processed-data/meta.db
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--bucket_name", type=str, default='cs-prod-dsci-west1')
    parser.add_argument("--sub_bucket", type=Path, default='science_data/users/mslawinski/sorel20m/vectors/')
    parser.add_argument("--vector_name", type=str, default='sorel20m')
    parser.add_argument("--features_path", type=str, default='./tmp') # assumes s3://sorel-20m/09-DEC-2020/processed-data/ember_features/data.mdb sits in ./tmp locally

    args = parser.parse_args()
    db_filepath = args.db_filepath
    chunk_size = args.chunk_size
    bucket_name = args.bucket_name
    sub_bucket = args.sub_bucket
    vector_name = args.vector_name
    features_path = args.features_path    

    boto_wrapper = BotoWrapper(bucket_name,sub_bucket,vector_name)
    features_db = lmdb.open(features_path, readonly=True, map_size=int(1e13), max_readers=1024)
    meta_df = load_db_to_dataframe(db_filepath)
    meta_df_columns = ['sha256',
                       'is_malware',
                       'adware',
                       'flooder',
                       'ransomware',
                       'dropper',
                       'spyware',
                       'packed',
                       'crypto_miner',
                       'file_infector',
                       'installer',
                       'worm',
                       'downloader']
    meta_df = meta_df[meta_df_columns]
    meta_df.set_index('sha256',inplace=True,drop=True)
    send_vectors_to_s3(features_db,meta_df,boto_wrapper,chunk_size)










































