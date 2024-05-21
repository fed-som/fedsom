
import pandas as pd    
import numpy as np    
from scipy.stats import percentileofscore
import torch 
from deepclustering.encoder.utils.data_utils import recast_columns
from scipy.spatial.distance import pdist, squareform, cdist
from torch.utils.data import Dataset, DataLoader
import concurrent
import copy    
import matplotlib.pyplot as plt
import torch.nn.functional as F
import multiprocessing




class Metric(object):
    def __init__(self,cat_bool_index):

        self.cat_bool_index = cat_bool_index  

    def __call__(self,v,w):

        v_num,w_num = v[:,~self.cat_bool_index],w[:,~self.cat_bool_index]
        d_num = np.sum(np.abs(v_num[:,np.newaxis,:]-w_num),axis=-1)

        v_cat,w_cat = v[:,self.cat_bool_index],w[:,self.cat_bool_index]
        d_cat = (v_cat[:,np.newaxis,:]!=w_cat).astype(int).sum(axis=-1)

        return d_num+d_cat



class Categorical(object):

    def __init__(self, num_bool_index,coarseness=None,min_max=False):

        self.num_bool_index = num_bool_index
        self.coarseness = coarseness
        self.min_max = min_max

    def __call__(self,df):

        num_columns = df.columns[self.num_bool_index]
        if self.min_max:
            normalized = (df[num_columns] - df[num_columns].min())/(df[num_columns].max()-df[num_columns].min())
            normalized = 100*normalized
            normalized = np.round(normalized / self.coarseness) * self.coarseness
            df[num_columns] = normalized/100
        else:
            rank_df = df[num_columns].rank()
            percentile = 100*rank_df/len(df)
            percentile = np.round(percentile / self.coarseness) * self.coarseness
            df[num_columns] = percentile/100

        return df   




def plot_mnist_digits_side_by_side(vector1, vector2):

    image_size = int(np.sqrt(len(vector1)))
    mnist_digit_image1 = np.array(vector1).reshape(image_size, image_size)
    mnist_digit_image2 = np.array(vector2).reshape(image_size, image_size)

    # Create a subplot with 1 row and 2 columns
    plt.subplot(1, 2, 1)
    plt.imshow(mnist_digit_image1, cmap='gray')
    plt.title('MNIST Digit 1')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(mnist_digit_image2, cmap='gray')
    plt.title('MNIST Digit 2')
    plt.axis('off')

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Show the combined image
    plt.show(block=False)
    input('')
    plt.close()



def compute_dist_attention(batch,batch_aug,metric):

    distances = torch.tensor(cdist(batch.values,batch_aug.values,metric))
    return F.softmax(distances.view(-1),dim=0).view(distances.shape)




if __name__=='__main__':

    np.random.seed(0)






    batch = pd.DataFrame(np.array([[1,2,0],[1,3,0],[1,5,1]]))
    batch_aug = pd.DataFrame(np.array([[0,4,0],[2,3,0],[1,3,1]]))
    # print(batch)


    print('tensor')
    batch_tensor = copy.deepcopy(batch)
    batch_aug_tensor = copy.deepcopy(batch_aug)
    batch_tensor = torch.tensor(batch_tensor.values)
    batch_aug_tensor = torch.tensor(batch_aug_tensor.values)
    print(batch_tensor)
    print(batch_aug_tensor)
    print('---------------------\n')



    print('distance matrix')
    cat_bool_index = np.array([False, False, False])
    metric = Metric(cat_bool_index)
    dist_attention = compute_dist_attention(batch,batch_aug,metric)
    print(dist_attention)


    print('---------------------\n')



    print('similarity matrix')
    result = torch.matmul(batch_tensor, batch_aug_tensor.T)
    print(result)
    print('--------------------------\n')

    input('==========')


    # create fake, easy-to-understand batch 
    # transform rows via categorical 
    # compute contrastive sim matrix (pre exponential and log) on raw vectors 
    # compute custom sim matrix and compare 
    # make sure shape is the same, make sure diagonal is the same, and off diagonal is the same 
    # in terms of ordering 





# ==========================================================================================



    # v = np.random.randn(452)
    # # print(v)

    # integer = 5  
    # for x in v[:10]:

        
    #     value = percentileofscore(v,x)
    #     r = coarseness_multiple(value,integer)
    #     print(r)







    # cont_values = pd.DataFrame(np.random.randn(100,2))
    # cat_values = pd.DataFrame(np.random.choice(['a','b','c'],(100,2),replace=True))
    # cat_values.columns = [2,3]
    # df = pd.concat([cont_values,cat_values],axis=1)
    # df = recast_columns(df) 
    # cat_bool_index = np.array([True if str(df[c].dtype) == "category" else False for c in df.columns])

    # categorical = Categorical(df,cat_bool_index)
    # df = categorical(df)

    # metric = Metric(cat_bool_index)
    # dm = squareform(pdist(df, metric))
    # print(dm)



    values = pd.DataFrame([[1,1,0],[1,1,1],[1,1,5],[1,1,10]])


    index_pairs = []
    labels = [1,2,2,4]
    for i,x in enumerate(labels):
        for j,y in enumerate(labels[i+1:]):
            index_pairs.append([x,y])


    for idx in index_pairs:
        print(idx)




    # index_pairs = []
    # for i in range(4):
    #     for j in range(i+1,4):
    #         index_pairs.append([i,j])
    # for idx in index_pairs:
    #     print(idx)
    # print(values)

    metric = Metric(cat_bool_index = np.array([False,False,False]))
    dm = pdist(values,metric)
    print(dm)


    input('DONE')














    show_perturbed_digits = True

    mnist = pd.read_csv('../data/mnist/mnist_small.csv')
    population = mnist[[c for c in mnist.columns if c!='class']]
    population.columns = [n for n,_ in enumerate(population.columns)]
    cat_bool_index = np.array([False for _ in range(population.shape[1])])


    categorical = Categorical(population,cat_bool_index,courseness=100)
    # population = categorical(population)


    metric = Metric(cat_bool_index)



    avg_distances = {}


    filepath = '../data/mnist/mnist_medium.csv'
    mnist_dataset = MNISTDataset(filepath)
    dataloader = DataLoader(mnist_dataset, batch_size=1000, shuffle=False)

    for n,(batch,labels) in enumerate(dataloader):


        batch = pd.DataFrame(batch)
        batch = categorical(batch)
        # pop_value = np.mean(pdist(batch, metric))
        # print(pop_value)
        # print(labels)
        # print(len(set(labels)))

        # print('------------')


        values = {}
        for num in set(labels.numpy()):

            idx = (num==labels).numpy() 
            non_idx = (num!=labels).numpy()  


            this_batch = batch.loc[idx,:]  # all of the 1's
            other_batch = batch.loc[non_idx,:]



            intra = []
            for n in range(this_batch.shape[0]):
                row = this_batch.values[n,:]
                for m in range(this_batch.shape[0]):
                    if n!=m:
                        other_row = this_batch.values[m,:]
                        intra.append(metric(row,other_row))
            intra_avg = np.mean(intra)

            inter = []
            for n in range(this_batch.shape[0]):
                for m in range(other_batch.shape[0]):
                    row = this_batch.values[n,:]
                    other_row = other_batch.values[m,:]
                    inter.append(metric(row,other_row))
            inter_avg = np.mean(inter)

            values[num] = {'within': intra_avg,'without':inter_avg}
            print(num)




    avg_ratio = []
    for key,value in values.items():
        # print(key,value)

        within = values[key]['within']
        without = values[key]['without']
        avg_ratio.append(without/within)
        print(f'{key}:  {without/within:.6}')

    print(np.mean(avg_ratio))




            # value = np.mean(pdist(this_batch, metric))
            # print(num,value,value<pop_value)
            # # print(idx.values)

            # # input('dflkfj')








        # input('DONE')

        # categorical_batch = copy.deepcopy(batch)
        # categorical_batch = categorical(categorical_batch)



        # # if show_perturbed_digits:
        # cat_vector1 = categorical_batch.values[0,:]
        # cat_vector2 = categorical_batch.values[1,:]


        # # if show_perturbed_digits:
        # vector1 = batch.values[0,:]
        # vector2 = batch.values[1,:]


    
        # print(metric(cat_vector1,cat_vector2))
        # plot_mnist_digits_side_by_side(vector1, vector2)









































