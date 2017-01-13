import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm


def read_csv(filename):
    with open(filename, 'rb') as f:
        reader= csv.reader(f)
        your_list=map(tuple,reader)
    
    print your_list
    print your_list[1]

    return your_list

def read_pivot(filename,sheetname):
    import pandas as pd
    import numpy as np
    f=pd.read_excel(filename, sheetname=sheetname)
    print f
    piv=pd.pivot_table(f,index='Offer #', columns='Customer Last Name', aggfunc=np.count_nonzero, fill_value=0)
    piv=piv.replace(2, 1, regex=True)
    #I think the table is reporting 2 instead of one because it is also seeing the column number label. The replacement is a bandaid. I need to trouble shoot this later.
    print piv
    return piv

def read_OI(filename, sheetname):
    import pandas as pd
    import numpy as np
    oi=pd.read_excel(filename, sheetname=sheetname)
    oi=oi.set_index('Offer #')
    print oi
    return oi

def concat(oi,rp):
    frames=(oi,rp)
    merge=pd.concat(frames, axis=1)
    return merge

def trans(concat):
    trans=concat.loc[1:,"Adams":]
    trans=trans.transpose()
    return trans

def kmeans(trans,nclust):
    from sklearn.cluster import KMeans
    model=KMeans(n_clusters=nclust)
    return (model.fit(trans),model.fit_transform(trans))
    print model.labels_
