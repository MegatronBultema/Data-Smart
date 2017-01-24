import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm


def read_pivot(filename,sheetname):
    """Read in a sheet from an excel file and create a pivot table as a Pandas DataFrame using the Offer # as the index and Customer Last Name as columns. 
    Aggregate function is count non-zero and the fill value is 0. Function should be returning 1 but is returning 2, I think because it 
    also counting the column number label. At the moment there is a bandaid to replace all instances of 2 with 1 but this needs to be addressed."""
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
    """Read in sheet from excel and create a Pandas DataFrame using the Offer # as the index)
    """
    import pandas as pd
    import numpy as np
    oi=pd.read_excel(filename, sheetname=sheetname)
    oi=oi.set_index('Offer #')
    print oi
    return oi

def concat(oi,rp):
    """Concatinate the Offer Information DataFrame with the pivot table
    """
    frames=(oi,piv)
    merge=pd.concat(frames, axis=1)
    return merge

def trans(concat):
    trans=concat.loc[1:,"Adams":]
    trans=trans.transpose()
    return trans

def kmeans(data,nclust):
    from sklearn.cluster import KMeans
    model=KMeans(n_clusters=nclust).fit(data)
    modeldist=model.transform(data)
    md=pd.DataFrame(modeldist,columns=('Distance to Cluster0', 'Distance to Cluster1','Distance to Cluster2', 'Distance to Cluster3'))
    data_out=data.copy()
    data_out['Cluster Class']=pd.Series(model.labels_,index=data_out.index)
    data_out=pd.concat([data_out, md.set_index(data_out.index[:len(md)])],axis=1)
    clcent=pd.DataFrame(model.cluster_centers_,columns=range(1,33),index=('Center 0','Center 1','Center 2','Center 3'))
    data_out=pd.concat([data_out,clcent])
    
    
