import pandas as pd
import numpy as np
from spherecluster import SphericalKMeans

def read_pivot(filename,sheetname):
    """Read in a sheet from an excel file and create a pivot table as a Pandas DataFrame using the Offer # as the index and Customer Last Name as columns. 
    Aggregate function is count non-zero and the fill value is 0. Function should be returning 1 but is returning 2, I think because it 
    also counting the column number label. At the moment there is a bandaid to replace all instances of 2 with 1 but this needs to be addressed."""
    f=pd.read_excel(filename, sheetname=sheetname)
    piv=pd.pivot_table(f,index='Offer #', columns='Customer Last Name', aggfunc=np.count_nonzero, fill_value=0)
    piv=piv.replace(2, 1, regex=True)
    #I think the table is reporting 2 instead of one because it is also seeing the column number label. The replacement is a bandaid. I need to trouble shoot this later.
    return piv

def read_OI(filename, sheetname):
    """Read in sheet from excel and create a Pandas DataFrame using the Offer # as the index)
    """
    oi=pd.read_excel(filename, sheetname=sheetname)
    oi=oi.set_index('Offer #')
    return oi

def kmeans(piv,nclust):
    """ Run spherical kmeans analysis of transaction data and output data with the assigned cluster, distance to each cluster, and cluster centers"""
    data=piv.transpose()
    model=SphericalKMeans(n_clusters=nclust).fit(data)
    modeldist=model.transform(data)
    dclind=[]
    for i in range(nclust):
        dclind.append('Distance to Cluster %d' %i)
    md=pd.DataFrame(modeldist,columns=(dclind))
    data_out=data.copy()
    data_out['Cluster Class']=pd.Series(model.labels_,index=data_out.index)
    data_out=pd.concat([data_out, md.set_index(data_out.index[:len(md)])],axis=1)
    clind=[]
    for i in range(nclust):
        clind.append('Cluster %d' %i)
    clcent=pd.DataFrame(model.cluster_centers_,columns=range(1,33),index=(clind))
    data_out=pd.concat([clcent,data_out])
    return data_out

def dealsbycluster(analysis,dataout,cl):
    for i in range(cl-1):
        a=analysis[analysis['Cluster Class']==i].sum()
        dataout['SumDealsCluster%d'%i]=a[0:32]
    return dataout

def sortdbc(offerinfo_deals,cl):
    return offerinfo_deals.sort('SumDealsCluster%d'%cl,ascending=0)

print "Input excel file name"
excel=raw_input(">")
print "Input sheet name with trasaction data"
transh=raw_input(">")
print "Input sheet name with offer information"
oish=raw_input(">")
print "Input number of clusters for spherical K-means clustering"
cl=input(">")

piv =read_pivot(excel,transh)
offer_info=read_OI(excel,oish)
cluster=kmeans(piv,cl)
dataout=cluster.transpose()
final=pd.concat([offer_info,dataout],axis=1)
print final

print "Would you like to see the deals prefered by each cluster? yes or no" 
ans=raw_input(">")

if ans=="yes":
    print "Here are the deal counts for each cluster indexed with the offer information"
    dbc=dealsbycluster(cluster,offer_info,cl)
    print dbc
    print "Would you like to sort by deal sum? If yes, input cluster number you would like to sort by"
    anscl=input(">")
    sdbc=sortdbc(dbc,anscl)
    print sdbc
else:
    print "your loss"

