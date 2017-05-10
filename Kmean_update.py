'''
Updated K-means analysis of wine purchasing history.
Goal: segemt customers to recieve customized emails.
'''

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score


def clean_data(filename, sheetname):
    fname = filename+'.xlsx'
    f=pd.read_excel(fname, sheetname=sheetname)
    # duplicate the Offer # column so have an unreferenced column to aggregate
    f['Offer#_count'] = f['Offer #']
    piv = f.pivot_table(index = 'Offer #', columns = 'Customer Last Name', values = 'Offer#_count', aggfunc = np.count_nonzero, fill_value=0)
    return piv

def read_offerinfo(filename, sheetname):
    """Read in sheet from excel and create a Pandas DataFrame using the Offer # as the index)
    """
    fname = filename+'.xlsx'
    oi=pd.read_excel(fname, sheetname=sheetname)
    oi=oi.set_index('Offer #')
    return oi

def kmean_score(piv, nclust):
    data=piv.transpose()
    km = KMeans(n_clusters = nclust, init = 'random', n_init = 10, max_iter = 300, n_jobs = -1) #look at hyperparameters
    km.fit(data)
    rss = -km.score(data)
    return rss

def fit_kmeans(piv, nclust):
    data=piv.transpose()
    km = KMeans(n_clusters = nclust, init = 'random', n_init = 10, max_iter = 300, n_jobs = -1, random_state = 10) #look at hyperparameters
    km.fit(data)
    return km

def plot_kmeans(piv, max_num_clusters):
    scores = [kmean_score(piv,i) for i in range(1, max_num_clusters)]
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(range(1,max_num_clusters), scores, 'o--')
    ax.set_xlabel('K')
    ax.set_ylabel('RSS')
    ax.set_title('RSS versus K')
    plt.savefig('ResidualSumSquares_vs_K.png')

def vis_clusters(piv, nclust):
    kmfit = fit_kmeans(piv, nclust)
    data_out = piv.transpose()
    data_out['Cluster Class']=pd.Series(model.labels_,index=data_out.index)
    return data_out

def cluster_and_plot(piv, n_clusters):
    X=piv.transpose()
    # Create a subplot with 1 row and 2 columns
    fig, ax1 = plt.subplots()
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = fit_kmeans(piv, n_clusters)
    cluster_labels = clusterer.predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters = {0} the average silhouette score is {1:0.3f}.".\
        format(n_clusters, silhouette_avg))

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhoutte score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])


    plt.savefig('Silhouette_{}'.format(n_clusters))

def get_silhouette_score(piv, n_clusters):
    kmfit = fit_kmeans(piv, n_clusters)
    sil_avg = silhouette_score(piv.transpose(), kmfit.labels_)
    return sil_avg

def plot_silhouette_score(piv, max_num_clusters):
    sil_scores = [get_silhouette_score(piv,i) for i in range(2,max_num_clusters)]
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(range(2,max_num_clusters), sil_scores, 'o--', c = 'g')
    ax.set_xlabel('K')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score vs K')
    plt.savefig('Silhouette_Score_vs_K.png')

def return_analysis(piv, number_nclust):
    kmfit = fit_kmeans(piv, number_nclust)
    data_out = piv.transpose()
    data_out['Cluster Class']=pd.Series(kmfit.labels_,index=data_out.index)
    #clind=[]
    #for i in range(number_nclust):
    #    clind.append('Cluster %d' %i)
    #clcent=pd.DataFrame(kmfit.cluster_centers_,columns=range(1,33),index=(clind))
    #data_out=pd.concat([clcent,data_out])
    return data_out

def dealsbycluster(analysis,dataout,cl):
    for i in range(cl):
        a=analysis[analysis['Cluster Class']==i].sum()
        dataout['Number_of_Deals_Cluster{}'.format(i+1)]=a[0:32]
    return dataout

def sortdbc(offerinfo_deals,cl):
    return offerinfo_deals.sort('Number_of_Deals_Cluster{}'.format(cl+1),ascending=0)

if __name__ == '__main__':
    print('Run recursive? yes or no')
    ans1 = raw_input('>')
    if ans1 == 'yes':
        print "Input excel file name"
        excel=raw_input(">")
        print "Input sheet name with trasaction data"
        transh=raw_input(">")
        print "Input sheet name with offer information"
        oish=raw_input(">")
        offer_info = read_offerinfo(excel, oish)
        print('Would you like to see a range of possible clusters?  yes or no')
        ans=raw_input(">")
        if ans=="yes":
            print('Input maximum number of clusters to test')
            max_nclusters = input('>')
            piv =clean_data(excel,transh)
            for i in range(2,max_nclusters):
                cluster_and_plot(piv, i)
            plot_silhouette_score(piv, max_nclusters)
            print('Please see saved images to evaluate number of appropriate clusters')
            print('Have you chosen a number of clusters you wish to evaluate on?')
            ans2 = raw_input(">")
            if ans2 == 'yes':
                print('Input a number of clusters to test')
                number_nclust = input('>')
                piv =clean_data(excel,transh)
                print('Residual Sum of Squares for {} clusters is {}'.format(number_nclust, kmean_score(piv, number_nclust)))
                data_out = return_analysis(piv, number_nclust)
                dbc=dealsbycluster(data_out,offer_info,number_nclust)
                print('Results with {} clusters'.format(number_nclust))
                print(dbc)
                sorted_oi = []
                for i in range(number_nclust):
                    sdbc = sortdbc(dbc, i)
                    sorted_oi.append(sdbc)
                print('\n\nTob 10 transactions ordered by the 1st cluster')
                print(sorted_oi[0].iloc[:11,:])
                print('\n\nTob 10 transactions ordered by the 2nd cluster')
                print(sorted_oi[1].iloc[:11,:])
                print('\n\nTob 10 transactions ordered by the 3rd cluster')
                print(sorted_oi[2].iloc[:11,:])
                print('\n\nTob 10 transactions ordered by the 4th cluster')
                print(sorted_oi[3].iloc[:11,:])
            else:
                print('Maybe you would like to try a different method. Goodbye.')

        else:
            print('Input a number of clusters to test')
            number_nclust = input('>')
            piv =clean_data(excel,transh)
            print('Residual Sum of Squares for {} clusters is {}'.format(number_nclust, kmean_score(piv, number_nclust)))
            data_out = return_analysis(piv, number_nclust)
            dbc=dealsbycluster(data_out,offer_info,number_nclust)
            print('Results with {} clusters'.format(number_nclust))
            print(dbc)
            sorted_oi = []
            for i in range(number_nclust):
                sdbc = sortdbc(dbc, i)
                sorted_oi.append(sdbc)
            print('\n\nTob 10 transactions ordered by the 1st cluster')
            print(sorted_oi[0].iloc[:11,:])
            print('\n\nTob 10 transactions ordered by the 2nd cluster')
            print(sorted_oi[1].iloc[:11,:])
            print('\n\nTob 10 transactions ordered by the 3rd cluster')
            print(sorted_oi[2].iloc[:11,:])
            print('\n\nTob 10 transactions ordered by the 4th cluster')
            print(sorted_oi[3].iloc[:11,:])


    else:
        transactions = clean_data('WineKMC', 'Transactions')
        score = kmean_score(transactions, 3)
        plot_kmeans(transactions, 10)
        for i in range(2,6):
            cluster_and_plot(transactions, i)
        plot_silhouette_score(transactions, 6)
        data_out = return_analysis(transactions, 4)
        offer_info = read_offerinfo('WineKMC', 'OfferInformation')
        dbc=dealsbycluster(data_out,offer_info,4)
        print('Results with 4 clusters')
        #print(dbc)
        sdbc_0 = sortdbc(dbc, 0)
        sdbc_1 = sortdbc(dbc, 1)
        sdbc_2 = sortdbc(dbc, 2)
        sdbc_3 = sortdbc(dbc, 3)
        print('\n\nTob 10 transactions ordered by the 1st cluster')
        print(sdbc_0.iloc[:11,:])
        print('\n\nTob 10 transactions ordered by the 2nd cluster')
        print(sdbc_1.iloc[:11,:])
        print('\n\nTob 10 transactions ordered by the 3rd cluster')
        print(sdbc_2.iloc[:11,:])
        print('\n\nTob 10 transactions ordered by the 4th cluster')
        print(sdbc_3.iloc[:11,:])

    '''
        piv =clean_data(excel,transh)
        offer_info=read_offerinfo(excel,oish)
        cluster=kmeans(piv,cl)
        dataout=cluster.transpose()
        final=pd.concat([offer_info,dataout],axis=1)
        print final
    '''
