# Data-Smart
Python coding of Data Smart (John W. Foreman) projects

As part of my journey into Data Science I am reading Data Smart. This book gives data for projects in excel. In order to up my python game and explore data science principles I will attempt to complete the projects by writing python programs for each of them. Should be fun and challenging. 

Janurary 26, 2017: Completed first project in DataSmart. Analysis of customer purchases from a wine import-export business. Hypethetical company tracks email newsleter offers and purchases by customer name. The goal was to analyse the purchases using K-means cluster analysis to group the customer base and generate customized emails to each segment. The offer information and transactions per customer were provided. The SKmeans.py program is able to read in this data and preform a spherical Kmeans clustering with user input cluster number. The user can then request analysis of which deals were prefered in each cluster. During the example analysis of WineKMC.xlsx data with spherical Kmeans clustering K=5, there are clear customer groups that prefer: 1) Offers from France, 2) Pinot Noir offers, 3) minimum quantity offers (6 kg), 4) Champagne or Prosecco offers, 5) large quantity offers (72 or 144kg).

May 9, 2017: Composted an updated script after a month at Galvanize Data Science program. Added in ability to analyze the number of clusters. 1) Plot the residual sum of squares for increasing number of clusters, 2) Plot the silhouette score for inscreasing number of clusters, 3) Plot the silhouette profile for increasing number of clusters. 

Two approaches to picking the appropriate value of clusters (K):
1) Pick the value of K which drastically reduces the value or RSS (residual sum of squares), aka Elbow Method
2) Pick the value of K which gives a mean silhouette score closest to 1
    A silhouette score of 1 would indicate clear seperation of clusters while a silhouette score or -1 would indicate clusters are complettly overlapping.

Example of running Kmeans Cluster analysis on WineKMC dataset:
python run Kmean_update.py
'''
Run recursive? yes or no
>yes
Input excel file name
>WineKMC
Input sheet name with trasaction data
>Transactions
Input sheet name with offer information
>OfferInformation
Would you like to see a range of possible clusters?  yes or no
>yes
Input maximum number of clusters to test
>9
For n_clusters = 2 the average silhouette score is 0.092.
For n_clusters = 3 the average silhouette score is 0.112.
For n_clusters = 4 the average silhouette score is 0.125.
For n_clusters = 5 the average silhouette score is 0.113.
For n_clusters = 6 the average silhouette score is 0.100.
For n_clusters = 7 the average silhouette score is 0.119.
For n_clusters = 8 the average silhouette score is 0.115.
Please see saved images to evaluate number of appropriate clusters
Have you chosen a number of clusters you wish to evaluate on?
>yes
Input a number of clusters to test
>4
Residual Sum of Squares for 4 clusters is 213.647846165
..... Output the offer information and number of transactions sorted by clusters
'''
Although I do not find a cluster number which gives clear seperation of descisive groups, I can still gain information from some groupings. My goal is to group users transactions such that I can send customized emails therefore, I must consider how many customized emails I am willing to craft. Evaluating upto 8 clusters I find that dividing the user transactions into 4 clusters yeilds the highest silhouette score. While the RSS curve does not indicate a clear elbow it does reveal an overall trend indicating that 4 clusters would yeild an acceptable RSS value. So after choosing 4 clusters I want to look into the transaction of each cluster add see what defining features are associated with each cluster. After evaluating the top 10 transactions for each cluster I find: 
Cluster 1 = Favors deals with large discount and maybe Champagne (?)
Cluster 2 = Favors deals with a small minimum quantatiy requirement
Cluster 3 = Favors deals with Prosecco or Champagne
Cluster 4 = Favors deals with Pinot Noir 

It should be easy to send out custom emails to these groups!
