# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 17:21:50 2024

@author: akinola
"""

# import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np 
from sklearn.impute import KNNImputer
from mlxtend.preprocessing import minmax_scaling

#for interactive visualization
import chart_studio
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot


sns.set_style('darkgrid')
imputer = KNNImputer(n_neighbors=10)

#seting my credentials so i can move it to my plotly account
chart_studio.tools.set_credentials_file(username='akinola',api_key='fBtlKegjI3i0LyYOxWV5')
#custom modules
import visuals as vs
import checks as ch


#------reading in provided data------------
file_path = r'C:/Users/akinola/Documents/PROJECTS/bluechip projects/customer segmentation for marketing/Raw_data.xlsx'
Raw_data = pd.ExcelFile(file_path)

#----loading sheets into varioables-------------
Transactions = pd.read_excel(Raw_data,'Transactions')

NewCustomerList = pd.read_excel(Raw_data,'NewCustomerList',parse_dates=['DOB'])

CustomerDemographic = pd.read_excel(Raw_data,'CustomerDemographic')

CustomerAddress = pd.read_excel(Raw_data,'CustomerAddress')

#------------------------Data exploration------------------
''' key task:
        Understand the features, data types, and any missing values and
        visualize the data using appropriate plots (e.g., histograms, scatter plots, etc.).
'''
#-------------------checking for features------------
#-----checking for features and dtype--------
# df_check = ch.Musage(Transactions,NewCustomerList,CustomerDemographic,CustomerAddress)

#------------------checking for missing data----------------
null_check = ch.Null_Check(Transactions,NewCustomerList,CustomerDemographic,CustomerAddress)

Missing_integers= ch.Null_by_dtype(Transactions,NewCustomerList,CustomerDemographic,CustomerAddress,Type='int')

Missing_object= ch.Null_by_dtype(Transactions,NewCustomerList,CustomerDemographic,CustomerAddress,Type='object')

Missing_float= ch.Null_by_dtype(Transactions,NewCustomerList,CustomerDemographic,CustomerAddress,Type='float')

percentage_missing = (Transactions.isna().sum() / np.product(Transactions.shape)) * 100


#-----------------checking for duplicates----------------------
dup_check = ch.Dup_Tot(Transactions,NewCustomerList,CustomerDemographic,CustomerAddress)

#-----------------------checking for and inspecing unamed and unusable columns-------------
unamed_cols = ( NewCustomerList[NewCustomerList.columns[NewCustomerList.columns.str.contains('Unnamed')]])

#---------first to check if 'default' column is encoded before removing-------------
check = ch.detect_encoding(file_path, 'CustomerDemographic' , 'default')
#-------encoded but still not useful so i'll remove it---------
unusable = ( CustomerDemographic[CustomerDemographic.columns[CustomerDemographic.columns.str.contains('default')]])

#-----------------------checking for outliers-----------------------
#-----------using box plot from custom module--------------------
Transactions['year'] = Transactions['transaction_date'].dt.year

# plt.figure()
# data = vs.box(Transactions, 'standard_cost', 'year')
# dat = vs.box(Transactions,'product_first_sold_date','year')
# datt = vs.box(CustomerDemographic,'tenure','gender')


# chekcing for distribution
Transactions['order_status_rank'] = Transactions['order_status'].map({'Approved':1,'Cancelled':0})
# print(Transactions.order_status.value_counts())

# Group by order_status and computing sum of order_status_rank
grouped_data = Transactions.groupby('order_status')['order_status_rank'].sum().reset_index()

# Pivot to create a suitable structure for clustering
pivot_data = grouped_data.pivot(index='order_status', columns='order_status_rank', values='order_status_rank')

# Filling with NaN values if evident
pivot_data = pivot_data.fillna(0)

# Creating a dendrogram to check distribution
# plt.figure()
data2 = sns.clustermap(pivot_data, cmap='mako')

# print(data2)


#-------------------------data preprocessing----------------------------
'''key objective:
    Clean the data by handling missing values, outliers, and duplicates 
    and perform feature engineering if necessary (e.g., creating new features, scaling, etc.).
'''
#----------------------dropping unamed and unusable columns----------------------------------
NewCustomerList = NewCustomerList.drop(unamed_cols,axis=1)

CustomerDemographic = CustomerDemographic.drop(unusable,axis=1)


#----------------handling missing data---------------------
#------------------Impute missing values in Transactions--------------------------

#filing categories with most common occurence
Transactions['online_order'] = Transactions['online_order'].fillna(Transactions['online_order'].mode()[0])
for x in ['brand', 'product_line','product_class','product_size']:
    Transactions[x] = Transactions[x].fillna(Transactions[x].mode()[0])

#----filling missing numerics with KNN
Transactions['standard_cost'] = imputer.fit_transform(Transactions[['standard_cost']])

Transactions.product_first_sold_date = imputer.fit_transform(Transactions[['product_first_sold_date']] )

#--------------------------------correcting flawed column data---------------------------
CustomerDemographic['gender'] = CustomerDemographic['gender'].replace({'F':'Female','M':'Male','Femal':'Female','U':'Unisex'})
NewCustomerList['gender'] = NewCustomerList['gender'].replace({'F':'Female','M':'Male','Femal':'Female','U':'Unisex'})

Transactions['online_order'] = Transactions['online_order'].map({1:'TRUE',0:'FALSE'})


#----------------imputing missing values in NewCustomerList------------
#filling lastname with '--'
NewCustomerList['last_name'] = NewCustomerList['last_name'].fillna('--')

#fillling,job title,job_industry_category with unprovided
NewCustomerList['job_title'] = NewCustomerList['job_title'].fillna('unprovided')
NewCustomerList['job_industry_category'] = NewCustomerList['job_industry_category'].fillna('unprovided')


#--------------inputing data into CustomerDemographic-----------------

CustomerDemographic['last_name'] = CustomerDemographic['last_name'].fillna('--')


#standardizing dob
CustomerDemographic['DOB'] = pd.to_datetime(CustomerDemographic['DOB'],errors='coerce')
NewCustomerList['DOB'] = pd.to_datetime(NewCustomerList['DOB'],errors='coerce')
for df in [NewCustomerList,CustomerDemographic,Transactions]:
    for col in ['DOB']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
            df[col]=df[col].dt.date

# fillling,job title,job_industry_category with unprovided
CustomerDemographic['job_title'] = CustomerDemographic['job_title'].fillna('unprovided')
CustomerDemographic['job_industry_category'] = CustomerDemographic['job_industry_category'].fillna('unprovided')

#filling tenure with median due to the presence of outliers
CustomerDemographic['tenure'] = imputer.fit_transform(CustomerDemographic[['tenure']])


#----------------validating changes----------------

# info = ch.Musage(Transactions,NewCustomerList,CustomerDemographic,CustomerAddress)
null_check2 = ch.Null_Check(Transactions,NewCustomerList,CustomerDemographic,CustomerAddress)
# print(null_check2)
dtypenull = ch.Null_by_dtype(Transactions,NewCustomerList ,Type='int')
# print(dtypenull)

#---------------feature engineering---------------

#conserving memory by converting categorical data to category
# checking for object columns to decide which to convert
# print(ch.columns(Transactions,NewCustomerList,CustomerDemographic,CustomerAddress,dtype='object'))

#converting to  category datatype
for cols in ['online_order','order_status', 'brand', 'product_line','product_class', 'product_size',
             'gender', 'job_title','job_industry_category', 'wealth_segment', 'deceased_indicator',
             'owns_car', 'address', 'state', 'country','postcode']:
    if cols in Transactions.columns:
        Transactions[cols] = Transactions[cols].astype('category')
    if cols in NewCustomerList.columns:
        NewCustomerList[cols] = NewCustomerList[cols].astype('category')
    if cols in CustomerDemographic.columns:
        CustomerDemographic[cols] = CustomerDemographic[cols].astype('category')
    if cols in CustomerAddress.columns:
        CustomerAddress[cols] = CustomerAddress[cols].astype('category')

# cluster analysis 
cluster = NewCustomerList[['state','tenure','property_valuation']].copy()

#standardizing metrics to give equal weight
std = cluster[['tenure','property_valuation']].copy()
std=preprocessing.scale(std)

# using elbow method to find optimal clusters
wcss = []
for i in range(1,10):
    kmeans=KMeans(i,random_state=3)
    kmeans.fit(std)
    inert=kmeans.inertia_
    wcss.append(inert)
data = list(range(1,10))
# plt.plot(data,wcss,marker='|',markersize=15)
# plt.axline(4, color='r',linestyle='--')

kmeans = KMeans(4,random_state=3)
cluster['id'] = kmeans.fit_predict(std)
plt.figure()
plt.scatter(cluster['tenure'],cluster['property_valuation'],c=cluster['id'],cmap='rainbow')
plt.xlabel('tenure')
plt.ylabel('property valuation')

#using ploty to make it interactive
scat = go.Scatter(
    x=cluster['tenure'],
    y=cluster['property_valuation'],
    mode = 'markers',
    marker = (
        dict(
            size=12,
            color=cluster['id'],
            colorscale='rainbow'
            )),
    showlegend=True,
    text =cluster['id'],
    hoverinfo = 'text+x+y'
    )
lay = go.Layout(
    title='custeomer segment',
    xaxis = dict(title='tenure'),
    yaxis = dict(title='poperty_valuation')),
fig= go.Figure(data=[scat],layout=lay),
py.iplot(fig,filename='customer segments')

# cluster quality assesment using silouhette score
wcss = []
for i in range(2,10):
    kmeans = KMeans(i)
    cluster = kmeans.fit_predict(std)
    validation = silhouette_score(std,cluster)
    wcss.append(validation)
print(wcss)
