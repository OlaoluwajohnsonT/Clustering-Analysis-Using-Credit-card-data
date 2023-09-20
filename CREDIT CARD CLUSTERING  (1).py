#!/usr/bin/env python
# coding: utf-8

# # CLUSTERING ANALYSIS USING CREDIT CARD DATA

# In[33]:


from PIL import Image

im = Image.open("C:\\Users\TOJMARK LTD\\DATA SCIENCE PROJECT\\Apple stock prediction project\\MYIMAGE.png")
display(im)


# Hi, I'm a data enthusiast with a knack for making sense of numbers. I thrive on turning data into practical insights that drive business decisions. My background in marketing gives me an edge in understanding customer behavior. I love experimenting with data, using statistical tools and machine learning to find hidden patterns. My goal is to become a data scientist, supercharging my data skills. My journey is guided by a passion for ethical data practices and a strong belief in data's power to transform businesses.

# # INTRODUCTION
# 
# In this project, I will perform clustering analysis on a dataset containing information about credit card customers. The dataset consists of 8,950 entries, each representing a customer, and includes 18 columns with various features such as balance, purchase history, credit limits, and more.
# 
# My objective is to employ clustering techniques to uncover hidden patterns and segment customers based on their credit card usage behavior. By grouping customers with similar characteristics together, I aim to provide valuable insights for decision-making in the financial industry.
# 
# Through exploratory data analysis (EDA), data preprocessing, and clustering algorithms like K-means and hierarchical clustering, I will identify distinct customer segments. These segments can help financial institutions tailor their services and marketing strategies to meet the specific needs of different customer groups.
# 
# By the end of this analysis, I intend to provide a clear understanding of how customers are distributed within these clusters and offer insights into the characteristics that define each group. This information can be instrumental in making data-driven decisions, such as optimizing credit offerings, managing risk, and enhancing customer experiences.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# In[2]:


data = pd.read_csv("C:\\Users\\TOJMARK LTD\\CREDITCARDgeneral.csv")
data.head()


# In[3]:


data.describe()


# # DATA EXPLORATION

# In[4]:


data.info()


# In[5]:


data.notnull().sum()


# In[6]:


#Let check the distribution of the 'TENURE' column
print(data['TENURE'].value_counts())


# In[7]:


#Let calculate the average balance for customers who made at least one purchase:
avg_balance_with_purchases = data[data['PURCHASES'] > 0]['BALANCE'].mean()
print("Average balance for customers with purchases:", avg_balance_with_purchases)


# In[8]:


#the customers with the highest and lowest credit limits
max_credit_limit = data['CREDIT_LIMIT'].max()
min_credit_limit = data['CREDIT_LIMIT'].min()
print("Customer with the highest credit limit:", data[data['CREDIT_LIMIT'] == max_credit_limit]['CUST_ID'].values[0])
print("Customer with the lowest credit limit:", data[data['CREDIT_LIMIT'] == min_credit_limit]['CUST_ID'].values[0])


# In[9]:


#the correlation between 'BALANCE' and 'PAYMENTS'
avg_full_payment = data['PRC_FULL_PAYMENT'].mean()
avg_full_payment


# In[10]:


#The proportion of customers who made one-off purchases versus installments
oneoff_vs_installments = data['ONEOFF_PURCHASES'].sum() / data['INSTALLMENTS_PURCHASES'].sum()
oneoff_vs_installments


# In[11]:


# the average number of purchases per transaction
avg_purchases_per_transaction = data['PURCHASES'].sum() / data['PURCHASES_TRX'].sum()
avg_purchases_per_transaction


# In[12]:


#let check if there are any missing values in the 'MINIMUM_PAYMENTS' column
missing_values = data['MINIMUM_PAYMENTS'].isnull().sum()
missing_values


# In[13]:


# Remove rows with null values
data = data.dropna()

# Check the number of rows after removing null values
len(data)


# In[14]:


import matplotlib.pyplot as plt

plt.hist(data['BALANCE'], bins=20)
plt.xlabel('Balance')
plt.ylabel('Frequency')
plt.title('Histogram of Balance')
plt.show()


# In[15]:


plt.boxplot(data['PAYMENTS'])
plt.ylabel('Payments')
plt.title('Box plot of Payments')
plt.show()


# In[16]:


plt.scatter(data['PURCHASES'], data['CASH_ADVANCE'])
plt.xlabel('Purchases')
plt.ylabel('Cash Advance')
plt.title('Scatter plot of Purchases vs. Cash Advance')
plt.show()


# In[17]:


tenure_counts = data['TENURE'].value_counts()
plt.bar(tenure_counts.index, tenure_counts.values)
plt.xlabel('Tenure')
plt.ylabel('Number of Customers')
plt.title('Number of Customers in Each Tenure Category')
plt.show()


# In[18]:


oneoff_sum = data['ONEOFF_PURCHASES'].sum()
installments_sum = data['INSTALLMENTS_PURCHASES'].sum()
proportions = [oneoff_sum, installments_sum]
labels = ['One-off Purchases', 'Installments']
plt.pie(proportions, labels=labels, autopct='%1.1f%%')
plt.title('Proportion of One-off Purchases to Installments')
plt.show()


# In[19]:


import seaborn as sns

data_numeric = data.drop(columns=['CUST_ID'])

# Remove rows with null values
data_cleaned = data_numeric.dropna()

# Compute correlation matrix
correlation_matrix = data_cleaned.corr()

# Create the correlation heatmap
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Heatmap')
plt.show()


# In[20]:


plt.hist(data['CREDIT_LIMIT'], bins=[0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000])
plt.xlabel('Credit Limit')
plt.ylabel('Frequency')
plt.title('Histogram of Credit Limit')
plt.show()


# In[21]:


selected_columns = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'PAYMENTS']
sns.pairplot(data[selected_columns])
plt.suptitle('Pair Plot for Selected Numerical Columns', y=1.02)
plt.show()


# In[22]:


credit_limit_bins = [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000]
data['CREDIT_LIMIT_BIN'] = pd.cut(data['CREDIT_LIMIT'], bins=credit_limit_bins)
credit_limit_counts = data['CREDIT_LIMIT_BIN'].value_counts().sort_index()
credit_limit_counts.plot(kind='bar')
plt.xlabel('Credit Limit Range')
plt.ylabel('Number of Customers')
plt.title('Number of Customers in Different Credit Limit Ranges')
plt.xticks(rotation=45)
plt.show()


# In[23]:


avg_full_payment_by_limit = data.groupby('CREDIT_LIMIT_BIN')['PRC_FULL_PAYMENT'].mean()
avg_full_payment_by_limit.plot(kind='bar')
plt.xlabel('Credit Limit Range')
plt.ylabel('Average Percentage of Full Payment')
plt.title('Percentage of Full Payment by Credit Limit Range')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.show()


# In[24]:


# Drop the 'CUST_ID' and 'CREDIT_LIMIT' columns
model_data = data.drop(columns=['CUST_ID', 'CREDIT_LIMIT_BIN'])
model_data


# In[25]:


# Compute correlation matrix
model_data.dropna()
model_data = model_data.apply(pd.to_numeric)

correlation_matrix = model_data.corr()
# Create the correlation heatmap
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Heatmap')
plt.show()


# In[26]:


cols = model_data.columns.tolist()
for col in cols:
    model_data[col] = np.log(1 + model_data[col])
plt.figure(figsize=(15,20))
for i, col in enumerate(cols):
    ax = plt.subplot(6, 2, i+1)
    sns.kdeplot(model_data[col], ax=ax)
plt.show()


# # Principal Component Analysis

# In[27]:


# Standardize the data before applying PCA
scaler = StandardScaler()
data_scaled = scaler.fit_transform(model_data)

# Perform PCA to reduce dimensions
pca = PCA(n_components=2)  # You can choose the number of components you want to keep
data_pca = pca.fit_transform(data_scaled)


# # Determine the No Of Clustering

# In[28]:


#Determine the optimal number of clusters (k) using the elbow method
inertia = []
for k in range(1, 11):  # Trying k values from 1 to 10
   kmeans = KMeans(n_clusters=k, random_state=42)
   kmeans.fit(data_pca)
   inertia.append(kmeans.inertia_)

# Plot the elbow method graph
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method to Find Optimal k')
plt.show()


# I selected 2 as my number of K

# # K-MEANS CLUSTERING ANALYSIS

# In[29]:


# Apply k-means clustering with k=2
K = 2
kmeans = KMeans(n_clusters=K, random_state=42)
clusters = kmeans.fit_predict(data_pca)

# Add the cluster labels to the DataFrame
model_data['Cluster'] = clusters

# Print the first few rows of the DataFrame with cluster labels
model_data.head()


# In[40]:


# Visualize the clusters using scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(data_pca[model_data['Cluster'] == 0][:, 0], data_pca[model_data['Cluster'] == 0][:, 1], label='Cluster 1', s=50)
plt.scatter(data_pca[model_data['Cluster'] == 1][:, 0], data_pca[model_data['Cluster'] == 1][:, 1], label='Cluster 2', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='X', color='red', s=200, label='Centroids')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-Means Clustering Results')
plt.legend()
plt.show()


# In[41]:


plt.scatter(model_data['BALANCE'], model_data['PURCHASES'], c=model_data['Cluster'], cmap='viridis')
plt.xlabel('Balance')
plt.ylabel('Purchases')
plt.title('Scatter plot of BALANCE vs. PURCHASES (Colored by Cluster)')
plt.colorbar(label='Cluster')
plt.show()


# In[42]:


plt.figure(figsize=(8, 6))
sns.boxplot(x=model_data['Cluster'], y=model_data['BALANCE'])
plt.xlabel('Cluster')
plt.ylabel('Balance')
plt.title('Box plot of BALANCE by Cluster')
plt.show()


# In[43]:


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(model_data['BALANCE'], model_data['PURCHASES'], model_data['PAYMENTS'], c=model_data['Cluster'], cmap='viridis')
ax.set_xlabel('Balance')
ax.set_ylabel('Purchases')
ax.set_zlabel('Payments')
ax.set_title('3D Scatter plot of BALANCE, PURCHASES, and PAYMENTS (Colored by Cluster)')
plt.show()

# Selected features for distribution visualization
selected_features = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'PAYMENTS', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES']

# Distribution visualization for each cluster
for cluster_num in range(k):
    cluster_data = model_data[model_data['Cluster'] == cluster_num]
    plt.figure(figsize=(10, 6))
    for feature in selected_features:
        sns.histplot(cluster_data[feature], label=feature, kde=True)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Features in Cluster {cluster_num}')
    plt.legend()
    plt.show()
# In[31]:


plt.figure(figsize=(8, 6))
sns.histplot(data=model_data, x='PURCHASES', hue='Cluster', kde=True)
plt.xlabel('Purchases')
plt.title('Distribution of PURCHASES based on Clusters')
plt.show()


# In[32]:


plt.figure(figsize=(8, 6))
sns.violinplot(data=model_data, x='Cluster', y='PAYMENTS')
plt.xlabel('Cluster')
plt.ylabel('Payments')
plt.title('Distribution of PAYMENTS based on Clusters')
plt.show()


# In[47]:


plt.figure(figsize=(8, 6))
sns.boxplot(data=model_data, x='Cluster', y='ONEOFF_PURCHASES')
plt.xlabel('Cluster')
plt.ylabel('One-off Purchases')
plt.title('Distribution of ONEOFF_PURCHASES based on Clusters')
plt.show()


# In[48]:


plt.figure(figsize=(8, 6))
sns.histplot(data=model_data, x='CASH_ADVANCE', hue='Cluster', kde=True)
plt.xlabel('Cash Advance')
plt.title('Distribution of CASH_ADVANCE based on Clusters')
plt.show()


# In[49]:


plt.figure(figsize=(8, 6))
sns.boxplot(data=model_data, x='Cluster', y='INSTALLMENTS_PURCHASES')
plt.xlabel('Cluster')
plt.ylabel('Installments Purchases')
plt.title('Distribution of INSTALLMENTS_PURCHASES based on Clusters')
plt.show()


# # Thank You

# In[ ]:




