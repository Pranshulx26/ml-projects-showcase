import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Title of the app
st.title("Customer Segmentation using K-Means Clustering")

# Load the dataset
st.write("Loading dataset...")
data = pd.read_csv('data/Mall_Customers.csv')

# Display the dataset
st.write("Here is a glimpse of the dataset:")
st.dataframe(data.head())

# Select features for clustering
st.write("Selecting relevant features: 'Annual Income' and 'Spending Score'")
features = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Normalize the features (scaling)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Perform K-Means clustering
kmeans = KMeans(n_clusters=5)
data['Cluster'] = kmeans.fit_predict(scaled_features)

# Visualize the clusters
st.write("Visualizing the clusters:")
plt.figure(figsize=(10, 6))
sns.scatterplot(x=features['Annual Income (k$)'],
                y=features['Spending Score (1-100)'],
                hue=data['Cluster'], palette='Set2')
st.pyplot(plt)