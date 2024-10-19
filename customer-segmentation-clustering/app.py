import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import plotly.express as px

# Title of the app
st.title("Clustering Analysis with K-Means")

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    try:
        # Read the uploaded file
        data = pd.read_csv(uploaded_file)

        # Show the data
        st.write("Data Preview:")
        st.write(data)

        # Display the data types of the columns
        st.write("Data Types:")
        st.write(data.dtypes)

        # Keep only numerical columns
        numerical_data = data.select_dtypes(include=[np.number])

        # Allow users to select features
        selected_features = st.multiselect("Select features for clustering", numerical_data.columns.tolist())

        if selected_features:
            # Create a subset of the data based on selected features
            data_subset = numerical_data[selected_features]

            # Number of clusters
            k = st.slider("Select number of clusters", 1, 10, 3)

            if not data_subset.empty and len(selected_features) >= 2:
                # KMeans clustering
                kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
                data_subset['Cluster'] = kmeans.fit_predict(data_subset)

                # Show the results
                st.write("Clustered Data:")
                st.write(data_subset)

                # Calculate silhouette score
                silhouette_avg = silhouette_score(data_subset, data_subset['Cluster'])
                st.write(f"Silhouette Score: {silhouette_avg:.2f}")

                # Elbow method implementation
                sse = []
                k_range = range(1, 11)  # From 1 to 10 clusters
                for i in k_range:
                    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
                    kmeans.fit(data_subset)
                    sse.append(kmeans.inertia_)

                # Plotting the elbow method
                plt.figure(figsize=(10, 6))
                plt.plot(k_range, sse, marker='o')
                plt.title("Elbow Method For Optimal k")
                plt.xlabel("Number of clusters (k)")
                plt.ylabel("Sum of squared distances (SSE)")
                st.pyplot(plt)

                # Interactive cluster visualization using Plotly
                fig = px.scatter(data_subset, x=selected_features[0], y=selected_features[1], color='Cluster',
                                 title='K-Means Clustering with Plotly',
                                 hover_data=data_subset.columns)
                st.plotly_chart(fig)

                # Provide a download link for the clustered data
                csv = data_subset.to_csv(index=False).encode('utf-8')
                st.download_button("Download Clustered Data", data=csv, file_name='clustered_data.csv', mime='text/csv')

                # Cluster summary statistics
                cluster_summary = data_subset.groupby('Cluster').agg(['mean', 'median', 'std'])
                st.write("Cluster Summary Statistics:")
                st.write(cluster_summary)

            else:
                st.error("Please select at least two features for clustering.")
        else:
            st.warning("Please select features to continue.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info("Please upload a CSV file to get started.")
