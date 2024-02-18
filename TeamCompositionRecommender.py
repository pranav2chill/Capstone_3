import pandas as pd
import os
from sklearn.cluster import KMeans  # KMeans for clustering
from sklearn.preprocessing import StandardScaler  # StandardScaler for feature scaling
from sklearn.metrics import silhouette_score  # silhouette_score for evaluating the quality of clusters
from sklearn.decomposition import PCA  # PCA for dimensionality reduction
from mpl_toolkits.mplot3d import Axes3D  # Axes3D for 3D plotting
import matplotlib.pyplot as plt
import warnings

# Set the OMP_NUM_THREADS environment variable to 1 (chatgpt helped with this)
os.environ['OMP_NUM_THREADS'] = '1'

class TeamCompositionRecommender:
    def __init__(self, data):
        """
        Initialize the TeamCompositionRecommender class.

        Parameters:
        - data (pandas.DataFrame): The input data containing team compositions and agent performance.
        """
        self.data = data
    
    def agent_performance(self):
        """
        Calculate the average performance of agents for each map.

        Prints the top 5 performing agents for each map.
        """
        for map_name in self.data['map'].unique():
            print(f"Map: {map_name}")
            map_data = self.data.loc[self.data['map'] == map_name].copy()  # Use .loc and .copy() --> (chatgpt helped with this)
            cols_to_convert = ['rating', 'acs', 'adr']  # Define the columns to convert to numeric

            # For each column in cols_to_convert, convert the column to numeric, replacing non-numeric values with NaN
            for col in cols_to_convert:
                map_data.loc[:, col] = pd.to_numeric(map_data[col], errors='coerce') 

            # Group the data by 'agent' and calculate the mean of numeric columns ['rating', 'acs', 'kill', 'assist', 'adr']
            agent_performance = map_data.groupby('agent').mean(numeric_only=True)[['rating', 'acs', 'kill', 'assist', 'adr']]  

            # Sort the agent_performance DataFrame by multiple columns in descending order
            top_agents = agent_performance.sort_values(by=['rating', 'acs', 'kill', 'assist', 'adr'], ascending=False)

            print(top_agents.head(5))  
            print("\n")  

    def cluster_teams(self, map_name):
        """
        Cluster team compositions based on agent selection.

        Prints the average composition for each cluster and visualizes the clusters in a 3D plot.
        """
        print(f"Map: {map_name}")
        map_data = self.data[self.data['map'] == map_name]
        
        # Group by game_id and team, and aggregate the agent data into a list
        team_compositions = map_data.groupby(['game_id', 'team'])['agent'].apply(list)
        
        # Convert the list of agents into a string so it can be used with get_dummies
        team_compositions = team_compositions.apply(lambda x: ' '.join(x))
        
        # Convert the string of agents into dummy variables
        team_compositions = team_compositions.str.get_dummies(' ')
        
        scaler = StandardScaler()  # Initialize a StandardScaler object

        # Fit the scaler to the team compositions and transform the data
        team_compositions_scaled = scaler.fit_transform(team_compositions)

        # Initialize a KMeans object with 3 clusters and 10 initializations
        kmeans = KMeans(n_clusters=3, n_init=10)

        # Fit the KMeans object to the scaled data and predict the clusters
        clusters = kmeans.fit_predict(team_compositions_scaled)
        team_compositions['cluster'] = clusters
        for cluster in range(3):
            print(f"Cluster {cluster}:")  # Print the cluster number

            # Print the mean values of the team compositions in the cluster, sorted in descending order
            print(team_compositions[team_compositions['cluster'] == cluster].mean().sort_values(ascending=False))

        print("\n") 

        # Reduce to 3 dimensions using PCA
        pca = PCA(n_components=3)
        principalComponents = pca.fit_transform(team_compositions_scaled)

        # Create a DataFrame with the three components and the cluster number
        pca_df = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2', 'principal component 3'])
        pca_df['Cluster'] = clusters

        # Plot the clusters
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(pca_df['principal component 1'], pca_df['principal component 2'], pca_df['principal component 3'], c=pca_df['Cluster'])

        plt.title(f'3D visualization of clusters for {map_name}')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')

        # Create a legend
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)

        plt.show()

    def plot_elbow_silhouette(self):
        """
        Plot the elbow method and silhouette method to determine the optimal number of clusters.

        The elbow method plots the distortion for different numbers of clusters.
        The silhouette method plots the silhouette score for different numbers of clusters.
        """
        team_compositions = self.data.groupby(['team', 'agent']).size().unstack(fill_value=0)
        scaler = StandardScaler()
        team_compositions_scaled = scaler.fit_transform(team_compositions)
        distortions = []
        silhouette_scores = [] 
        # For each number of clusters from 2 to 7
        for i in range(2, 8):
            # Initialize a KMeans object with i clusters and a fixed random state for reproducibility
            km = KMeans(n_clusters=i, random_state=0)

            # Fit the KMeans object to the scaled data and predict the clusters
            clusters = km.fit_predict(team_compositions_scaled)

            # Append the distortion (sum of squared distances to the closest centroid for all samples) to the list
            distortions.append(km.inertia_)

            # Append the silhouette score (measure of how close each sample in one cluster is to the samples in the neighboring clusters) to the list
            silhouette_scores.append(silhouette_score(team_compositions_scaled, clusters))

            # Calculate the average silhouette score for the current number of clusters
            silhouette_avg = silhouette_score(team_compositions_scaled, clusters)
            print(f"For n_clusters = {i}, the average silhouette_score is : {silhouette_avg}")
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(range(2, 8), distortions, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion')
        plt.title('Elbow Method')
        plt.subplot(1, 2, 2)
        plt.bar(range(2, 8), silhouette_scores)
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Method')
        plt.tight_layout()
        plt.show()