import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.preprocessing import StandardScaler

class ExploratoryAnalyzer:
  """
  Handles similarity matrix computations and descriptive statistics 
  for the stellar data matrix X. [cite: 15, 19, 23]
  """
  def __init__(self, data, feature_names=None):
    """
    Initializes the analyzer. If no feature names are provided, it assumes 
    the input is a DataFrame and extracts all numerical columns.

    Args:
        data (pd.DataFrame or np.ndarray): The stellar dataset to analyze.
        feature_names (list, optional): List of column names if data is an ndarray.
    """
    if isinstance(data, pd.DataFrame):
        self.df = data.select_dtypes(include=[np.number]).dropna()
        self.feature_names = self.df.columns.tolist()
        self.X = self.df.values
    else:
        self.X = data
        self.feature_names = feature_names if feature_names else [f"feat_{i}" for i in range(data.shape[1])]
        self.df = pd.DataFrame(data, columns=self.feature_names)
        
    sns.set_theme(style="whitegrid", palette="magma")

  def compute_similarity(self, n_subset: int = 50):
      """
      Computes pairwise cosine similarities for a subset of objects. [cite: 20]
      Visualizes the result as a heatmap. [cite: 21]
      """
      # Ensure we don't exceed the number of available samples
      subset_size = min(n_subset, self.X.shape[0])
      X_subset = self.X[:subset_size]
      
      # Calculate Cosine Similarity: S = (u · v) / (||u|| ||v||)
      sim_matrix = cosine_similarity(X_subset)
      
      plt.figure(figsize=(10, 8))
      sns.heatmap(sim_matrix, cmap='magma', annot=False)
      plt.title(f"3.2 Pairwise Cosine Similarity (n={subset_size})")
      plt.xlabel("Object Index")
      plt.ylabel("Object Index")
      plt.show()
      
      return sim_matrix

  def get_descriptive_stats(self):
        """
        Computes mean, median, variance, and standard deviation. [cite: 24]
        Identifies high variance and skewed distributions. [cite: 25]
        """
        # Calculate core metrics [cite: 24]
        stats = self.df.agg(['mean', 'median', 'var', 'std']).T
        
        # Identification of distribution shapes [cite: 25, 30]
        stats['is_skewed'] = (stats['mean'] - stats['median']).abs() > (0.5 * stats['std'])
        
        print("--- 3.3 Descriptive Statistics ---")
        print(stats)
        return stats

  def perform_pca(self, n_components: int = 2):
        """
        Transforms the internal feature space into principal components.
        Identifies which variables contribute most to the dataset's variance.

        Args:
            n_components (int): Number of orthogonal components to retain.

        Returns:
            tuple: (PCA object, Loadings DataFrame, Projected components).
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(X_scaled)
        
        loadings = pd.DataFrame(
            pca.components_.T, 
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=self.feature_names
        )
        
        print(f"PCA: PC1 explains {pca.explained_variance_ratio_[0]*100:.2f}% of variance.")
        return pca, loadings, pca_features

  