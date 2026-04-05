import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

class ExploratoryAnalyzer:
    """
    Handles similarity matrix computations and descriptive statistics 
    for the stellar data matrix X. [cite: 15, 19, 23]
    """
    def __init__(self, X: np.ndarray, feature_names: list):
        self.X = X
        self.feature_names = feature_names
        # Represent each sample as a row vector [cite: 16]
        self.df = pd.DataFrame(X, columns=feature_names)

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