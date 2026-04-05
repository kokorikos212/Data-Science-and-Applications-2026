from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd
from typing import List, Dict

class MultimodalDatasetFactory:
    def __init__(self, base_path: str):
        """
        Initializes the factory with a root directory and an expanded modality registry.
        
        Args:
            base_path (str): The root directory for the dataset.
        """
        self.base_path = base_path
        # Added 'time_series' to the default registry to prevent KeyErrors
        self.registry = {
            "text": [],
            "image": [],
            "audio": [],
            "video": [],
            "time_series": [] 
        }

    def combine_modalities(self, selected_map: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Aggregates disparate subsets across different modalities into a single, 
        structured manifest for batch processing.
        ...
        """
        combined_data = []

        for modality, names in selected_map.items():
            # SAFETY CHECK: Ensure the modality exists in our registry before searching
            if modality not in self.registry:
                print(f"[Warning] Modality '{modality}' is not supported. Skipping.")
                continue
                
            for name in names:
                # Search the list of registered subsets for a name match
                subset_info = next(
                    (s for s in self.registry[modality] if s['name'] == name), 
                    None
                )
                
                if subset_info:
                    combined_data.append({
                        "subset": name,
                        "modality": modality,
                        "path": os.path.join(self.base_path, modality, name),
                        "meta": subset_info
                    })
                else:
                    print(f"[Warning] Subset '{name}' not found in '{modality}' registry.")
        
        return pd.DataFrame(combined_data)


class MultimodalUniverseFactory(MultimodalDatasetFactory):
    """
    A factory class for managing and joining multimodal astronomical datasets.

    This class provides utilities for loading streaming astronomical data into 
    structured DataFrames and performing spatial cross-matches using 
    celestial coordinates.

    Attributes
    ----------
    base_path : str
        The root directory or URI where the datasets are stored.
    primary_keys : list of str
        The standard column names used for data alignment (ra, dec, object_id).
    """

    def __init__(self, base_path: str):
        """
        Initialize the factory with a base data path.

        Parameters
        ----------
        base_path : str
            Path to the multimodal dataset repository.
        """
        super().__init__(base_path)
        # Standard columns used across the MMU for joining
        self.primary_keys = ["ra", "dec", "object_id"]

    def load_as_dataframe(self, path: str, subset: str, n_samples: int = 100):
        """
        Loads a subset of streaming data and flattens it into a clean pandas DataFrame.

        This helper handles the extraction of nested JSON-like structures (e.g., 
        'image.data' or 'tabular.value') and strips hierarchical prefixes from 
        column names for easier analysis.

        Parameters
        ----------
        path : str
            The specific dataset path or identifier.
        subset : str
            The data split to load (e.g., 'train', 'test', 'val').
        n_samples : int, optional
            Number of examples to stream from the source, by default 100.

        Returns
        -------
        pd.DataFrame
            A flattened DataFrame with simplified column headers.
        """
        examples = self.get_streaming_examples(path, n_samples=n_samples)
        # Flatten the 'weird' nested structure automatically
        df = pd.json_normalize(examples)
        # Clean up column names (remove 'image.', 'tabular.' prefixes)
        df.columns = [c.split('.')[-1] for c in df.columns]
        return df

    def spatial_join(self, df_left, df_right, radius_arcsec=1.0):
        """
        Perform a spatial cross-match between two DataFrames based on sky proximity.

        Uses `astropy.coordinates.SkyCoord` to find the nearest neighbor in the 
        right DataFrame for every entry in the left DataFrame within a specified 
        angular radius.

        Parameters
        ----------
        df_left : pd.DataFrame
            The primary DataFrame containing 'ra' and 'dec' columns in degrees.
        df_right : pd.DataFrame
            The secondary DataFrame to join against.
        radius_arcsec : float, optional
            The maximum angular separation for a valid match, by default 1.0 arcsec.

        Returns
        -------
        pd.DataFrame
            A concatenated DataFrame containing matched rows. Columns from the 
            secondary DataFrame are appended with the suffix '_secondary'.
        
        Notes
        -----
        This method performs a 1-to-1 mapping based on the nearest neighbor. 
        Rows in `df_left` without a neighbor within `radius_arcsec` are excluded.
        """
        coord_l = SkyCoord(ra=df_left['ra'].values*u.degree, dec=df_left['dec'].values*u.degree)
        coord_r = SkyCoord(ra=df_right['ra'].values*u.degree, dec=df_right['dec'].values*u.degree)
        
        # Find nearest matches
        idx, d2d, _ = coord_l.match_to_catalog_sky(coord_r)
        
        # Filter by radius
        mask = d2d < radius_arcsec * u.arcsec
        
        # Merge data
        matched = df_left[mask].reset_index(drop=True)
        right_data = df_right.iloc[idx[mask]].reset_index(drop=True)
        
        return pd.concat([matched, right_data.add_suffix('_secondary')], axis=1)


