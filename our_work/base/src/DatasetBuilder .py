from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd
import numpy as np 
from typing import List, Dict, Any
from datasets import load_dataset

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

  def load_as_dataframe(self, path: str, split: str = 'train', n_samples: int = 100):
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
      raw_examples = self.get_streaming_examples(path, split=split, n_samples=n_samples)
      
      # Μετατροπή των Pandas Rows σε Dicts (Η κρίσιμη διόρθωση)
      clean_dicts = [
          ex.to_dict('records')[0] if isinstance(ex, pd.DataFrame) else ex 
          for ex in raw_examples
      ]
      
      # Flattening και καθαρισμός ονομάτων στηλών
      df = pd.json_normalize(clean_dicts)
      df.columns = [c.split('.')[-1] for c in df.columns]
      
      print(f"Successfully created DataFrame with shape: {df.shape}")
      return df


  def get_streaming_examples(self, path: str, split: str = 'train', n_samples: int = 4) -> List[Dict[str, Any]]:
      """
      Connects to a Hugging Face stream and fetches a small batch of examples.
      
      Args:
          path (str): The HF dataset path (e.g., "MultimodalUniverse/gaia").
          split (str): Dataset split to use ('train', 'test').
          n_samples (int): How many objects to fetch.
          
      Returns:
          List[Dict]: A list of dictionaries ready for the Visualizer or DataFrame conversion.
      """
      print(f"Opening stream to {path}...")
      
      # 1. Load the stream (streaming=True ensures no massive download)
      ds = load_dataset(path, split=split, streaming=True)
      
      # 2. Set format to pandas to handle tabular data easily
      # Use "numpy" instead if you are strictly handling image tensors
      ds = ds.with_format("pandas")
      
      # 3. Pull n samples
      examples = []
      ds_iter = iter(ds)
      for _ in range(n_samples):
          try:
              # next(ds_iter) returns a single-row Pandas DataFrame/Dict
              examples.append(next(ds_iter))
          except StopIteration:
              break
            
      print(f"Successfully fetched {len(examples)} examples via stream.")
      return examples


  def register_subset(self, modality: str, subset_name: str, metadata: Dict[str, Any]):
      """
      Indexes a specific data subset into the factory, enabling it for later retrieval.
      
      Args:
          modality (str): The category of data. Supported: 'text', 'image', 
              'audio', 'video', or 'time_series'.
          subset_name (str): A unique identifier for the specific dataset 
              (e.g., 'hubble_ultra_deep' or 'pulsar_timings_v2').
          metadata (Dict[str, Any]): A dictionary of arbitrary key-value pairs 
              describing the subset. 
              *Note*: For 'time_series', metadata MUST include 'sampling_rate_hz' 
              and 'dimensions'.
      
      Raises:
          ValueError: If the provided 'modality' string is not found in the 
              supported registry keys.
          KeyError: If 'time_series' is registered without mandatory temporal 
              metadata (sampling_rate_hz, dimensions).
      
      Side Effects:
          Updates the internal 'registry' attribute by appending the new subset info.
      """
      # 1. Supported Modality Check
      if modality not in self.registry:
          raise ValueError(
              f"Unsupported modality: '{modality}'. "
              f"Supported types are: {list(self.registry.keys())}"
          )

      # 2. Strict Schema Validation for Time Series
      # Logic: We prevent registration if we lack the info to align signals later.
      if modality == "time_series":
          required_ts_keys = ["sampling_rate_hz", "dimensions"]
          for key in required_ts_keys:
              if key not in metadata:
                  raise KeyError(
                      f"Missing critical metadata '{key}' for time_series subset '{subset_name}'. "
                      "Temporal data requires sampling_rate_hz and dimensions for alignment."
                  )

      # 3. Create the Entry
      # We add a timestamp to track when the subset was indexed in your session.
      entry = {
          "name": subset_name,
          "registered_at": pd.Timestamp.now(),
          **metadata
      }

      # 4. Update Registry
      self.registry[modality].append(entry)
      print(f"Successfully indexed {subset_name} under {modality.upper()}.")


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


  def load_validated_dataset(self, path: str, n_wanted: int = 100, batch_size: int = 50, filter_names: list = None):
    """
    Streams Gaia data and applies registered class filters by name 
    until the target sample size is validated.
    """
    if filter_names is None:
        filter_names = ['_apply_star_filters', '_filter_completeness']

    final_df = pd.DataFrame()
    ds = load_dataset(path, split='train', streaming=True)
    ds_iter = iter(ds)

    while len(final_df) < n_wanted:
        # Fetch raw batch
        batch_raw = []
        for _ in range(batch_size):
            try:
                batch_raw.append(next(ds_iter))
            except StopIteration:
                break
        if not batch_raw: break

        # Use the internal logic of load_as_dataframe to flatten
        clean_dicts = [ex.to_dict('records')[0] if isinstance(ex, pd.DataFrame) else ex for ex in batch_raw]
        batch_df = pd.json_normalize(clean_dicts)
        batch_df.columns = [c.split('.')[-1] for c in batch_df.columns]
        
        # Apply registered filters by name
        validated_batch = batch_df
        for method_name in filter_names:
            filter_func = getattr(self, method_name)
            validated_batch = filter_func(validated_batch)

        # Build the final clean matrix X
        final_df = pd.concat([final_df, validated_batch], ignore_index=True)
        if len(final_df) >= n_wanted:
            final_df = final_df.iloc[:n_wanted].copy()
            break
            
    # Compute Absolute Magnitude (y) target
    final_df['abs_mag'] = final_df['phot_g_mean_mag'] + 5 + 5 * np.log10(final_df['parallax'] / 1000.0)
    return final_df
  
  def _filter_completeness(self, df: pd.DataFrame, required_vars: list = None) -> pd.DataFrame:
    """
    Ensures statistical integrity by removing rows with missing values 
    in critical feature columns.

    This method prevents 'ShapeMismatch' or 'NaN' propagation during 
    Tensor conversion and Matrix multiplications (e.g., Cosine Similarity).

    Parameters
    ----------
    df : pd.DataFrame
        The batch of Gaia data to be validated.
    required_vars : list, optional
        Columns that must be fully populated. Defaults to 
        ['phot_g_mean_mag', 'parallax', 'bp_rp'].
    """
    if required_vars is None:
        required_vars = ['phot_g_mean_mag', 'parallax', 'bp_rp']
    
    # Identify and drop incomplete stellar records
    clean_df = df.dropna(subset=required_vars)
    
    return clean_df

  def _apply_star_filters(self, df: pd.DataFrame) -> pd.DataFrame:
          """
          Applies astrophysical quality cuts to a DataFrame batch.
          
          Parameters
          ----------
          df : pd.DataFrame
              The raw flattened batch.

          Returns
          -------
          pd.DataFrame
              Cleaned data meeting RUWE and Parallax criteria[cite: 18].
          """
          # 1. RUWE Filter: Reject unreliable astrometric solutions
          if 'ruwe' in df.columns:
              df = df[df['ruwe'] < 1.4]
              
          # 2. Physical Parallax: Ensure distance-based calculations are valid
          if 'parallax' in df.columns:
              df = df[df['parallax'] > 0]

          # 3. Completeness: Drop rows missing features required for Absolute Magnitude [cite: 13]
          df = df.dropna(subset=['phot_g_mean_mag', 'parallax', 'bp_rp'])
          
          return df