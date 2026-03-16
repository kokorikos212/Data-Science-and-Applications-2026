import json
import numpy as np

import os
import pandas as pd
from typing import List, Dict, Any, Optional

from datasets import load_dataset


class MultimodalDatasetFactory:
    """
    A utility factory to fetch, filter, and combine subsets of 
    multimodal data (Text, Image, Audio, Video).
    """

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

    def register_subset(self, modality: str, subset_name: str, metadata: Dict[str, Any]):
        """
        mess
        Validates and indexes a specific data subset into the factory registry.
        
        Args:
            modality (str): The category of data (e.g., 'image', 'text', 'time_series').
            subset_name (str): A unique string identifier for the dataset.
            metadata (Dict[str, Any]): Attributes describing the data. 
                For 'time_series', this dictionary MUST contain 'sampling_rate_hz' 
                (int/float) and 'dimensions' (int) to ensure temporal alignment.
        
        Raises:
            ValueError: If 'modality' is not supported.
            KeyError: If 'modality' is 'time_series' but mandatory temporal 
                metadata keys are missing.
        
        Logic:
            1. Validate the modality type against supported keys.
            2. If 'time_series', verify that 'sampling_rate_hz' and 'dimensions' 
               are provided in the metadata to allow for future resampling.
            3. Append the subset and its verified metadata to the internal registry.
        """
        # 1. Verification of modality support
        if modality not in self.registry:
            raise ValueError(f"Modality '{modality}' is not supported by the Universe Factory.")

        # 2. Strict Schema Validation for Time Series
        if modality == "time_series":
            required_ts_keys = ["sampling_rate_hz", "dimensions"]
            for key in required_ts_keys:
                if key not in metadata:
                    raise KeyError(
                        f"Missing critical metadata '{key}' for time_series subset '{subset_name}'. "
                        "Time series registration requires sampling_rate_hz and dimensions."
                    )

        # 3. Registration
        entry = {
            "name": subset_name,
            "registered_at": pd.Timestamp.now(),
            **metadata
        }
        self.registry[modality].append(entry)
        print(f"Successfully indexed {subset_name} [{modality.upper()}]")

    def fetch_modality_subset(self, modality: str, filters: Optional[Dict] = None) -> List[Dict]:
        """
        Queries the registry for specific datasets belonging to a modality, 
        optionally filtered by their metadata attributes.
        
        Args:
            modality (str): The target modality to search within.
            filters (Optional[Dict]): A dictionary of key-value pairs. Only subsets 
                whose metadata exactly matches every key-value pair in this 
                dictionary will be returned. If None, all subsets for the 
                modality are returned.
        
        Returns:
            List[Dict]: A list of metadata dictionaries representing the matching 
                subsets. Returns an empty list if no matches are found or if the 
                modality is empty.
        
        Example:
            If filters={'lang': 'fr'}, only subsets registered with 'fr' are returned.
        """
        subsets = self.registry.get(modality, [])
        
        if not filters:
            return subsets
        
        return [
            s for s in subsets 
            if all(s.get(k) == v for k, v in filters.items())
        ]

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

    def create_unified_generator(self, combined_df: pd.DataFrame):
        """
        Transforms a manifest DataFrame into a Python generator for memory-efficient 
        data streaming.
        
        Args:
            combined_df (pd.DataFrame): The output from 'combine_modalities' 
                containing the paths and modality labels.
        
        Yields:
            Dict[str, Any]: A dictionary representing a single data entry, 
                containing:
                - 'id': The name of the subset.
                - 'modality': The string label of the modality.
                - 'data_ptr': The resolved filesystem path to the asset.
        
        Performance Note:
            This is a lazy-loading mechanism. It allows the user to iterate 
            over millions of multimodal records without exceeding RAM limits.
        """
        for _, row in combined_df.iterrows():
            # In a real scenario, you'd yield the actual loaded bytes/tensors here
            yield {
                "id": row['subset'],
                "modality": row['modality'],
                "data_ptr": row['path']
            }
    def get_streaming_examples(self, path: str, split: str = 'train', n_samples: int = 4) -> List[Dict[str, Any]]:
        """
        Connects to a Hugging Face stream and fetches a small batch of examples.
        
        Logic:
            1. Initializes a streaming dataset (no full download).
            2. Formats the stream to return NumPy arrays (crucial for image tensors).
            3. Iterates through the first 'n' items and collects them into a list.
        
        Expertise Note: 
            Streaming is critical for the Multimodal Universe because a single 
            subset like 'legacysurvey' can exceed 100GB. This method ensures 
            we only consume bandwidth for the samples we actually visualize.
            
        Args:
            path (str): The HF dataset path (e.g., "MultimodalUniverse/legacysurvey").
            split (str): Dataset split to use ('train', 'test').
            n_samples (int): How many objects to fetch.
            
        Returns:
            List[Dict]: A list of dictionaries ready for the Visualizer.
        """
        print(f"Opening stream to {path}...")
        
        # 1. Load the stream
        ds = load_dataset(path, split=split, streaming=True)
        
        # 2. Set format to numpy to ensure 'rgb' and 'mask' keys are arrays
        ds = ds.with_format("numpy")
        
        # 3. Pull n samples
        examples = []
        # We use a loop with iter() to avoid loading the whole dataset
        ds_iter = iter(ds)
        for _ in range(n_samples):
            try:
                examples.append(next(ds_iter))
            except StopIteration:
                break
                
        print(f"Successfully fetched {len(examples)} examples via stream.")
        return examples


def print_structure(data_dict):

    # A helper to handle NumPy arrays so they don't flood the screen
    def serializer(obj):
        if isinstance(obj, np.ndarray):
            return f"NumPy Array (Shape: {obj.shape}, Dtype: {obj.dtype})"
        return str(obj)

    # Indent=4 creates the nested "JSON" look
    print(json.dumps(data_dict, indent=4, default=serializer))


